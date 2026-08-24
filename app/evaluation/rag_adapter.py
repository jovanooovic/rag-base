from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from core.eval.judge import Judge
from core.eval.metrics.agentic import CostPerTask
from core.eval.metrics.base import Metric
from core.eval.metrics.generation import (
    AnswerCorrectness,
    CitationPrecision,
    CitationRecall,
    ClarificationRate,
    Faithfulness,
    RefusalAccuracy,
)
from core.eval.metrics.operational import LatencyPercentile
from core.eval.metrics.retrieval import retrieval_metrics
from core.eval.types import Case, Prediction

from ..answer.generate import build_context
from ..core.config import Settings
from ..pipeline import RAGPipeline
from ..retrieve.hybrid import HybridRetriever
from ..store.base import ScoredChunk

# How many chunks the retrieval-only diagnostic pass asks for, so recall/precision/ndcg
# at k up to 10 have room to surface enough distinct documents. This is deliberately
# separate from the pipeline's configured top_k (which governs what a real user's
# answer is actually generated from) so widening it for retrieval metrics never changes
# the system's real generation behaviour -- see `run_case` below.
EVAL_RETRIEVAL_K = 20


def load_golden_cases(path: str | Path) -> list[Case]:
    """Read eval/data/golden.jsonl into core.eval Cases.

    `gold_answer` is dropped from `expected` for unanswerable and ambiguous cases:
    there is no real answer text to grade a refusal or a clarifying question against
    (the ambiguous slice's `gold_answer` describes what a good clarification *would*
    cover, not a candidate answer), so leaving it out lets AnswerCorrectness skip those
    cases automatically instead of scoring prose against an unrelated rubric.
    `gold_doc_ids` stays for every type, ambiguous included -- retrieval quality is a
    separate question from whether the system asked instead of guessing, and dropping
    it here would silently shrink the retrieval metrics' n by the whole ambiguous slice.
    """
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cases = []
    for r in rows:
        expected: dict[str, Any] = {}
        if r.get("gold_doc_ids"):
            expected["gold_doc_ids"] = r["gold_doc_ids"]
        if r.get("type") not in ("unanswerable", "ambiguous") and r.get("gold_answer"):
            expected["gold_answer"] = r["gold_answer"]
        cases.append(Case(
            id=r["id"], input={"question": r["question"]}, expected=expected,
            tags=tuple(r.get("tags", [])),
            metadata={"type": r.get("type", ""), "difficulty": r.get("difficulty", "")},
        ))
    return cases


def _retrieved_doc_ids(hits: list[ScoredChunk]) -> list[str]:
    """Doc-level ranking, deduped by first occurrence, from a chunk-level hit list --
    the golden set's gold_doc_ids are document paths, not chunk ids."""
    seen: list[str] = []
    for h in hits:
        if h.chunk.source not in seen:
            seen.append(h.chunk.source)
    return seen


def _span_ms(spans: Any, name: str) -> float:
    return sum(s.duration_ms for s in spans if s.name == name)


def build_rag_system(settings: Settings | None = None,
                     config_path: str | None = None) -> Callable[[Case], Prediction]:
    """One fresh RAGPipeline per case -- no shared mutable state (store connection,
    trace, LLM client), so this is safe to call concurrently from core.eval's thread
    pool. Mirrors the pattern the original hand-rolled harness used, for the same
    reason: per-case isolation of the cost/call budget and the trace.
    """
    base_settings = settings or Settings.load(config_path)
    # An eval run legitimately makes many calls; the per-run guard is sized for a
    # single user query, so raise it here rather than in the shipped config.
    base_settings = replace(base_settings, max_llm_calls_per_run=10_000)

    def run_case(case: Case) -> Prediction:
        pipeline = RAGPipeline(base_settings)
        question = case.input["question"]

        # Retrieval-only probe at a wide k, with the reranker deliberately left out:
        # recall@k should measure whether hybrid retrieval's candidate set contains the
        # gold document at all, not the reranker's job of ordering a small top-k for
        # display. Sharing the pipeline's store/embeddings keeps this free of any LLM
        # call, so it costs nothing and leaves no trace/cost footprint on the pipeline.
        probe = HybridRetriever(
            pipeline.store, pipeline.embeddings, top_k=EVAL_RETRIEVAL_K,
            fetch_k=max(EVAL_RETRIEVAL_K, pipeline.retriever.fetch_k),
            vector_weight=pipeline.retriever.vector_weight,
            keyword_weight=pipeline.retriever.keyword_weight, reranker=None,
        )
        wide_hits = probe.retrieve(question, top_k=EVAL_RETRIEVAL_K)

        started = time.time()
        result = pipeline.ask(question)
        latency_ms = round((time.time() - started) * 1000, 1)

        context, _ = build_context(result.hits,
                                   max_chars=int(pipeline.settings.extra.get("max_context_chars", 8000)))
        cited_sources = sorted({c.source for c in result.answer.citations
                               if c.number in result.answer.used_citations})

        output = {
            "answer_text": result.answer.text,
            "retrieved_doc_ids": _retrieved_doc_ids(wide_hits),
            # What the pipeline's own top_k actually ranked after reranking -- the
            # "retrieved_doc_ids" battery above is deliberately pre-rerank (see the
            # docstring on EVAL_RETRIEVAL_K); this is its post-rerank counterpart, i.e.
            # what a real user's answer was actually generated from.
            "reranked_doc_ids": _retrieved_doc_ids(result.hits),
            "citations": cited_sources,
            "retrieved_context": context,
            "refused": result.refused,
            "asked_clarification": result.needs_clarification,
        }
        return Prediction(
            case_id=case.id, output=output, latency_ms=latency_ms,
            cost_usd=float(pipeline.trace.counters.get("cost_usd", 0.0)),
            trace={"retrieve_ms": _span_ms(pipeline.trace.spans, "retrieve"),
                  "answer_ms": _span_ms(pipeline.trace.spans, "answer")},
        )

    return run_case


def build_metrics(settings: Settings | None = None, config_path: str | None = None,
                  judge_model: str | None = None) -> list[Metric]:
    s = settings or Settings.load(config_path)
    # The judge's Trace is shared across every case's correctness + faithfulness calls
    # (2 metrics x up to 3 votes x every case in the suite), so the per-run guard sized
    # for a single user query would trip well before the suite finishes.
    s = replace(s, max_llm_calls_per_run=10_000)
    judge = Judge(s, model=judge_model or s.llm_model)
    return [
        *retrieval_metrics(ks=(1, 3, 5, 10)),
        # Post-rerank counterpart, capped at 5: result.hits never exceeds the
        # pipeline's configured top_k, so a @10 slot here would just repeat @5's cases
        # padded with nothing. This is the number closer to "what did the user actually
        # get shown at rank 1", as opposed to the pre-rerank battery above.
        *retrieval_metrics(ks=(1, 3, 5), field="reranked_doc_ids", suffix="_reranked"),
        CitationPrecision(),
        CitationRecall(),
        RefusalAccuracy(),
        ClarificationRate(),
        AnswerCorrectness(judge),
        Faithfulness(judge),
        CostPerTask(),
        LatencyPercentile(),
        LatencyPercentile(trace_field="retrieve_ms", name="latency_retrieve_ms"),
        LatencyPercentile(trace_field="answer_ms", name="latency_answer_ms"),
    ]


def build_adapter() -> tuple[Callable[[], Callable[[Case], Prediction]], Callable[[], list[Metric]],
                             Callable[[str | Path], list[Case]]]:
    """Entry point for `python -m core.eval run --adapter app.evaluation.rag_adapter:build_adapter`.

    Both settings and judge model come from project.config.json auto-discovery (and its
    usual APP_<FIELD> env overrides) rather than CLI flags, matching how every other
    entry point in this repo resolves its config.
    """
    def build_system() -> Callable[[Case], Prediction]:
        return build_rag_system()

    def metrics_factory() -> list[Metric]:
        return build_metrics()

    return build_system, metrics_factory, load_golden_cases
