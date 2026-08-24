from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .answer.generate import Answer, answer_question
from .answer.guardrails import check_answer, redact
from .core.config import Settings
from .core.providers import Message, build_embeddings, build_llm
from .core.trace import Trace
from .ingest.pipeline import Ingestor
from .retrieve.hybrid import HybridRetriever
from .retrieve.query import rewrite
from .retrieve.rerank import LLMReranker
from .store.base import ScoredChunk
from .store.sqlite_store import SQLiteStore


def build_store(settings: Settings):
    """One place to swap storage. Client says 'we already run Postgres'? Change here."""
    backend = settings.extra.get("store_backend", "sqlite")
    if backend == "sqlite":
        return SQLiteStore(Path(settings.data_dir) / "index.db")
    if backend == "pgvector":
        from .store.pgvector_store import PgVectorStore
        return PgVectorStore(settings.extra["postgres_dsn"], dim=settings.embedding_dim)
    raise ValueError(f"unknown store_backend {backend!r}")


@dataclass
class RAGResult:
    question: str
    answer: Answer
    hits: list[ScoredChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    needs_clarification: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.text,
            "answered": self.answer.answered,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "needs_clarification": self.needs_clarification,
            "unsupported": self.answer.unsupported,
            "queries": self.queries,
            "citations": [
                {"n": c.number, "source": c.source, "heading": c.heading_path,
                 "chunk_id": c.chunk_id, "excerpt": c.text[:300]}
                for c in self.answer.citations
            ],
            "retrieved": [
                {"chunk_id": h.chunk.chunk_id, "source": h.chunk.source,
                 "score": round(h.score, 4), "signals": {k: round(v, 4) for k, v in h.signals.items()}}
                for h in self.hits
            ],
            "usage": self.answer.usage,
            "trace": self.trace,
        }


class RAGPipeline:
    """The whole system, assembled. This is what `api.py` and `cli.py` both call."""

    def __init__(self, settings: Settings | None = None, *, store=None, trace: Trace | None = None):
        self.settings = settings or Settings.load()
        self.trace = trace or Trace(enabled=self.settings.trace_enabled,
                                    out_dir=self.settings.trace_dir)
        self.llm = build_llm(self.settings, self.trace)
        self.embeddings = build_embeddings(self.settings)
        self.store = store or build_store(self.settings)

        ex = self.settings.extra
        reranker = LLMReranker(self.llm) if ex.get("use_reranker", True) else None
        self.retriever = HybridRetriever(
            self.store, self.embeddings,
            top_k=int(ex.get("top_k", 5)),
            fetch_k=int(ex.get("fetch_k", 20)),
            vector_weight=float(ex.get("vector_weight", 1.0)),
            keyword_weight=float(ex.get("keyword_weight", 1.0)),
            reranker=reranker,
        )
        self.ingestor = Ingestor(self.store, self.settings, self.embeddings)

    # -- write side -----------------------------------------------------
    def ingest(self, path: str | Path, **kw: Any):
        with self.trace.span("ingest", path=str(path)):
            return self.ingestor.ingest_path(path, **kw)

    # -- read side ------------------------------------------------------
    def ask(self, question: str, *, history: Sequence[Message] | None = None,
            where: dict[str, Any] | None = None, top_k: int | None = None) -> RAGResult:
        ex = self.settings.extra
        with self.trace.span("ask", question=question[:300]):
            queries = [question]
            if ex.get("rewrite_queries", True) and history:
                with self.trace.span("rewrite"):
                    queries = rewrite(self.llm, question, history)

            merged: dict[str, ScoredChunk] = {}
            for q in queries:
                for hit in self.retriever.retrieve(q, top_k=top_k, where=where, trace=self.trace):
                    prev = merged.get(hit.chunk.chunk_id)
                    if prev is None or hit.score > prev.score:
                        merged[hit.chunk.chunk_id] = hit
            hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[
                : (top_k or self.retriever.top_k)]

            answer = answer_question(self.llm, question, hits, history=history,
                                     max_context_chars=int(ex.get("max_context_chars", 8000)),
                                     trace=self.trace)
            gate = check_answer(
                answer, hits,
                min_top_score=ex.get("min_top_score"),
                require_citations=bool(ex.get("require_citations", True)),
            )
            final = gate.answer or answer
            if ex.get("redact_pii", False):
                final.text = redact(final.text)

            result = RAGResult(question=question, answer=final, hits=hits, queries=queries,
                               refused=not gate.ok, refusal_reason=gate.reason,
                               needs_clarification=final.needs_clarification,
                               trace={"run_id": self.trace.run_id, **self.trace.counters})
        self.trace.save()
        return result
