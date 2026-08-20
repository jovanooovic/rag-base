"""Ablation matrix: chunking x retrieval mode x reranker, scored on retrieval quality
against the golden set.

Answers "which chunking strategy should we use" and "is the reranker worth buying"
with a table instead of an opinion -- the single highest-value artifact in this repo,
per the README's own decision table.

Scope: this measures RETRIEVAL only (recall/precision/mrr/ndcg@k), not full answer
generation. Chunking, retrieval mode, and reranker choice are all retrieval-side
decisions; answer quality is a separate question `make eval` already covers. That also
means this harness makes no LLM calls at all (CrossEncoderReranker is a local model,
not a chat model), so it is cheap to run against any embedding provider.

    python -m eval.ablations                          # full matrix
    python -m eval.ablations --chunking fixed-512 semantic --retrieval hybrid-rrf
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import Settings
from app.core.providers import build_embeddings
from app.evaluation.rag_adapter import load_golden_cases
from app.ingest.chunking import CHUNKING_STRATEGIES
from app.ingest.loaders import load_path
from app.ingest.pipeline import Ingestor
from app.retrieve.hybrid import HybridRetriever
from app.store.sqlite_store import SQLiteStore
from core.eval.metrics.retrieval import retrieval_metrics
from core.eval.runner import run_suite
from core.eval.types import Case, Prediction

RETRIEVAL_MODES = ("dense-only", "bm25-only", "hybrid-rrf")
RERANKER_MODES = ("off", "cross-encoder")

_RETRIEVAL_WEIGHTS = {"dense-only": (1.0, 0.0), "bm25-only": (0.0, 1.0), "hybrid-rrf": (1.0, 1.0)}


def _build_reranker(mode: str) -> tuple[Any, str | None]:
    """Returns (reranker, skip_reason). A missing optional dependency skips just
    this row rather than failing the whole matrix."""
    if mode == "off":
        return None, None
    try:
        from app.retrieve.rerank import CrossEncoderReranker
        return CrossEncoderReranker(), None
    except ImportError:
        return None, "sentence-transformers not installed (pip install sentence-transformers)"


def _retrieved_doc_ids(hits: list[Any]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.source not in seen:
            seen.append(h.chunk.source)
    return seen


def run_matrix(settings: Settings, *, corpus_path: str = "data/sample",
               golden_path: str = "eval/data/golden.jsonl",
               chunkings: tuple[str, ...] = CHUNKING_STRATEGIES,
               retrievals: tuple[str, ...] = RETRIEVAL_MODES,
               rerankers: tuple[str, ...] = RERANKER_MODES,
               top_k: int = 10, fetch_k: int = 20) -> list[dict[str, Any]]:
    """One ingestion per chunking strategy (shared across all retrieval x reranker
    combinations for that strategy, since retrieval mode and reranker don't affect
    what got indexed), then every combination is scored against the same cases."""
    cases = load_golden_cases(golden_path)
    metrics = retrieval_metrics(ks=(1, 5, 10))
    rows: list[dict[str, Any]] = []

    for chunking in chunkings:
        with TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "index.db")
            try:
                embeddings = build_embeddings(settings)
                ingestor = Ingestor(store, settings, embeddings, state_path=Path(tmp) / "state.json")
                report = ingestor.ingest_documents(load_path(corpus_path), chunk_kwargs={"strategy": chunking})

                for retrieval in retrievals:
                    vector_weight, keyword_weight = _RETRIEVAL_WEIGHTS[retrieval]
                    for reranker_mode in rerankers:
                        reranker, skip_reason = _build_reranker(reranker_mode)
                        row: dict[str, Any] = {"chunking": chunking, "retrieval": retrieval,
                                               "reranker": reranker_mode,
                                               "n_chunks": report.total_chunks_in_index}
                        if skip_reason:
                            row["skipped"] = skip_reason
                            rows.append(row)
                            continue

                        retriever = HybridRetriever(store, embeddings, top_k=top_k, fetch_k=fetch_k,
                                                   vector_weight=vector_weight,
                                                   keyword_weight=keyword_weight, reranker=reranker)

                        def system(case: Case, retriever: HybridRetriever = retriever) -> Prediction:
                            hits = retriever.retrieve(case.input["question"], top_k=top_k)
                            return Prediction(case_id=case.id,
                                              output={"retrieved_doc_ids": _retrieved_doc_ids(hits)})

                        predictions = asyncio.run(run_suite(cases, system, concurrency=4))
                        for m in metrics:
                            row[m.name] = m.compute(cases, predictions).value
                        rows.append(row)
            finally:
                # Windows locks open file handles, so the temp dir cannot be deleted
                # on __exit__ unless the sqlite connection is closed first.
                store.close()

    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["chunking", "retrieval", "reranker", "recall@5", "recall@10", "mrr", "ndcg@5"]
    lines = ["# Ablation matrix", "", "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers)]
    for r in rows:
        if r.get("skipped"):
            cells = [r["chunking"], r["retrieval"], r["reranker"], f"_skipped: {r['skipped']}_", "", "", ""]
        else:
            cells = [r["chunking"], r["retrieval"], r["reranker"],
                    f"{r.get('recall@5', 0):.4f}", f"{r.get('recall@10', 0):.4f}",
                    f"{r.get('mrr', 0):.4f}", f"{r.get('ndcg@5', 0):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default="data/sample")
    ap.add_argument("--golden", default="eval/data/golden.jsonl")
    ap.add_argument("--out", default="eval/report")
    ap.add_argument("--chunking", nargs="*", default=list(CHUNKING_STRATEGIES), choices=CHUNKING_STRATEGIES)
    ap.add_argument("--retrieval", nargs="*", default=list(RETRIEVAL_MODES), choices=RETRIEVAL_MODES)
    ap.add_argument("--reranker", nargs="*", default=list(RERANKER_MODES), choices=RERANKER_MODES)
    args = ap.parse_args(argv)

    settings = Settings.load(args.config)
    rows = run_matrix(settings, corpus_path=args.corpus, golden_path=args.golden,
                      chunkings=tuple(args.chunking), retrievals=tuple(args.retrieval),
                      rerankers=tuple(args.reranker))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.json").write_text(json.dumps(rows, indent=2))
    md = render_markdown(rows)
    (out / "ablations.md").write_text(md)

    print(md)
    print(f"\nwrote {out}/ablations.md and {out}/ablations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
