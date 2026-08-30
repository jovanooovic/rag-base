from __future__ import annotations

from typing import Any, Sequence

from ..store.base import ScoredChunk, VectorStore


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists by reciprocal rank.

    RRF rather than score normalisation because cosine similarity and BM25 live
    on incomparable scales, and any min-max normalisation you write will get
    destabilised the first time one leg returns a single result. RRF only reads
    positions, so it cannot be destabilised that way.
    """
    weights = list(weights or [1.0] * len(rankings))
    acc: dict[str, ScoredChunk] = {}
    fused: dict[str, float] = {}
    for leg, ranking in enumerate(rankings):
        w = weights[leg] if leg < len(weights) else 1.0
        for rank, sc in enumerate(ranking, start=1):
            cid = sc.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
            if cid not in acc:
                acc[cid] = ScoredChunk(sc.chunk, 0.0, {})
            acc[cid].signals.update(sc.signals)
            acc[cid].signals[f"rank_leg{leg}"] = float(rank)

    # Best achievable fused score: every *contributing* leg ranks it first.
    # Counted by weight rather than by len(rankings) because a leg weighted 0
    # contributes nothing -- the dense-only and bm25-only rows of the ablation
    # matrix run exactly that way, and dividing by 2 there would cap confidence
    # at 0.5 for a unanimous top hit.
    best = sum(w for w in weights if w > 0) / (k + 1)
    for cid, score in fused.items():
        acc[cid].score = score
        acc[cid].signals["confidence"] = round(score / best, 4) if best else 0.0
    return sorted(acc.values(), key=lambda s: s.score, reverse=True)


class HybridRetriever:
    """Vector + BM25, fused, optionally reranked.

    Default configuration is the one that wins most client bake-offs: retrieve
    wide (fetch_k) on both legs, fuse, then let a reranker cut to top_k. Cheap
    recall first, expensive precision second.
    """

    def __init__(self, store: VectorStore, embeddings, *, top_k: int = 5, fetch_k: int = 20,
                 vector_weight: float = 1.0, keyword_weight: float = 1.0, reranker=None):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.reranker = reranker

    def retrieve(self, query: str, *, top_k: int | None = None,
                 where: dict[str, Any] | None = None, trace=None) -> list[ScoredChunk]:
        top_k = top_k or self.top_k
        span = trace.span("retrieve", query=query[:200]) if trace else _null_span()
        with span:
            qvec = self.embeddings.embed([query])[0]
            vector_hits = self.store.search(qvec, k=self.fetch_k, where=where)
            keyword_hits = self.store.keyword_search(query, k=self.fetch_k, where=where)
            fused = reciprocal_rank_fusion(
                [vector_hits, keyword_hits],
                weights=[self.vector_weight, self.keyword_weight],
            )
            if self.reranker is not None and fused:
                fused = self.reranker.rerank(query, fused[: self.fetch_k], top_k=top_k)
            return fused[:top_k]


class _null_span:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
