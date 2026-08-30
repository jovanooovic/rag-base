from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from ..store.access import AccessScope
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
                 vector_weight: float = 1.0, keyword_weight: float = 1.0, reranker=None,
                 recency_weight: float = 0.0, recency_half_life_days: float = 365.0,
                 now: date | None = None):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.reranker = reranker
        self.recency_weight = recency_weight
        self.recency_half_life_days = recency_half_life_days
        # Injectable so an eval run is reproducible. With datetime.now() baked
        # in, `make eval-accept` would drift a little every day and a rerun
        # would never reproduce its own baseline.
        self.now = now

    def retrieve(self, query: str, *, top_k: int | None = None,
                 where: dict[str, Any] | None = None, trace=None,
                 access: "AccessScope | None" = None) -> list[ScoredChunk]:
        top_k = top_k or self.top_k
        span = trace.span("retrieve", query=query[:200]) if trace else _null_span()
        with span:
            qvec = self.embeddings.embed([query])[0]
            # Both legs, always. Scoping one and not the other scopes
            # neither -- the unscoped leg puts the row into the fused set.
            vector_hits = self.store.search(qvec, k=self.fetch_k, where=where, access=access)
            keyword_hits = self.store.keyword_search(query, k=self.fetch_k, where=where,
                                                     access=access)
            fused = reciprocal_rank_fusion(
                [vector_hits, keyword_hits],
                weights=[self.vector_weight, self.keyword_weight],
            )
            if self.reranker is not None and fused:
                # Keep the reranker's own cut wide when recency is on, or a
                # fresh document the reranker ranked 6th could never be
                # promoted into a top-5 it was already excluded from.
                cut = self.fetch_k if self.recency_weight > 0 else top_k
                fused = self.reranker.rerank(query, fused[: self.fetch_k], top_k=cut)
            if self.recency_weight > 0 and fused:
                fused = apply_recency(fused, weight=self.recency_weight,
                                      half_life_days=self.recency_half_life_days,
                                      now=self.now)
            return fused[:top_k]


def apply_recency(hits: Sequence[ScoredChunk], *, weight: float,
                  half_life_days: float = 365.0, now: date | None = None) -> list[ScoredChunk]:
    """Re-rank by relevance blended with document age.

    Applied *after* reranking, not before, and that ordering is the whole
    reason this works. LLMReranker sets `score = rubric + fused * 0.001`, so a
    fused score survives only as a tiebreaker three orders of magnitude down --
    anything mixed in before reranking is multiplied into that same noise and
    changes nothing. Both shipped configs enable the reranker, so a recency
    weight applied earlier would be a no-op in production while looking fine in
    a unit test that skips the reranker.

    Confidence is deliberately left untouched: it answers "does this passage
    answer the question", and a document does not become a worse answer by
    being older. Only the ordering changes.
    """
    if weight <= 0 or not hits:
        return list(hits)

    today = now or date.today()
    # Relevance comes from signals["confidence"], not from min-max normalising
    # .score across this result set. Min-max was the first attempt and it is
    # wrong in a way that only shows up with few hits: with two candidates it
    # always maps them to exactly 0.0 and 1.0, so a trivial rubric gap (9.0 vs
    # 8.5) is amplified into the maximum possible difference and recency can
    # never move anything. Confidence is already a real 0-1 relevance scale and
    # does not depend on what else happens to be in the list.
    fallback = _minmax(hits)

    for h in hits:
        raw = h.chunk.metadata.get("effective_date")
        age_days = None
        if isinstance(raw, str):
            try:
                age_days = max((today - date.fromisoformat(raw)).days, 0)
            except ValueError:
                age_days = None
        # An undated document keeps a neutral freshness rather than being
        # treated as infinitely old: sinking documents for missing metadata
        # would silently bury a corpus that simply has no dates.
        freshness = 0.5 if age_days is None else 0.5 ** (age_days / half_life_days)
        h.signals["recency"] = round(freshness, 4)
        relevance = h.signals.get("confidence", fallback[h.chunk.chunk_id])
        h.score = (1 - weight) * relevance + weight * freshness

    return sorted(hits, key=lambda h: h.score, reverse=True)


def _minmax(hits: Sequence[ScoredChunk]) -> dict[str, float]:
    """Last-resort relevance scale for hits that carry no confidence signal --
    a custom retriever, say. Documented as a fallback because of the two-hit
    degeneracy described in apply_recency."""
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    return {h.chunk.chunk_id: (0.5 if span == 0 else (h.score - lo) / span) for h in hits}


class _null_span:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
