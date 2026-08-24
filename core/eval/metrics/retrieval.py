from __future__ import annotations

import math
from typing import Sequence

from ..types import Case, MetricResult, Prediction
from .base import Metric, pair_by_id, summarize

DEFAULT_FIELD = "retrieved_doc_ids"


def _gold(case: Case) -> set[str]:
    return set(case.expected.get("gold_doc_ids", []))


def _retrieved(pred: Prediction, k: int, field: str) -> list[str]:
    return list(pred.output.get(field, []))[:k]


class RecallAtK(Metric):
    """|retrieved_k intersect gold| / |gold|. Cases with no gold docs (e.g. unanswerable
    questions) are skipped -- they measure refusal behaviour, not retrieval.

    Reads `field` from Prediction.output -- by default the pre-rerank candidate set, so
    the same class scores a different stage of the pipeline (see
    `app/evaluation/rag_adapter.py`'s "retrieved_doc_ids" vs "reranked_doc_ids") just by
    naming a different field, without a second implementation to keep in sync.
    """
    higher_is_better = True

    def __init__(self, k: int, *, field: str = DEFAULT_FIELD, name: str | None = None):
        self.k = k
        self.field = field
        self.name = name or f"recall@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            hit = len(gold & set(_retrieved(pred, self.k, self.field)))
            per_case.append(hit / len(gold))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class PrecisionAtK(Metric):
    """|retrieved_k intersect gold| / k."""
    higher_is_better = True

    def __init__(self, k: int, *, field: str = DEFAULT_FIELD, name: str | None = None):
        self.k = k
        self.field = field
        self.name = name or f"precision@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = _retrieved(pred, self.k, self.field)
            if not retrieved:
                per_case.append(0.0)
                continue
            hit = len(gold & set(retrieved))
            per_case.append(hit / len(retrieved))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class HitRateAtK(Metric):
    """Fraction of cases with >=1 gold doc in the top k."""
    higher_is_better = True

    def __init__(self, k: int, *, field: str = DEFAULT_FIELD, name: str | None = None):
        self.k = k
        self.field = field
        self.name = name or f"hit_rate@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            hit = 1.0 if gold & set(_retrieved(pred, self.k, self.field)) else 0.0
            per_case.append(hit)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class MRR(Metric):
    """Mean of 1 / rank of the first gold doc (0 if none is retrieved)."""
    higher_is_better = True

    def __init__(self, *, field: str = DEFAULT_FIELD, name: str = "mrr"):
        self.field = field
        self.name = name

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = pred.output.get(self.field, [])
            rank = next((i + 1 for i, rid in enumerate(retrieved) if rid in gold), None)
            per_case.append(1.0 / rank if rank else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class NDCGAtK(Metric):
    """DCG@k / IDCG@k with binary gains."""
    higher_is_better = True

    def __init__(self, k: int, *, field: str = DEFAULT_FIELD, name: str | None = None):
        self.k = k
        self.field = field
        self.name = name or f"ndcg@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = _retrieved(pred, self.k, self.field)
            gains = [1.0 if rid in gold else 0.0 for rid in retrieved]
            dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
            ideal_hits = min(len(gold), self.k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            per_case.append(dcg / idcg if idcg else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


def retrieval_metrics(ks: Sequence[int] = (1, 3, 5, 10), *, field: str = DEFAULT_FIELD,
                      suffix: str = "") -> list[Metric]:
    """The standard retrieval battery at every k in `ks`, plus a single MRR.

    `field`/`suffix` let the same battery be instantiated twice against two different
    pipeline stages (e.g. pre-rerank vs post-rerank) without name collisions in the
    scorecard.
    """
    out: list[Metric] = [MRR(field=field, name=f"mrr{suffix}")]
    for k in ks:
        out += [RecallAtK(k, field=field, name=f"recall@{k}{suffix}"),
               PrecisionAtK(k, field=field, name=f"precision@{k}{suffix}"),
               HitRateAtK(k, field=field, name=f"hit_rate@{k}{suffix}"),
               NDCGAtK(k, field=field, name=f"ndcg@{k}{suffix}")]
    return out
