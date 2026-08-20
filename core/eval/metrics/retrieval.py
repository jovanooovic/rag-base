from __future__ import annotations

import math
from typing import Sequence

from ..types import Case, MetricResult, Prediction
from .base import Metric, pair_by_id, summarize


def _gold(case: Case) -> set[str]:
    return set(case.expected.get("gold_doc_ids", []))


def _retrieved(pred: Prediction, k: int) -> list[str]:
    return list(pred.output.get("retrieved_doc_ids", []))[:k]


class RecallAtK(Metric):
    """|retrieved_k intersect gold| / |gold|. Cases with no gold docs (e.g. unanswerable
    questions) are skipped -- they measure refusal behaviour, not retrieval."""
    higher_is_better = True

    def __init__(self, k: int):
        self.k = k
        self.name = f"recall@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            hit = len(gold & set(_retrieved(pred, self.k)))
            per_case.append(hit / len(gold))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class PrecisionAtK(Metric):
    """|retrieved_k intersect gold| / k."""
    higher_is_better = True

    def __init__(self, k: int):
        self.k = k
        self.name = f"precision@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = _retrieved(pred, self.k)
            if not retrieved:
                per_case.append(0.0)
                continue
            hit = len(gold & set(retrieved))
            per_case.append(hit / len(retrieved))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class HitRateAtK(Metric):
    """Fraction of cases with >=1 gold doc in the top k."""
    higher_is_better = True

    def __init__(self, k: int):
        self.k = k
        self.name = f"hit_rate@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            hit = 1.0 if gold & set(_retrieved(pred, self.k)) else 0.0
            per_case.append(hit)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class MRR(Metric):
    """Mean of 1 / rank of the first gold doc (0 if none is retrieved)."""
    name = "mrr"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = pred.output.get("retrieved_doc_ids", [])
            rank = next((i + 1 for i, rid in enumerate(retrieved) if rid in gold), None)
            per_case.append(1.0 / rank if rank else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class NDCGAtK(Metric):
    """DCG@k / IDCG@k with binary gains."""
    higher_is_better = True

    def __init__(self, k: int):
        self.k = k
        self.name = f"ndcg@{k}"

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            gold = _gold(case)
            if not gold:
                continue
            retrieved = _retrieved(pred, self.k)
            gains = [1.0 if rid in gold else 0.0 for rid in retrieved]
            dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
            ideal_hits = min(len(gold), self.k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            per_case.append(dcg / idcg if idcg else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


def retrieval_metrics(ks: Sequence[int] = (1, 3, 5, 10)) -> list[Metric]:
    """The standard retrieval battery at every k in `ks`, plus a single MRR."""
    out: list[Metric] = [MRR()]
    for k in ks:
        out += [RecallAtK(k), PrecisionAtK(k), HitRateAtK(k), NDCGAtK(k)]
    return out
