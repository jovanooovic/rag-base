from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..stats import bootstrap_ci
from ..types import Case, MetricResult, Prediction


class Metric(ABC):
    """A scoring function over a case/prediction pair.

    Subclassing this and nothing else is enough for a new metric to show up in every
    report and CI gate -- the runner and report layer only ever see the Metric interface,
    never a specific metric's internals.
    """
    name: str
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult: ...


def pair_by_id(cases: Sequence[Case], predictions: Sequence[Prediction]) -> list[tuple[Case, Prediction]]:
    """Match cases to predictions by id rather than assuming positional alignment --
    the runner does preserve order today, but a metric that silently depends on that is
    one refactor away from scoring the wrong case against the wrong prediction."""
    preds = {p.case_id: p for p in predictions}
    return [(c, preds[c.id]) for c in cases if c.id in preds]


def summarize(name: str, per_case: Sequence[float], *, higher_is_better: bool = True,
             regression_gated: bool = True, seed: int = 0,
             extra: dict[str, Any] | None = None) -> MetricResult:
    """Build a MetricResult with a 95% bootstrap CI.

    Shared by every concrete metric so the CI methodology is identical everywhere -- a
    metric that computes its own ad hoc interval is the exact inconsistency a client
    reviewer will catch.
    """
    values = list(per_case)
    ci_low, ci_high = bootstrap_ci(values, seed=seed)
    value = sum(values) / len(values) if values else 0.0
    return MetricResult(name=name, value=round(value, 4), per_case=tuple(values),
                        n=len(values), ci_low=round(ci_low, 4), ci_high=round(ci_high, 4),
                        higher_is_better=higher_is_better, regression_gated=regression_gated,
                        extra=dict(extra or {}))
