from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    """One evaluation input.

    `input` and `expected` are deliberately untyped dicts: core.eval is shared across a
    RAG pipeline and an agent runtime, and forcing both onto one strict schema would mean
    either field bloats to fit the union or one side hacks around the other's shape. The
    adapter for each system under test owns the schema; core.eval only ever sees dicts.
    """
    id: str
    input: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    output: dict[str, Any]
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    per_case: tuple[float, ...]
    n: int
    ci_low: float
    ci_high: float
    higher_is_better: bool = True
    # A fixed "N percentage points" regression threshold only makes sense on a metric
    # whose value lives on a bounded [0, 1] scale (recall, faithfulness, ...). A metric
    # on an unbounded scale -- latency in ms, cost in USD -- would trip that same
    # threshold on ordinary run-to-run noise, not a real regression. Baseline
    # comparison skips metrics with this set to False; they're still reported, just
    # not gated.
    regression_gated: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scorecard:
    label: str
    metrics: tuple[MetricResult, ...]
    n_cases: int
    meta: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, float] | None = None

    def get(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)
