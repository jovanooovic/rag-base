from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..types import Case, MetricResult, Prediction
from .base import Metric, summarize


class LatencyPercentile(Metric):
    """p50/p95/p99 latency in milliseconds.

    Reads Prediction.latency_ms by default (end-to-end), or a named
    Prediction.trace field when `trace_field` is set, so the same metric class covers
    both the overall number and a per-stage split (e.g. retrieve vs. generate) without
    duplicating the percentile math.
    """
    higher_is_better = False

    def __init__(self, *, trace_field: str | None = None, name: str | None = None):
        self.trace_field = trace_field
        self.name = name or (f"latency_{trace_field}" if trace_field else "latency_total_ms")

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        values: list[float] = []
        for p in predictions:
            v = p.trace.get(self.trace_field) if self.trace_field else p.latency_ms
            if v is not None:
                values.append(float(v))
        result = summarize(self.name, values, higher_is_better=self.higher_is_better,
                          regression_gated=False)
        if not values:
            return result
        s = sorted(values)
        p50 = s[max(0, int(0.50 * len(s)) - 1)]
        p95 = s[max(0, int(0.95 * len(s)) - 1)]
        p99 = s[max(0, int(0.99 * len(s)) - 1)]
        return replace(result, extra={**result.extra, "p50": round(p50, 2), "p95": round(p95, 2),
                                      "p99": round(p99, 2)})
