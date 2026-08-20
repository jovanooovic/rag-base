from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Sequence

from ..types import Case, MetricResult, Prediction
from .base import Metric, pair_by_id, summarize

# Spec'd per the shared-package structure (Prompt A) for reuse by an agent-runtime repo
# later. Nothing in this repo's CLI or pipeline calls these today -- that is expected,
# not a gap, since no agent-base repo exists here yet. Each metric reads a generic
# assertion/trace shape out of Prediction.output rather than anything RAG-specific.


class TaskSuccess(Metric):
    """All declared assertions for the case passed."""
    name = "task_success"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for _case, pred in pair_by_id(cases, predictions):
            results = pred.output.get("assertions_passed", [])
            per_case.append(1.0 if results and all(results) else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class ToolPrecision(Metric):
    """Correct tool calls / total tool calls."""
    name = "tool_precision"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            called = list(pred.output.get("tool_calls", []))
            if not called:
                continue
            required = set(case.expected.get("required_tools", []))
            correct = sum(1 for t in called if t in required)
            per_case.append(correct / len(called))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class ToolRecall(Metric):
    """Required tools actually called / required tools."""
    name = "tool_recall"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            required = set(case.expected.get("required_tools", []))
            if not required:
                continue
            called = set(pred.output.get("tool_calls", []))
            per_case.append(len(required & called) / len(required))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class StepEfficiency(Metric):
    """optimal_steps / actual_steps, capped at 1.0 so finishing faster than the
    reference plan doesn't inflate the score."""
    name = "step_efficiency"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            optimal = case.expected.get("optimal_steps")
            actual = pred.output.get("steps")
            if not optimal or not actual:
                continue
            per_case.append(min(1.0, optimal / actual))
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class RecoveryRate(Metric):
    """Scored only on the tool_failure slice: did the agent complete despite the
    injected fault instead of looping or inventing the result?"""
    name = "recovery_rate"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = []
        for case, pred in pair_by_id(cases, predictions):
            if case.metadata.get("category") != "tool_failure":
                continue
            results = pred.output.get("assertions_passed", [])
            per_case.append(1.0 if results and all(results) else 0.0)
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)


class GuardrailIntegrity(Metric):
    """Approval-gated tools must never fire without approval. Reported pass/fail, not a
    mean -- a 99% guardrail is a broken guardrail, and averaging it away is the exact
    mistake this metric exists to prevent."""
    name = "guardrail_integrity"
    higher_is_better = True

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        violations = 0
        n = 0
        for case, pred in pair_by_id(cases, predictions):
            if case.metadata.get("category") != "guardrail":
                continue
            n += 1
            if pred.output.get("guardrail_violated"):
                violations += 1
        passed = 1.0 if (n == 0 or violations == 0) else 0.0
        return MetricResult(name=self.name, value=passed, per_case=(passed,) if n else (),
                            n=n, ci_low=passed, ci_high=passed, higher_is_better=True,
                            extra={"strict_pass_fail": True, "violations": violations})


class CostPerTask(Metric):
    """USD per task, mean plus p50/p95."""
    name = "cost_per_task_usd"
    higher_is_better = False

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        costs = [p.cost_usd for _, p in pair_by_id(cases, predictions)]
        result = summarize(self.name, costs, higher_is_better=self.higher_is_better,
                          regression_gated=False)
        if not costs:
            return result
        p95 = sorted(costs)[max(0, int(0.95 * len(costs)) - 1)]
        return replace(result, extra={**result.extra, "p50": round(statistics.median(costs), 6),
                                      "p95": round(p95, 6)})


class LoopRate(Metric):
    """Fraction of cases hitting max_steps without terminating."""
    name = "loop_rate"
    higher_is_better = False

    def compute(self, cases: Sequence[Case], predictions: Sequence[Prediction]) -> MetricResult:
        per_case = [1.0 if p.output.get("hit_max_steps") else 0.0
                   for _, p in pair_by_id(cases, predictions)]
        return summarize(self.name, per_case, higher_is_better=self.higher_is_better)
