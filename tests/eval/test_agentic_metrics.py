from core.eval.metrics.agentic import (CostPerTask, GuardrailIntegrity, LoopRate, RecoveryRate,
                                       StepEfficiency, TaskSuccess, ToolPrecision, ToolRecall)
from core.eval.types import Case, Prediction


def test_task_success_requires_all_assertions():
    cases = [Case(id="1", input={})]
    preds = [Prediction(case_id="1", output={"assertions_passed": [True, False]})]
    assert TaskSuccess().compute(cases, preds).value == 0.0


def test_tool_precision_and_recall_hand_computed():
    case = Case(id="1", input={}, expected={"required_tools": ["a", "b"]})
    pred = Prediction(case_id="1", output={"tool_calls": ["a", "c"]})
    assert ToolPrecision().compute([case], [pred]).value == 0.5
    assert ToolRecall().compute([case], [pred]).value == 0.5


def test_step_efficiency_caps_at_one():
    case = Case(id="1", input={}, expected={"optimal_steps": 4})
    pred = Prediction(case_id="1", output={"steps": 2})  # better than optimal
    assert StepEfficiency().compute([case], [pred]).value == 1.0


def test_recovery_rate_only_scores_tool_failure_category():
    cases = [Case(id="1", input={}, metadata={"category": "tool_failure"})]
    preds = [Prediction(case_id="1", output={"assertions_passed": [True]})]
    result = RecoveryRate().compute(cases, preds)
    assert result.n == 1
    assert result.value == 1.0


def test_guardrail_integrity_is_pass_fail_not_a_mean():
    cases = [Case(id=str(i), input={}, metadata={"category": "guardrail"}) for i in range(10)]
    preds = [Prediction(case_id=str(i), output={"guardrail_violated": i == 0}) for i in range(10)]
    result = GuardrailIntegrity().compute(cases, preds)
    assert result.value == 0.0  # a single violation fails the whole metric
    assert result.extra["violations"] == 1


def test_cost_per_task_reports_p50_and_p95():
    cases = [Case(id=str(i), input={}) for i in range(4)]
    preds = [Prediction(case_id=str(i), output={}, cost_usd=c)
             for i, c in enumerate([0.1, 0.2, 0.3, 0.4])]
    result = CostPerTask().compute(cases, preds)
    assert "p50" in result.extra and "p95" in result.extra


def test_loop_rate_hand_computed():
    cases = [Case(id="1", input={}), Case(id="2", input={})]
    preds = [Prediction(case_id="1", output={"hit_max_steps": True}),
             Prediction(case_id="2", output={"hit_max_steps": False})]
    assert LoopRate().compute(cases, preds).value == 0.5
