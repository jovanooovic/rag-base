from core.eval.baseline import compare, load_baseline, write_baseline
from core.eval.types import MetricResult, Scorecard


def _scorecard(value, name="recall@5", higher_is_better=True, regression_gated=True):
    m = MetricResult(name=name, value=value, per_case=(value,), n=1, ci_low=value, ci_high=value,
                     higher_is_better=higher_is_better, regression_gated=regression_gated)
    return Scorecard(label="test", metrics=(m,), n_cases=1)


def test_write_and_load_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(path, _scorecard(0.8))
    assert load_baseline(path) == {"recall@5": 0.8}


def test_load_baseline_returns_none_when_missing(tmp_path):
    assert load_baseline(tmp_path / "does-not-exist.json") is None


def test_compare_flags_a_regression_on_a_higher_is_better_metric():
    result = compare(_scorecard(0.5), {"recall@5": 0.8}, regression_threshold=0.02)
    assert "recall@5" in result["regressions"]
    assert result["deltas"]["recall@5"] == -0.3


def test_compare_does_not_flag_improvement():
    result = compare(_scorecard(0.9), {"recall@5": 0.8}, regression_threshold=0.02)
    assert result["regressions"] == []


def test_compare_flags_regression_on_lower_is_better_metric():
    result = compare(_scorecard(0.5, name="cost", higher_is_better=False),
                     {"cost": 0.2}, regression_threshold=0.02)
    assert "cost" in result["regressions"]


def test_compare_with_no_baseline_flags_nothing():
    result = compare(_scorecard(0.5), None)
    assert result == {"deltas": {}, "regressions": []}


def test_compare_reports_delta_but_does_not_gate_an_ungated_metric():
    # A latency-style metric: 20ms slower would trip a naive 2pp threshold, but on an
    # unbounded-scale metric that threshold is meaningless -- ordinary run-to-run
    # timing noise, not a real regression.
    result = compare(_scorecard(150.0, name="latency_total_ms", higher_is_better=False,
                                regression_gated=False),
                     {"latency_total_ms": 130.0}, regression_threshold=0.02)
    assert result["deltas"]["latency_total_ms"] == 20.0
    assert result["regressions"] == []
