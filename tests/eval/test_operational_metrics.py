from core.eval.metrics.operational import LatencyPercentile
from core.eval.types import Case, Prediction


def test_latency_percentile_reads_top_level_latency_by_default():
    cases = [Case(id=str(i), input={}) for i in range(4)]
    preds = [Prediction(case_id=str(i), output={}, latency_ms=v)
             for i, v in enumerate([100.0, 200.0, 300.0, 400.0])]
    result = LatencyPercentile().compute(cases, preds)
    assert result.value == 250.0
    assert result.extra["p50"] == 200.0


def test_latency_percentile_reads_a_named_trace_field():
    cases = [Case(id="1", input={})]
    preds = [Prediction(case_id="1", output={}, trace={"retrieve_ms": 42.0})]
    result = LatencyPercentile(trace_field="retrieve_ms").compute(cases, preds)
    assert result.name == "latency_retrieve_ms"
    assert result.value == 42.0


def test_latency_percentile_is_lower_is_better():
    assert LatencyPercentile().higher_is_better is False


def test_latency_percentile_skips_missing_values():
    cases = [Case(id="1", input={}), Case(id="2", input={})]
    preds = [Prediction(case_id="1", output={}, trace={"retrieve_ms": 10.0}),
             Prediction(case_id="2", output={}, trace={})]  # no retrieve_ms recorded
    result = LatencyPercentile(trace_field="retrieve_ms").compute(cases, preds)
    assert result.n == 1
