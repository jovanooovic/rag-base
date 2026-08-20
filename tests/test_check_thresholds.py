from eval.check_thresholds import check


def test_check_passes_when_both_metrics_clear_the_floor():
    scorecard = {"metrics": [{"name": "faithfulness", "value": 0.9},
                            {"name": "refusal_accuracy", "value": 0.95}]}
    assert check(scorecard) == []


def test_check_flags_faithfulness_below_floor():
    scorecard = {"metrics": [{"name": "faithfulness", "value": 0.5},
                            {"name": "refusal_accuracy", "value": 0.95}]}
    failures = check(scorecard)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_check_ignores_metrics_that_were_not_measured():
    scorecard = {"metrics": [{"name": "recall@5", "value": 0.1}]}
    assert check(scorecard) == []
