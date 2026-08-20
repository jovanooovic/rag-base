from eval.scorecard_png import render


def _scorecard(n_metrics=3, kappa=None):
    metrics = [{"name": f"metric_{i}", "value": 0.9, "ci": [0.8, 1.0], "n": 10}
              for i in range(n_metrics)]
    return {"label": "test-scorecard", "n_cases": 135, "meta": {"kappa": kappa},
           "metrics": metrics}


def test_render_produces_an_image_sized_to_the_metric_count():
    small = render(_scorecard(n_metrics=3))
    large = render(_scorecard(n_metrics=20))
    assert small.width == 1200
    assert large.height > small.height


def test_render_handles_zero_metrics_without_crashing():
    img = render(_scorecard(n_metrics=0))
    assert img.width == 1200
    assert img.height > 0


def test_render_accepts_a_custom_title():
    img = render(_scorecard(), title="custom title")
    assert img.mode == "RGB"
