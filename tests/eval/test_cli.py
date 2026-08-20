import json
import sys
import types

from core.eval import cli
from core.eval.metrics.base import Metric, summarize
from core.eval.types import Case, Prediction


class _AlwaysOneMetric(Metric):
    name = "always_one"
    higher_is_better = True

    def compute(self, cases, predictions):
        return summarize(self.name, [1.0 for _ in predictions])


def _fake_system(case: Case) -> Prediction:
    return Prediction(case_id=case.id, output={"answer_text": f"answer for {case.id}"})


def _install_fake_adapter(monkeypatch, name: str) -> None:
    module = types.ModuleType(name)

    def build_adapter():
        return (lambda: _fake_system), (lambda: [_AlwaysOneMetric()]), _load_cases

    def _load_cases(path):
        rows = [json.loads(line) for line in open(path).read().splitlines() if line.strip()]
        return [Case(id=r["id"], input={"question": r["question"]}) for r in rows]

    module.build_adapter = build_adapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


def _write_suite(tmp_path):
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"id": "1", "question": "q1"}\n{"id": "2", "question": "q2"}')
    return suite


def test_run_writes_scorecard_latest_md_and_cases_json(tmp_path, monkeypatch):
    _install_fake_adapter(monkeypatch, "fake_adapter_mod")
    suite = _write_suite(tmp_path)
    out = tmp_path / "out"

    argv = ["run", "--suite", str(suite), "--adapter", "fake_adapter_mod:build_adapter",
           "--out", str(out), "--label", "test"]
    code = cli.main(argv)

    assert code == 0
    assert (out / "latest.json").is_file()
    assert (out / "latest.md").is_file()
    cases_json = json.loads((out / "cases.json").read_text())
    assert len(cases_json) == 2
    assert cases_json[0]["prediction"]["output"]["answer_text"] == "answer for 1"


def test_accept_writes_a_baseline_file(tmp_path, monkeypatch):
    _install_fake_adapter(monkeypatch, "fake_adapter_mod2")
    suite = _write_suite(tmp_path)
    baseline = tmp_path / "baseline.json"

    argv = ["accept", "--suite", str(suite), "--adapter", "fake_adapter_mod2:build_adapter",
           "--baseline", str(baseline)]
    code = cli.main(argv)

    assert code == 0
    data = json.loads(baseline.read_text())
    assert any(m["name"] == "always_one" and m["value"] == 1.0 for m in data["metrics"])


def test_run_fails_on_regression_when_requested(tmp_path, monkeypatch):
    _install_fake_adapter(monkeypatch, "fake_adapter_mod3")
    suite = _write_suite(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": [{"name": "always_one", "value": 5.0}]}))
    out = tmp_path / "out"

    argv = ["run", "--suite", str(suite), "--adapter", "fake_adapter_mod3:build_adapter",
           "--out", str(out), "--baseline", str(baseline), "--fail-on-regression"]
    code = cli.main(argv)

    assert code == 1  # always_one=1.0 vs baseline 5.0 is a real regression
