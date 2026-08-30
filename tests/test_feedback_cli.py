import json
from pathlib import Path

from app.cli import main


def _cfg(tmp_path):
    cfg = {"project_name": "t", "llm_provider": "mock", "embedding_provider": "mock",
           "data_dir": str(tmp_path), "trace_enabled": False}
    p = tmp_path / "project.config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_feedback_export_writes_draft_cases(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    (tmp_path / "feedback.jsonl").write_text(
        json.dumps({"ts": "t", "run_id": "a", "verdict": "down",
                    "question": "how long is the warranty", "note": "it is 24 months"}) + "\n"
        + json.dumps({"ts": "t", "run_id": "b", "verdict": "up", "question": "fine one"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "drafts.jsonl"

    code = main(["--config", str(cfg), "feedback-export", "--out", str(out)])

    assert code == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1, "only the thumbs-down should become a draft"
    assert rows[0]["question"] == "how long is the warranty"
    assert rows[0]["gold_answer"] == "", "a draft must not invent a gold answer"
    assert "needs review" in rows[0]["notes"]
    assert "it is 24 months" in rows[0]["notes"]


def test_feedback_export_deduplicates_by_question(tmp_path):
    cfg = _cfg(tmp_path)
    rows = [json.dumps({"ts": "t", "run_id": str(i), "verdict": "down",
                        "question": "same broken question"}) for i in range(5)]
    (tmp_path / "feedback.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "drafts.jsonl"

    main(["--config", str(cfg), "feedback-export", "--out", str(out)])

    written = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(written) == 1, "one broken answer found by five people is one case"


def test_feedback_export_needs_no_api_key(tmp_path, monkeypatch):
    """It reads a local JSONL file; building a pipeline would demand a provider
    key and make triaging feedback impossible offline."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg_path = tmp_path / "project.config.json"
    cfg_path.write_text(json.dumps({
        "project_name": "t", "llm_provider": "openrouter", "embedding_provider": "openrouter",
        "data_dir": str(tmp_path), "trace_enabled": False}))
    (tmp_path / "feedback.jsonl").write_text(
        json.dumps({"ts": "t", "run_id": "a", "verdict": "down", "question": "q"}) + "\n",
        encoding="utf-8")

    code = main(["--config", str(cfg_path), "feedback-export", "--out", str(tmp_path / "o.jsonl")])

    assert code == 0
