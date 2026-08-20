import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = {
        "project_name": "api-test",
        "llm_provider": "mock",
        "embedding_provider": "mock",
        "data_dir": str(tmp_path),
        "trace_enabled": False,
        "use_reranker": False,
    }
    (tmp_path / "project.config.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)

    from app import api
    api.get_settings.cache_clear()
    api.get_store.cache_clear()
    return TestClient(api.app), Path(__file__).resolve().parents[1]


def test_health_reports_an_empty_index_before_ingestion(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 0


def test_ask_before_ingestion_returns_409(client):
    c, _ = client
    assert c.post("/ask", json={"question": "anything?"}).status_code == 409


def test_ingest_then_ask_returns_citations(client):
    c, repo = client
    report = c.post("/ingest", json={"path": str(repo / "data" / "sample")}).json()
    assert report["chunks_embedded"] > 0

    body = c.post("/ask", json={"question": "how long is the return window"}).json()
    assert body["retrieved"], "expected retrieved chunks in the response"
    assert "answer" in body and "trace" in body


def test_ask_rejects_an_empty_question(client):
    c, _ = client
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_ingest_missing_path_is_a_clean_error(client):
    c, _ = client
    assert c.post("/ingest", json={"path": "/nope/does/not/exist"}).status_code in (404, 500)
