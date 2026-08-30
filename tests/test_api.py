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
    api._ANSWER_CACHE.clear()  # module-level, so tests don't leak hits into each other
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


def test_ask_caches_a_repeated_question(client):
    c, repo = client
    c.post("/ingest", json={"path": str(repo / "data" / "sample")})
    req = {"question": "how long is the return window"}

    first = c.post("/ask", json=req).json()
    second = c.post("/ask", json=req).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["citations"] == first["citations"]


def test_ask_cache_is_keyed_on_history_not_just_the_question(client):
    c, repo = client
    c.post("/ingest", json={"path": str(repo / "data" / "sample")})
    question = "how long is the return window"

    no_history = c.post("/ask", json={"question": question}).json()
    with_history = c.post("/ask", json={
        "question": question,
        "history": [{"role": "user", "content": "I bought a jacket last week"}],
    }).json()

    assert no_history["cached"] is False
    assert with_history["cached"] is False, "different history must not hit the other request's cache entry"


def test_ingest_clears_the_answer_cache(client):
    c, repo = client
    sample = str(repo / "data" / "sample")
    c.post("/ingest", json={"path": sample})
    req = {"question": "how long is the return window"}

    c.post("/ask", json=req)  # populates the cache
    assert c.post("/ask", json=req).json()["cached"] is True

    c.post("/ingest", json={"path": sample})  # re-ingest, even of the same corpus

    assert c.post("/ask", json=req).json()["cached"] is False, \
        "a fresh ingest must invalidate previously cached answers"


def test_ask_rejects_an_empty_question(client):
    c, _ = client
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_ingest_missing_path_is_a_clean_error(client):
    c, _ = client
    assert c.post("/ingest", json={"path": "/nope/does/not/exist"}).status_code in (404, 500)


def test_upload_ingests_a_supported_file(client):
    c, _ = client
    files = [("files", ("policy.md", b"# Policy\n\nReturns are accepted within 30 days.",
                        "text/markdown"))]
    resp = c.post("/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_embedded"] > 0
    assert body["saved_files"] == [f for f in body["saved_files"] if f.endswith("policy.md")]


def test_upload_rejects_an_unsupported_file_type(client):
    c, _ = client
    files = [("files", ("virus.exe", b"not really a document", "application/octet-stream"))]
    assert c.post("/upload", files=files).status_code == 415


def test_upload_rejects_more_than_the_file_limit(client):
    c, _ = client
    files = [("files", (f"doc{i}.md", b"# x\n\nbody text here.", "text/markdown"))
             for i in range(11)]
    assert c.post("/upload", files=files).status_code == 400


def test_upload_sanitises_path_traversal_in_filenames(client):
    c, _ = client
    files = [("files", ("../../evil.md", b"# x\n\nbody text here.", "text/markdown"))]
    resp = c.post("/upload", files=files)
    assert resp.status_code == 200
    for name in resp.json()["saved_files"]:
        assert ".." not in name and "/" not in name and "\\" not in name


def test_write_endpoints_are_open_when_no_admin_token_is_configured(client, monkeypatch):
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    c, repo = client
    resp = c.post("/ingest", json={"path": str(repo / "data" / "sample")})
    assert resp.status_code == 200


def test_write_endpoints_reject_a_missing_admin_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "secret123")
    c, repo = client
    resp = c.post("/ingest", json={"path": str(repo / "data" / "sample")})
    assert resp.status_code == 401


def test_write_endpoints_reject_a_wrong_admin_token(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "secret123")
    c, repo = client
    resp = c.post("/ingest", json={"path": str(repo / "data" / "sample")},
                  headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 401


def test_write_endpoints_accept_the_correct_admin_token(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "secret123")
    c, repo = client
    resp = c.post("/ingest", json={"path": str(repo / "data" / "sample")},
                  headers={"X-Admin-Token": "secret123"})
    assert resp.status_code == 200


def test_read_endpoints_are_unaffected_by_the_admin_token(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "secret123")
    c, _ = client
    assert c.get("/health").status_code == 200


def test_source_serves_the_original_bytes_of_an_indexed_document(client):
    c, repo = client
    c.post("/ingest", json={"path": str(repo / "data" / "sample")})
    doc_path = c.get("/documents").json()["by_source"][0]["source"]

    resp = c.get("/source", params={"path": doc_path})

    assert resp.status_code == 200
    assert resp.content == Path(doc_path).read_bytes()


def test_source_rejects_a_path_that_was_never_indexed(client):
    c, repo = client
    c.post("/ingest", json={"path": str(repo / "data" / "sample")})

    # a real file on disk, just never ingested -- not merely "doesn't exist"
    assert c.get("/source", params={"path": str(repo / "README.md")}).status_code == 404


def test_source_rejects_traversal_attempts(client):
    c, repo = client
    c.post("/ingest", json={"path": str(repo / "data" / "sample")})

    resp = c.get("/source", params={"path": "../../../../etc/passwd"})
    assert resp.status_code == 404


def test_feedback_records_a_verdict(client):
    c, _ = client
    resp = c.post("/feedback", json={"run_id": "abc123", "question": "how long is the warranty",
                                     "verdict": "down", "note": "wrong, it is 24 months"})
    assert resp.status_code == 200

    from app import api
    rows = api.read_feedback(api.get_settings())
    assert len(rows) == 1
    assert rows[0]["verdict"] == "down"
    assert rows[0]["note"] == "wrong, it is 24 months"
    assert rows[0]["ts"]


def test_feedback_rejects_a_bogus_verdict(client):
    c, _ = client
    assert c.post("/feedback", json={"run_id": "a", "question": "q",
                                     "verdict": "maybe"}).status_code == 422


def test_feedback_rejects_an_oversized_note(client):
    c, _ = client
    resp = c.post("/feedback", json={"run_id": "a", "question": "q", "verdict": "down",
                                     "note": "x" * 5000})
    assert resp.status_code == 422


def test_feedback_refuses_once_the_log_is_full(client, monkeypatch):
    """POST /feedback is the one write endpoint with no token, so an unbounded
    append-only file is a disk-fill DoS on anything publicly reachable."""
    from app import api
    monkeypatch.setattr(api, "FEEDBACK_MAX_BYTES", 50)
    c, _ = client

    for _ in range(5):
        last = c.post("/feedback", json={"run_id": "a", "question": "a longer question here",
                                         "verdict": "down"})
    assert last.status_code == 507


def test_feedback_redacts_pii_in_the_note_when_configured(client, monkeypatch):
    """The note is free text a human typed about a real answer -- which is
    exactly where a name, an email or an order number turns up."""
    from app import api
    api.get_settings().extra["redact_pii"] = True
    try:
        c, _ = client
        c.post("/feedback", json={"run_id": "a", "question": "who do I email",
                                  "verdict": "down", "note": "should say ana@acme-example.com"})
        rows = api.read_feedback(api.get_settings())
        assert "ana@acme-example.com" not in rows[-1]["note"]
        assert "<EMAIL>" in rows[-1]["note"]
    finally:
        api.get_settings().extra.pop("redact_pii", None)


def test_feedback_summary_is_admin_gated(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "secret123")
    c, _ = client
    assert c.get("/feedback").status_code == 401
    assert c.get("/feedback", headers={"X-Admin-Token": "secret123"}).status_code == 200


def test_feedback_summary_counts_and_returns_recent_negatives(client):
    c, _ = client
    c.post("/feedback", json={"run_id": "a", "question": "q1", "verdict": "up"})
    c.post("/feedback", json={"run_id": "b", "question": "q2", "verdict": "down"})

    body = c.get("/feedback").json()

    assert body["total"] == 2 and body["up"] == 1 and body["down"] == 1
    assert body["recent_negative"][0]["question"] == "q2"


def test_read_feedback_skips_a_corrupt_line(client):
    """A half-written row from a killed process must not take down the admin
    panel whose whole job is reading this file."""
    from app import api
    c, _ = client
    c.post("/feedback", json={"run_id": "a", "question": "q1", "verdict": "down"})
    path = api._feedback_path(api.get_settings())
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"truncated": \n')

    assert len(api.read_feedback(api.get_settings())) == 1
