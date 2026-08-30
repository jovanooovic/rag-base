"""The API in multi-user mode, through the HTTP layer.

The store and the directory are proven separately. This asks the question a
client actually asks: if Pera and Zika both hit /ask with the same question,
does either one see the other's documents -- and do /documents, /source and
the answer cache hold the same line as retrieval does.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from tests.conftest import POSTGRES_ADMIN_DSN as ADMIN_DSN
from tests.conftest import postgres_app_dsn as _app_dsn
from tests.conftest import requires_postgres

pytestmark = requires_postgres



@pytest.fixture
def tenant(tmp_path, monkeypatch):
    from app.store.directory import ROLE_MANAGER, Directory

    app_dsn = _app_dsn()
    cfg = {
        "project_name": "mt-test",
        "llm_provider": "mock", "embedding_provider": "mock", "embedding_dim": 256,
        "data_dir": str(tmp_path), "trace_enabled": False, "use_reranker": False,
        "store_backend": "pgvector", "postgres_dsn": app_dsn, "multi_tenant": True,
    }
    (tmp_path / "project.config.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)

    directory = Directory(app_dsn)
    with directory._conn() as conn:
        for table in ("memberships", "documents", "departments", "users", "companies"):
            conn.execute(f"TRUNCATE {table} CASCADE")

    acme = directory.create_company("Acme")
    finance = directory.create_department(acme, "Finance")
    it = directory.create_department(acme, "IT")
    pera = directory.create_user(acme, "pera@acme.rs")
    zika = directory.create_user(acme, "zika@acme.rs")
    directory.add_membership(pera, finance, ROLE_MANAGER)
    directory.add_membership(zika, it)

    from app import api
    api.get_settings.cache_clear()
    api.get_store.cache_clear()
    api.get_directory.cache_clear()
    api._ANSWER_CACHE.clear()

    store = api.get_store()
    store.execute("TRUNCATE chunks")

    # Two documents that answer the same question differently, each private to
    # its owner. Identical wording, so ranking cannot be what separates them.
    from app.ingest.chunking import Chunk
    body = "The quarterly figure is {}."
    store.upsert([
        Chunk("f1", "d-finance", body.format("EUR 4.2M in invoices"), "invoices.md", 0, "",
              {"company_id": acme, "owner_id": pera, "department_id": finance,
               "scope": "department", "status": "active"}),
        Chunk("i1", "d-it", body.format("48 open specification tickets"), "specs.md", 0, "",
              {"company_id": acme, "owner_id": zika, "department_id": it,
               "scope": "department", "status": "active"}),
    ], api.get_pipeline().embeddings.embed([body.format("EUR 4.2M in invoices"),
                                            body.format("48 open specification tickets")]))

    client = TestClient(api.app)
    yield {"client": client, "pera": pera, "zika": zika, "api": api}

    store.execute("TRUNCATE chunks")
    directory.close()


def _ask(tenant, user_key, question="what is the quarterly figure"):
    return tenant["client"].post("/ask", json={"question": question},
                                 headers={"X-Debug-User": tenant[user_key]}).json()


def test_a_request_without_identity_is_rejected(tenant):
    """In multi-user mode there is no anonymous read: without a user there is
    no scope, and no scope would mean the unfiltered corpus."""
    assert tenant["client"].post("/ask", json={"question": "anything"}).status_code == 401
    assert tenant["client"].get("/documents").status_code == 401


def test_an_unknown_user_is_rejected(tenant):
    resp = tenant["client"].post("/ask", json={"question": "anything"},
                                 headers={"X-Debug-User": "not-a-real-id"})
    assert resp.status_code == 401


def test_two_users_asking_the_same_question_get_their_own_documents(tenant):
    pera, zika = _ask(tenant, "pera"), _ask(tenant, "zika")

    assert [c["source"] for c in pera["citations"]] == ["invoices.md"]
    assert [c["source"] for c in zika["citations"]] == ["specs.md"]
    assert "specification" not in pera["answer"]
    assert "invoices" not in zika["answer"]


def test_the_answer_cache_does_not_serve_one_users_answer_to_another(tenant):
    """The cache key carries identity. Without that, the second user's question
    is a cache hit on the first user's answer -- retrieval never runs, and the
    isolation below it never gets a chance to matter."""
    first = _ask(tenant, "pera")
    second = _ask(tenant, "zika")

    assert first["cached"] is False
    assert second["cached"] is False, "cache hit across users"
    assert first["answer"] != second["answer"]

    assert _ask(tenant, "pera")["cached"] is True, "same user should still hit the cache"


def test_documents_lists_only_what_the_caller_can_open(tenant):
    client = tenant["client"]
    listing = client.get("/documents", headers={"X-Debug-User": tenant["zika"]}).json()

    assert [d["source"] for d in listing["by_source"]] == ["specs.md"]


def test_source_refuses_a_file_the_caller_cannot_see(tenant):
    """/documents used to hand out every filename and /source used to serve any
    of them to anyone who could name one."""
    client = tenant["client"]
    resp = client.get("/source", params={"path": "invoices.md"},
                      headers={"X-Debug-User": tenant["zika"]})

    assert resp.status_code == 404


def test_health_counts_only_the_callers_own_chunks(tenant):
    client = tenant["client"]
    body = client.get("/health", headers={"X-Debug-User": tenant["pera"]}).json()

    assert body["chunks_indexed"] == 1, "an unscoped count reveals how much exists"
