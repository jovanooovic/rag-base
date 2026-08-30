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

    monkeypatch.setenv("APP_AUTH_SECRET", "t" * 48)

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

    # Real passwords now: identity comes from a signed session, not a header.
    from app.core import auth
    directory.set_password_hash(pera, auth.hash_password("pera-password-long"))
    directory.set_password_hash(zika, auth.hash_password("zika-password-long"))

    client = TestClient(api.app)
    yield {"client": client, "pera": pera, "zika": zika, "api": api,
           "passwords": {"pera": "pera-password-long", "zika": "zika-password-long"},
           "emails": {"pera": "pera@acme.rs", "zika": "zika@acme.rs"},
           "directory": directory}

    store.execute("TRUNCATE chunks")
    directory.close()


def _signed_in(tenant, user_key):
    """A client carrying that user's session cookie."""
    client = tenant["client"]
    resp = client.post("/auth/login", json={
        "email": tenant["emails"][user_key],
        "password": tenant["passwords"][user_key]})
    assert resp.status_code == 200, resp.text
    return client


def _ask(tenant, user_key, question="what is the quarterly figure"):
    return _signed_in(tenant, user_key).post("/ask", json={"question": question}).json()


def test_a_request_without_a_session_is_rejected(tenant):
    """In multi-user mode there is no anonymous read: without a user there is
    no scope, and no scope would mean the unfiltered corpus."""
    assert tenant["client"].post("/ask", json={"question": "anything"}).status_code == 401
    assert tenant["client"].get("/documents").status_code == 401


def test_a_forged_session_cookie_is_rejected(tenant):
    """The old X-Debug-User header let any caller name themselves. This is the
    test that it is really gone -- an unsigned value must buy nothing."""
    client = tenant["client"]
    client.cookies.set("rag_session", tenant["pera"])   # the bare user id

    assert client.post("/ask", json={"question": "anything"}).status_code == 401
    client.cookies.clear()


def test_the_debug_header_no_longer_grants_anything(tenant):
    resp = tenant["client"].post("/ask", json={"question": "anything"},
                                 headers={"X-Debug-User": tenant["pera"]})
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
    listing = _signed_in(tenant, "zika").get("/documents").json()

    assert [d["source"] for d in listing["by_source"]] == ["specs.md"]


def test_source_refuses_a_file_the_caller_cannot_see(tenant):
    """/documents used to hand out every filename and /source used to serve any
    of them to anyone who could name one."""
    resp = _signed_in(tenant, "zika").get("/source", params={"path": "invoices.md"})

    assert resp.status_code == 404


def test_health_counts_only_the_callers_own_chunks(tenant):
    body = _signed_in(tenant, "pera").get("/health").json()

    assert body["chunks_indexed"] == 1, "an unscoped count reveals how much exists"


# ---------------------------------------------------------------- auth flows

def test_registration_creates_a_company_whose_first_user_manages_it(tenant):
    client = tenant["client"]
    resp = client.post("/auth/register", json={
        "company_name": "Globex", "email": "boss@globex.rs",
        "password": "a-sufficiently-long-password"})
    assert resp.status_code == 200

    me = client.get("/auth/me").json()
    assert me["company_id"] == resp.json()["company_id"]
    assert me["manages"], "whoever signs a company up has nobody to approve them"


def test_a_second_company_cannot_reuse_an_email(tenant):
    client = tenant["client"]
    client.post("/auth/register", json={"company_name": "Globex", "email": "dup@x.rs",
                                        "password": "a-sufficiently-long-password"})
    again = client.post("/auth/register", json={"company_name": "Initech", "email": "dup@x.rs",
                                                "password": "a-sufficiently-long-password"})
    assert again.status_code == 409


def test_a_wrong_password_and_an_unknown_account_are_indistinguishable(tenant):
    """Different messages here would let anyone enumerate which addresses have
    accounts, one request at a time."""
    client = tenant["client"]
    wrong = client.post("/auth/login", json={"email": "pera@acme.rs", "password": "not-it-at-all"})
    missing = client.post("/auth/login", json={"email": "ghost@acme.rs", "password": "not-it-at-all"})

    assert wrong.status_code == missing.status_code == 401
    assert wrong.json() == missing.json()


def test_repeated_failures_are_rate_limited(tenant):
    client = tenant["client"]
    tenant["api"]._LOGIN_LIMITER.clear()

    codes = [client.post("/auth/login",
                         json={"email": "pera@acme.rs", "password": "wrong"}).status_code
             for _ in range(12)]

    assert 429 in codes, "an unlimited login endpoint verifies passwords as fast as they arrive"
    tenant["api"]._LOGIN_LIMITER.clear()


def test_a_successful_login_clears_the_failure_count(tenant):
    """One bad day at the keyboard must not keep counting against someone who
    then gets it right."""
    client = tenant["client"]
    tenant["api"]._LOGIN_LIMITER.clear()
    for _ in range(5):
        client.post("/auth/login", json={"email": "pera@acme.rs", "password": "wrong"})

    ok = client.post("/auth/login", json={"email": "pera@acme.rs",
                                          "password": tenant["passwords"]["pera"]})
    assert ok.status_code == 200
    assert tenant["api"]._LOGIN_LIMITER.check("email:pera@acme.rs")


def test_logout_stops_the_session_being_used(tenant):
    client = _signed_in(tenant, "pera")
    assert client.get("/auth/me").status_code == 200

    client.post("/auth/logout")

    assert client.get("/auth/me").status_code == 401


def test_the_session_cookie_is_httponly_and_samesite(tenant):
    """httponly keeps XSS from reading it; SameSite is the CSRF defence."""
    resp = tenant["client"].post("/auth/login", json={
        "email": "pera@acme.rs", "password": tenant["passwords"]["pera"]})

    cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie


def test_a_manager_can_add_someone_to_their_own_company_only(tenant):
    manager = _signed_in(tenant, "pera")

    created = manager.post("/users", json={"email": "novi@acme.rs",
                                           "password": "another-long-password"})
    assert created.status_code == 200

    from app.store.directory import Directory
    scope = tenant["directory"].scope_for(created.json()["user_id"])
    assert scope.company_id == tenant["directory"].scope_for(tenant["pera"]).company_id


def test_a_plain_member_cannot_add_users(tenant):
    member = _signed_in(tenant, "zika")

    resp = member.post("/users", json={"email": "sneaky@acme.rs",
                                       "password": "another-long-password"})

    assert resp.status_code == 403


def test_a_manager_cannot_add_users_to_a_department_they_do_not_manage(tenant):
    manager = _signed_in(tenant, "pera")
    zika_scope = tenant["directory"].scope_for(tenant["zika"])

    resp = manager.post("/users", json={"email": "other@acme.rs",
                                        "password": "another-long-password",
                                        "department_id": zika_scope.department_ids[0]})

    assert resp.status_code == 403
