"""PgVectorStore against a live Postgres.

Skipped unless one is reachable, so the default `pytest` run stays offline:

    docker compose --profile pg up -d postgres
    export RAG_TEST_POSTGRES_DSN=postgresql://postgres:dev@localhost:5432/rag

This file exists because the store carried `# pragma: no cover - needs a live
Postgres` and had never actually been run. "Probably fine" is not a thing the
rest of this repo accepts about a metric, and it should not accept it about the
storage layer either.
"""
from __future__ import annotations

import os
import random

import pytest

from app.ingest.chunking import Chunk

DSN = os.environ.get("RAG_TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set RAG_TEST_POSTGRES_DSN to run")


@pytest.fixture
def store():
    from app.store.pgvector_store import PgVectorStore
    s = PgVectorStore(DSN, dim=4, table="test_chunks")
    s.conn.execute("TRUNCATE test_chunks")
    yield s
    s.conn.execute("DROP TABLE IF EXISTS test_chunks")
    s.conn.close()


def _chunk(cid, text, source, dept="finance"):
    return Chunk(cid, f"doc-{cid}", text, source, 0, "", {"dept": dept})


def test_upsert_search_and_delete_round_trip(store):
    store.upsert(
        [_chunk("c1", "Refunds take 30 days.", "refunds.md"),
         _chunk("c2", "Batteries last 12 months.", "warranty.md", dept="it")],
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0]],
    )
    assert store.count() == 2

    hits = store.search([1.0, 0, 0, 0], k=2)
    assert hits[0].chunk.chunk_id == "c1", "nearest vector should rank first"
    assert len(store.all_chunks()) == 2

    assert store.delete_document("doc-c1") == 1
    assert store.count() == 1


def test_keyword_leg_uses_postgres_full_text(store):
    store.upsert(
        [_chunk("c1", "Refunds take 30 days.", "refunds.md"),
         _chunk("c2", "Batteries last 12 months.", "warranty.md")],
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0]],
    )
    hits = store.keyword_search("refunds", k=5)
    assert [h.chunk.chunk_id for h in hits] == ["c1"]


def test_where_filter_scopes_both_legs(store):
    store.upsert(
        [_chunk("c1", "Quarterly invoice totals.", "invoices.md", dept="finance"),
         _chunk("c2", "Quarterly invoice totals.", "specs.md", dept="it")],
        [[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
    )
    assert [h.chunk.chunk_id for h in store.search([1.0, 0, 0, 0], k=5, where={"dept": "it"})] == ["c2"]
    assert [h.chunk.chunk_id for h in store.keyword_search("invoice", k=5, where={"dept": "it"})] == ["c2"]


def test_filtered_search_returns_k_rows_when_the_index_is_used(store):
    """Regression for silent row loss under a selective filter.

    HNSW filters *after* searching the index, so a selective WHERE discards
    most of what it found and the query comes back short with no error. It
    only happens once the planner picks the index, which needs more rows than
    a test would otherwise bother creating -- hence enable_seqscan=off, which
    is what a production-sized corpus does on its own.

    Measured before the fix: 1 row returned for k=10.
    """
    random.seed(0)
    n, depts = 4000, 40
    chunks, vectors = [], []
    for i in range(n):
        chunks.append(Chunk(f"c{i}", f"d{i}", f"body {i}", f"s{i}.md", 0, "",
                            {"dept": f"dept{i % depts}"}))
        vectors.append([random.gauss(0, 1) for _ in range(4)])
    for j in range(0, n, 500):
        store.upsert(chunks[j:j + 500], vectors[j:j + 500])
    store.conn.execute("ANALYZE test_chunks")
    store.conn.execute("SET enable_seqscan = off")

    hits = store.search([random.gauss(0, 1) for _ in range(4)], k=10, where={"dept": "dept7"})

    assert len(hits) == 10, f"asked for 10, got {len(hits)} -- hnsw.iterative_scan not in effect"
    assert all(h.chunk.metadata["dept"] == "dept7" for h in hits), "filter leaked other departments"


def test_filtered_search_refuses_rather_than_silently_truncating(store, monkeypatch):
    """On pgvector < 0.8 the parameter does not exist. Returning short results
    quietly is the one outcome that must not happen."""
    monkeypatch.setattr(store, "_iterative_scan", False)

    with pytest.raises(RuntimeError, match="pgvector >= 0.8"):
        store.search([1.0, 0, 0, 0], k=10, where={"dept": "finance"})

    store.search([1.0, 0, 0, 0], k=10)  # unfiltered is unaffected


# ---------------------------------------------------------------- access control

def _acl_chunk(cid, text, *, company, owner, dept=None, scope="private", status="active"):
    return Chunk(cid, f"doc-{cid}", text, f"{cid}.md", 0, "", {
        "company_id": company, "owner_id": owner, "department_id": dept,
        "scope": scope, "status": status,
    })


@pytest.fixture
def acl_store(store):
    """Pera in finance, Zika in IT, plus an unrelated company.

    Every document says the same thing, so nothing here can pass by ranking
    accident -- only the ACL predicate decides what comes back.
    """
    text = "Quarterly figures and coverage details."
    store.upsert([
        _acl_chunk("pera_private", text, company="acme", owner="pera"),
        _acl_chunk("finance_shared", text, company="acme", owner="mgr",
                   dept="finance", scope="department"),
        _acl_chunk("finance_pending", text, company="acme", owner="ana",
                   dept="finance", scope="department", status="pending_approval"),
        _acl_chunk("it_shared", text, company="acme", owner="zika",
                   dept="it", scope="department"),
        _acl_chunk("other_company", text, company="globex", owner="someone",
                   dept="finance", scope="department"),
    ], [[1.0, 0, 0, 0]] * 5)
    return store


def _seen(hits):
    return {h.chunk.chunk_id for h in hits}


def test_a_member_sees_own_private_and_approved_department_documents(acl_store):
    from app.store.access import AccessScope
    pera = AccessScope(company_id="acme", user_id="pera", department_ids=("finance",))

    assert _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=pera)) == {
        "pera_private", "finance_shared"}


def test_a_document_awaiting_approval_is_not_shared_yet(acl_store):
    """The queue exists so a manager decides; visible-while-pending would make
    the queue decorative."""
    from app.store.access import AccessScope
    colleague = AccessScope(company_id="acme", user_id="pera", department_ids=("finance",))

    assert "finance_pending" not in _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=colleague))


def test_the_submitter_still_sees_their_own_pending_document(acl_store):
    """Ownership outranks approval state: submitting a document for review must
    not make it vanish from the person who wrote it."""
    from app.store.access import AccessScope
    ana = AccessScope(company_id="acme", user_id="ana", department_ids=("finance",))

    assert "finance_pending" in _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=ana))


def test_departments_do_not_see_each_other(acl_store):
    from app.store.access import AccessScope
    zika = AccessScope(company_id="acme", user_id="zika", department_ids=("it",))

    seen = _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=zika))
    assert seen == {"it_shared"}
    assert "pera_private" not in seen and "finance_shared" not in seen


def test_the_company_boundary_holds(acl_store):
    from app.store.access import AccessScope
    outsider = AccessScope(company_id="globex", user_id="someone",
                           department_ids=("finance",))

    assert _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=outsider)) == {"other_company"}


def test_the_keyword_leg_is_scoped_too(acl_store):
    """Both retrieval legs run on every query; scoping one is scoping neither."""
    from app.store.access import AccessScope
    pera = AccessScope(company_id="acme", user_id="pera", department_ids=("finance",))

    assert _seen(acl_store.keyword_search("quarterly", k=10, access=pera)) == {
        "pera_private", "finance_shared"}


def test_count_and_listing_are_scoped(acl_store):
    """/documents and the index-size badge read these; unscoped, they leak the
    existence and filenames of documents the user cannot open."""
    from app.store.access import AccessScope
    pera = AccessScope(company_id="acme", user_id="pera", department_ids=("finance",))

    assert acl_store.count() == 5           # unscoped: the whole table
    assert acl_store.count(access=pera) == 2
    assert {c.chunk_id for c in acl_store.all_chunks(access=pera)} == {
        "pera_private", "finance_shared"}


def test_a_user_in_no_department_sees_only_their_own(acl_store):
    from app.store.access import AccessScope
    newcomer = AccessScope(company_id="acme", user_id="pera")

    assert _seen(acl_store.search([1.0, 0, 0, 0], k=10, access=newcomer)) == {"pera_private"}
