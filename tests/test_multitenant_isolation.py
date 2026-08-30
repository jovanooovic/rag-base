"""Directory and PgVectorStore together, with real ids.

The pieces are tested apart elsewhere. This is the join: a user id goes into
`scope_for`, the resulting AccessScope goes into the store, and the question is
whether Pera from accounting can reach Zika's specs. Everything here uses
generated uuids rather than readable strings like "acme"/"pera", because a
predicate that happens to work on hand-written ids is not evidence.
"""
from __future__ import annotations

import pytest

from app.ingest.chunking import Chunk

from tests.conftest import POSTGRES_ADMIN_DSN as ADMIN_DSN
from tests.conftest import postgres_app_dsn as _app_dsn
from tests.conftest import requires_postgres

pytestmark = requires_postgres



@pytest.fixture
def org():
    """Two companies. Inside the first: Pera manages Finance, Zika is in IT.

    Every chunk carries identical text, so retrieval cannot separate them --
    only the access predicate can.
    """
    from app.store.directory import ROLE_MANAGER, Directory
    from app.store.pgvector_store import PgVectorStore

    directory = Directory(_app_dsn())
    with directory._conn() as conn:
        for table in ("memberships", "documents", "departments", "users", "companies"):
            conn.execute(f"TRUNCATE {table} CASCADE")

    store = PgVectorStore(_app_dsn(), dim=4, table="test_isolation")
    store.execute("TRUNCATE test_isolation")

    acme = directory.create_company("Acme")
    globex = directory.create_company("Globex")
    finance = directory.create_department(acme, "Finance")
    it = directory.create_department(acme, "IT")
    pera = directory.create_user(acme, "pera@acme.rs")
    zika = directory.create_user(acme, "zika@acme.rs")
    # A second IT person owns the shared spec, so Zika reaches it through
    # membership alone. Owning it would make every membership assertion below
    # pass for the wrong reason.
    marko = directory.create_user(acme, "marko@acme.rs")
    rival = directory.create_user(globex, "rival@globex.rs")
    directory.add_membership(pera, finance, ROLE_MANAGER)
    directory.add_membership(zika, it)
    directory.add_membership(marko, it)

    text = "Quarterly figures, coverage details and internal notes."

    def chunk(cid, owner, company, dept=None, scope="private", status="active"):
        return Chunk(cid, f"doc-{cid}", text, f"{cid}.md", 0, "", {
            "company_id": company, "owner_id": owner, "department_id": dept,
            "scope": scope, "status": status})

    store.upsert([
        chunk("invoices", pera, acme, finance, "department"),
        chunk("pera_notes", pera, acme),
        chunk("specs", marko, acme, it, "department"),
        chunk("draft_spec", zika, acme, it, "department", "pending_approval"),
        chunk("rival_doc", rival, globex),
    ], [[1.0, 0, 0, 0]] * 5)

    yield {"directory": directory, "store": store, "pera": pera, "zika": zika,
           "marko": marko, "rival": rival, "finance": finance, "it": it}

    store.execute("DROP TABLE IF EXISTS test_isolation")
    store.close()
    directory.close()


def _visible(org, user_key):
    scope = org["directory"].scope_for(org[user_key])
    return {h.chunk.chunk_id for h in org["store"].search([1.0, 0, 0, 0], k=10, access=scope)}


def test_accounting_cannot_read_the_it_specification(org):
    """The sentence this whole tier exists to make true."""
    assert _visible(org, "pera") == {"invoices", "pera_notes"}
    assert "specs" not in _visible(org, "pera")


def test_it_cannot_read_the_invoices(org):
    assert _visible(org, "zika") == {"specs", "draft_spec"}
    assert "invoices" not in _visible(org, "zika")


def test_a_draft_awaiting_approval_reaches_nobody_but_its_author(org):
    assert "draft_spec" in _visible(org, "zika"), "the author must still see their own draft"
    assert "draft_spec" not in _visible(org, "pera")


def test_another_company_shares_nothing(org):
    assert _visible(org, "rival") == {"rival_doc"}


def test_the_same_question_returns_different_sources_per_user(org):
    """Identical query, identical documents, different answers -- because the
    boundary is applied at retrieval rather than filtered out afterwards."""
    assert _visible(org, "pera") != _visible(org, "zika")
    assert _visible(org, "pera") & _visible(org, "zika") == set()


def test_a_scope_cannot_be_widened_by_the_callers_own_filter(org):
    """`where` is the caller's; `access` is the boundary. A caller passing a
    filter must narrow what they see, never reach past it."""
    scope = org["directory"].scope_for(org["pera"])
    hits = org["store"].search([1.0, 0, 0, 0], k=10,
                               where={"department_id": org["it"]}, access=scope)

    assert {h.chunk.chunk_id for h in hits} == set()


def test_losing_a_membership_removes_access_immediately(org):
    """Access is derived at query time, not cached into a token -- so removing
    someone from a department takes effect on their next question."""
    directory = org["directory"]
    assert "specs" in _visible(org, "zika")

    with directory._conn() as conn:
        conn.execute("DELETE FROM memberships WHERE user_id = %s AND department_id = %s",
                     (org["zika"], org["it"]))

    assert "specs" not in _visible(org, "zika")
    assert "draft_spec" in _visible(org, "zika"), "still theirs by ownership"
