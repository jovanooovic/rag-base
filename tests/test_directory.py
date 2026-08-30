"""Directory (companies, departments, users, memberships) against live Postgres.

Same skip rule as test_pgvector_store.py -- see its docstring.
"""
from __future__ import annotations

import pytest

from tests.conftest import postgres_app_dsn, requires_postgres

pytestmark = requires_postgres


@pytest.fixture
def directory():
    from app.store.directory import Directory
    d = Directory(postgres_app_dsn())
    with d._conn() as conn:
        # Children first: every one of these is a FK into the next.
        for table in ("memberships", "documents", "departments", "users", "companies"):
            conn.execute(f"TRUNCATE {table} CASCADE")
    yield d
    d.close()


@pytest.fixture
def acme(directory):
    """Pera manages finance, Zika is a member of IT."""
    from app.store.directory import ROLE_MANAGER

    company = directory.create_company("Acme")
    finance = directory.create_department(company, "Finance")
    it = directory.create_department(company, "IT")
    pera = directory.create_user(company, "pera@acme.rs")
    zika = directory.create_user(company, "zika@acme.rs")
    directory.add_membership(pera, finance, ROLE_MANAGER)
    directory.add_membership(zika, it)
    return {"company": company, "finance": finance, "it": it, "pera": pera, "zika": zika}


def test_scope_for_carries_company_and_every_membership(directory, acme):
    scope = directory.scope_for(acme["pera"])

    assert scope.company_id == acme["company"]
    assert scope.user_id == acme["pera"]
    assert scope.department_ids == (acme["finance"],)


def test_scope_for_a_user_in_several_departments(directory, acme):
    directory.add_membership(acme["pera"], acme["it"])

    scope = directory.scope_for(acme["pera"])

    assert set(scope.department_ids) == {acme["finance"], acme["it"]}


def test_scope_for_an_unknown_user_raises_rather_than_returning_an_empty_scope(directory):
    """An empty scope would be a valid-looking object that quietly matches
    nothing -- or worse, gets widened later. Missing users are an error."""
    with pytest.raises(LookupError):
        directory.scope_for("nobody")


def test_a_user_with_no_memberships_gets_a_scope_with_no_departments(directory, acme):
    solo = directory.create_user(acme["company"], "new@acme.rs")

    scope = directory.scope_for(solo)

    assert scope.department_ids == ()
    assert scope.company_id == acme["company"]


def test_manager_role_is_recorded_per_department(directory, acme):
    assert directory.manages(acme["pera"], acme["finance"]) is True
    assert directory.manages(acme["pera"], acme["it"]) is False
    assert directory.manages(acme["zika"], acme["it"]) is False, "membership is not management"
    assert directory.managed_departments(acme["pera"]) == [acme["finance"]]


def test_email_is_unique_across_the_whole_directory(directory, acme):
    """One address, one account: login happens before we know which company is
    being asked for, so a per-company unique index would make it ambiguous."""
    import psycopg

    other = directory.create_company("Globex")
    with pytest.raises(psycopg.errors.UniqueViolation):
        directory.create_user(other, "pera@acme.rs")


def test_email_is_normalised_on_the_way_in(directory, acme):
    uid = directory.create_user(acme["company"], "  MiXeD@Acme.RS  ")

    found = directory.find_user_by_email("mixed@acme.rs")

    assert found is not None and found["id"] == uid


def test_documents_record_their_owner_and_sharing_state(directory, acme):
    doc_id = directory.record_document(
        company_id=acme["company"], owner_id=acme["zika"], source="specs.md",
        department_id=acme["it"], scope="department", status="pending_approval")

    with directory._conn() as conn:
        row = conn.execute(
            "SELECT owner_id, department_id, scope, status FROM documents WHERE id = %s",
            (doc_id,)).fetchone()

    assert row == (acme["zika"], acme["it"], "department", "pending_approval")


def test_deleting_a_company_takes_its_people_with_it(directory, acme):
    """No orphaned users pointing at a company that no longer exists -- an
    orphan is a row whose company_id can never match anyone's scope again."""
    with directory._conn() as conn:
        conn.execute("DELETE FROM companies WHERE id = %s", (acme["company"],))
        remaining = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    assert remaining == 0
