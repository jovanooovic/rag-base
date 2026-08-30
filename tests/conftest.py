import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings          # noqa: E402
from app.pipeline import RAGPipeline          # noqa: E402
from app.store.sqlite_store import SQLiteStore  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_name="test",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dim=256,
        data_dir=str(tmp_path),
        trace_enabled=False,
        extra={"use_reranker": False, "top_k": 5, "fetch_k": 20},
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "index.db")


@pytest.fixture
def pipeline(settings, store):
    p = RAGPipeline(settings, store=store)
    p.ingest("data/sample")
    return p


# ---------------------------------------------------------------- Postgres

POSTGRES_ADMIN_DSN = os.environ.get("RAG_TEST_POSTGRES_DSN", "")
_APP_ROLE, _APP_PASSWORD = "rag_test_app", "app"

requires_postgres = pytest.mark.skipif(
    not POSTGRES_ADMIN_DSN, reason="set RAG_TEST_POSTGRES_DSN to run")

# Directory tables plus the chunk tables the suites create, children first so
# foreign keys do not block the drop.
_MANAGED_TABLES = ("memberships", "documents", "departments", "users", "companies")


def postgres_app_dsn() -> str:
    """A DSN for a role that does NOT bypass RLS, owning everything it creates.

    Two things have to be true at once and they pull against each other: the
    role must not be a superuser (superusers ignore row-level security, so
    tests would report isolation working while it does nothing), and it must
    own its tables (FORCE ROW LEVEL SECURITY applies to the owner). So the
    admin connection creates the role and drops anything a previous run left
    owned by postgres, and the role creates its own tables from there.
    """
    import psycopg

    with psycopg.connect(POSTGRES_ADMIN_DSN, autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s",
                             (_APP_ROLE,)).fetchone():
            admin.execute(f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD '{_APP_PASSWORD}'")
        admin.execute(f"GRANT CREATE, USAGE ON SCHEMA public TO {_APP_ROLE}")
        for table in _MANAGED_TABLES:
            owner = admin.execute(
                "SELECT pg_get_userbyid(relowner) FROM pg_class WHERE relname = %s",
                (table,)).fetchone()
            if owner and owner[0] != _APP_ROLE:
                admin.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    return re.sub(r"//[^@]+@", f"//{_APP_ROLE}:{_APP_PASSWORD}@", POSTGRES_ADMIN_DSN)
