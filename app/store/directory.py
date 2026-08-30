"""Companies, departments, users, memberships, documents.

Separate from PgVectorStore on purpose: this module answers "who is this
person and what are they a member of", the store answers "which rows may that
person see". Keeping them in one class would blur the moment identity stops
being a lookup and becomes a boundary.

The one function that matters is `scope_for`: it is the single place a user id
becomes an AccessScope, so there is exactly one answer to "what may this user
see" and every caller gets the same one.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any

from .access import AccessScope

ROLE_MEMBER = "member"
ROLE_MANAGER = "manager"


def new_id() -> str:
    return uuid.uuid4().hex


class Directory:
    def __init__(self, dsn: str, *, max_size: int = 5):
        from psycopg_pool import ConnectionPool  # type: ignore
        self.pool = ConnectionPool(dsn, min_size=1, max_size=max_size, open=True)
        with self.pool.connection() as conn:
            self._migrate(conn)

    @contextmanager
    def _conn(self):
        with self.pool.connection() as conn:
            yield conn

    def _migrate(self, conn) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS departments (
                id         TEXT PRIMARY KEY,
                company_id TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                name       TEXT NOT NULL,
                UNIQUE (company_id, name)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                company_id    TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                -- Globally unique, not per-company: an address identifies one
                -- account, and login happens before we know which company is
                -- being asked for.
                email         TEXT NOT NULL UNIQUE,
                -- Filled in when authentication lands; nullable now so that
                -- does not need a second migration on live data.
                password_hash TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS memberships (
                user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                department_id TEXT NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
                role          TEXT NOT NULL DEFAULT '{ROLE_MEMBER}',
                PRIMARY KEY (user_id, department_id)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id            TEXT PRIMARY KEY,
                company_id    TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                owner_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                department_id TEXT REFERENCES departments(id) ON DELETE SET NULL,
                scope         TEXT NOT NULL DEFAULT 'private',
                status        TEXT NOT NULL DEFAULT 'active',
                source        TEXT NOT NULL,
                title         TEXT NOT NULL DEFAULT '',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        # Managers read this constantly to find what is waiting on them.
        conn.execute("CREATE INDEX IF NOT EXISTS documents_pending_idx ON documents "
                     "(company_id, department_id, status)")
        conn.commit()

    # -- writes ---------------------------------------------------------
    def create_company(self, name: str) -> str:
        cid = new_id()
        with self._conn() as conn:
            conn.execute("INSERT INTO companies (id, name) VALUES (%s, %s)", (cid, name))
        return cid

    def create_department(self, company_id: str, name: str) -> str:
        did = new_id()
        with self._conn() as conn:
            conn.execute("INSERT INTO departments (id, company_id, name) VALUES (%s, %s, %s)",
                         (did, company_id, name))
        return did

    def create_user(self, company_id: str, email: str, password_hash: str | None = None) -> str:
        uid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users (id, company_id, email, password_hash) VALUES (%s, %s, %s, %s)",
                (uid, company_id, email.strip().lower(), password_hash))
        return uid

    def add_membership(self, user_id: str, department_id: str, role: str = ROLE_MEMBER) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memberships (user_id, department_id, role) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, department_id) DO UPDATE SET role = EXCLUDED.role",
                (user_id, department_id, role))

    def record_document(self, *, company_id: str, owner_id: str, source: str,
                        department_id: str | None = None, scope: str = "private",
                        status: str = "active", title: str = "") -> str:
        doc_id = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents (id, company_id, owner_id, department_id, scope, "
                "status, source, title) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (doc_id, company_id, owner_id, department_id, scope, status, source, title))
        return doc_id

    # -- reads ----------------------------------------------------------
    def scope_for(self, user_id: str) -> AccessScope:
        """Turn a user id into the boundary their queries run inside.

        The only place this conversion happens. If a caller ever builds an
        AccessScope by hand from request data, the boundary becomes whatever
        the request said it should be.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT company_id FROM users WHERE id = %s", (user_id,)).fetchone()
            if row is None:
                raise LookupError(f"no such user: {user_id}")
            departments = conn.execute(
                "SELECT department_id FROM memberships WHERE user_id = %s", (user_id,)).fetchall()
        return AccessScope(company_id=row[0], user_id=user_id,
                           department_ids=tuple(d[0] for d in departments))

    def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, company_id, email, password_hash FROM users WHERE email = %s",
                (email.strip().lower(),)).fetchone()
        if row is None:
            return None
        return {"id": row[0], "company_id": row[1], "email": row[2], "password_hash": row[3]}

    def set_password_hash(self, user_id: str, password_hash: str) -> None:
        """Used to upgrade a hash in place when argon2's tuning moves on."""
        with self._conn() as conn:
            conn.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                         (password_hash, user_id))

    def manages(self, user_id: str, department_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM memberships WHERE user_id = %s AND department_id = %s "
                "AND role = %s", (user_id, department_id, ROLE_MANAGER)).fetchone()
        return row is not None

    def managed_departments(self, user_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT department_id FROM memberships WHERE user_id = %s AND role = %s",
                (user_id, ROLE_MANAGER)).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self.pool.close()
