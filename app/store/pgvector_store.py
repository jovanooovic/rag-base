from __future__ import annotations

import json
from typing import Any, Sequence

from ..ingest.chunking import Chunk
from .access import SCOPE_DEPARTMENT, SCOPE_PRIVATE, STATUS_ACTIVE, AccessScope
from .base import ScoredChunk


class PgVectorStore:  # pragma: no cover - needs a live Postgres
    """Postgres + pgvector + native full-text search.

    Use this when: the corpus is over ~100k chunks, several processes write
    concurrently, or the client already runs Postgres and wants one datastore.
    Requires `pip install psycopg[binary]` and the pgvector extension.

    Note this uses Postgres `tsvector` for the keyword leg rather than the
    Python BM25 -- at this scale you want the index doing the work. Ranking
    differs slightly from SQLiteStore, so re-run the eval suite after switching.
    """

    def __init__(self, dsn: str, *, dim: int = 1536, table: str = "chunks"):
        import psycopg  # type: ignore
        self.dim = dim
        self.table = table
        self.conn = psycopg.connect(dsn, autocommit=True)
        self._iterative_scan = self._enable_iterative_scan()
        self._migrate()

    def _enable_iterative_scan(self) -> bool:
        """Make filtered vector search return the k rows it was asked for.

        HNSW searches the index first and applies the WHERE clause to whatever
        it found, so a selective filter throws most of the result away and the
        query returns short -- silently, with no error. Measured on this store:
        4000 rows, filter matching 100 of them, k=10, index forced -> **1 row
        back** with this off, 10 with it on.

        It only bites once the planner actually chooses the index. Below a few
        thousand rows a sequential scan is cheaper, filtering is exact, and
        everything looks correct -- so this passes every small-corpus test and
        starts losing results on the client's production corpus instead.

        strict_order rather than relaxed_order: relaxed can return rows
        slightly out of distance order, and those distances feed RRF ranking
        upstream. Needs pgvector >= 0.8; older servers reject the parameter,
        which is handled in search() rather than here -- an unfiltered corpus
        is unaffected, so refusing to start would punish people this cannot
        hurt.
        """
        import psycopg  # type: ignore
        try:
            self.conn.execute("SET hnsw.iterative_scan = strict_order")
            return True
        except psycopg.errors.UndefinedObject:
            return False

    def _migrate(self) -> None:
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                text         TEXT NOT NULL,
                source       TEXT NOT NULL,
                ordinal      INT  NOT NULL,
                heading_path TEXT DEFAULT '',
                metadata     JSONB DEFAULT '{{}}'::jsonb,
                embedding    vector({self.dim}),
                tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                -- Access control as real columns, not metadata keys: these are
                -- indexed, they are what RLS policies will read, and a typo in
                -- a JSONB key silently widens visibility where a typo in a
                -- column name is a query error.
                company_id    TEXT,
                owner_id      TEXT,
                department_id TEXT,
                scope         TEXT NOT NULL DEFAULT 'private',
                status        TEXT NOT NULL DEFAULT 'active'
            )""")
        # Older installs created before the ACL columns existed.
        for col, ddl in (
            ("company_id", "TEXT"), ("owner_id", "TEXT"), ("department_id", "TEXT"),
            ("scope", "TEXT NOT NULL DEFAULT 'private'"),
            ("status", "TEXT NOT NULL DEFAULT 'active'"),
        ):
            self.conn.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS {col} {ddl}")
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_tsv_idx ON {self.table} USING GIN (tsv)")
        # Ordered company-first: every access-scoped query filters on it, so it
        # is the most selective prefix available to the planner.
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_acl_idx ON {self.table} "
            f"(company_id, department_id, owner_id)")
        # HNSW over IVFFlat: no training step, better recall at the same latency,
        # and it does not need rebuilding as the corpus grows.
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_emb_idx ON {self.table} "
            f"USING hnsw (embedding vector_cosine_ops)")

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """ACL fields ride in chunk metadata at ingest and are promoted to
        columns here, so the ingest pipeline stays storage-agnostic."""
        rows = []
        for c, v in zip(chunks, vectors, strict=True):
            m = c.metadata
            rows.append((c.chunk_id, c.doc_id, c.text, c.source, c.ordinal, c.heading_path,
                         json.dumps(m), str(list(v)),
                         m.get("company_id"), m.get("owner_id"), m.get("department_id"),
                         m.get("scope", SCOPE_PRIVATE), m.get("status", STATUS_ACTIVE)))
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} "
                "(chunk_id, doc_id, text, source, ordinal, heading_path, metadata, embedding, "
                " company_id, owner_id, department_id, scope, status) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET text=EXCLUDED.text, "
                "embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata, "
                "heading_path=EXCLUDED.heading_path, company_id=EXCLUDED.company_id, "
                "owner_id=EXCLUDED.owner_id, department_id=EXCLUDED.department_id, "
                "scope=EXCLUDED.scope, status=EXCLUDED.status", rows)
        return len(rows)

    def _row_to_scored(self, r, score: float, signal: str) -> ScoredChunk:
        return ScoredChunk(
            Chunk(chunk_id=r[0], doc_id=r[1], text=r[2], source=r[3], ordinal=r[4],
                  heading_path=r[5], metadata=r[6] or {}),
            score, {signal: score})

    def search(self, vector, k: int = 10, where: dict[str, Any] | None = None,
               access: AccessScope | None = None) -> list[ScoredChunk]:
        if (where or access) and not self._iterative_scan:
            # Refused here rather than at connect: this is the exact moment the
            # missing feature would start dropping rows without saying so, and
            # a store used without filters is genuinely fine on 0.7.
            raise RuntimeError(
                "filtered vector search needs pgvector >= 0.8 (hnsw.iterative_scan); "
                "this server is older, and filtered searches would silently return "
                "fewer rows than requested once the corpus is large enough for the "
                "planner to use the HNSW index. Upgrade pgvector, or drop the filter."
            )
        clause, params = _predicate(where, access)
        sql = (f"SELECT chunk_id, doc_id, text, source, ordinal, heading_path, metadata, "
               f"1 - (embedding <=> %s::vector) AS score FROM {self.table} {clause} "
               f"ORDER BY embedding <=> %s::vector LIMIT %s")
        v = str(list(vector))
        rows = self.conn.execute(sql, [v, *params, v, k]).fetchall()
        return [self._row_to_scored(r, float(r[7]), "vector") for r in rows]

    def keyword_search(self, query: str, k: int = 10, where: dict[str, Any] | None = None,
                       access: AccessScope | None = None) -> list[ScoredChunk]:
        clause, meta_params = _predicate(where, access)
        match_clause = "tsv @@ plainto_tsquery('english', %s)"
        clause = f"{clause} AND {match_clause}" if clause else f"WHERE {match_clause}"
        sql = (f"SELECT chunk_id, doc_id, text, source, ordinal, heading_path, metadata, "
               f"ts_rank(tsv, plainto_tsquery('english', %s)) AS score FROM {self.table} "
               f"{clause} ORDER BY score DESC LIMIT %s")
        # Param order follows the SQL text: ts_rank's query, then metadata
        # filters, then the match clause's query, then the limit.
        rows = self.conn.execute(sql, [query, *meta_params, query, k]).fetchall()
        return [self._row_to_scored(r, float(r[7]), "bm25") for r in rows]

    def all_chunks(self, access: AccessScope | None = None) -> list[Chunk]:
        clause, params = _predicate(None, access)
        rows = self.conn.execute(
            f"SELECT chunk_id, doc_id, text, source, ordinal, heading_path, metadata "
            f"FROM {self.table} {clause}", params
        ).fetchall()
        return [Chunk(chunk_id=r[0], doc_id=r[1], text=r[2], source=r[3], ordinal=r[4],
                      heading_path=r[5], metadata=r[6] or {}) for r in rows]

    def count(self, access: AccessScope | None = None) -> int:
        clause, params = _predicate(None, access)
        return int(self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table} {clause}", params).fetchone()[0])

    def delete_document(self, doc_id: str) -> int:
        cur = self.conn.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))
        return cur.rowcount


def _predicate(where: dict[str, Any] | None,
               access: AccessScope | None = None) -> tuple[str, list[Any]]:
    """Compose the caller's metadata filter with the security boundary.

    Both are AND-ed, and the access half is built here rather than by callers
    so there is exactly one expression of "may this user see this row". The
    visibility rule it encodes is the one in app/store/access.py: same company,
    and either you own the row or a manager approved it for a department you
    are in.
    """
    parts, params = [], []
    for key, value in (where or {}).items():
        parts.append("metadata->>%s = %s")
        params += [key, str(value)]
    if access is not None:
        # `= ANY(%s)` over an empty array is false, which is the correct
        # reading: someone in no department sees only what they own.
        parts.append(
            "(company_id = %s AND (owner_id = %s OR "
            " (scope = %s AND status = %s AND department_id = ANY(%s))))")
        params += [access.company_id, access.user_id, SCOPE_DEPARTMENT, STATUS_ACTIVE,
                   list(access.department_ids)]
    return ("WHERE " + " AND ".join(parts)) if parts else "", params
