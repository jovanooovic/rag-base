from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from ..ingest.chunking import Chunk
from ..retrieve.bm25 import BM25
from .base import ScoredChunk


class SQLiteStore:
    """Default store: SQLite + in-process cosine similarity.

    Chosen as the default deliberately. It has no infrastructure, it is fast
    enough to roughly a few hundred thousand chunks, and it means a client can
    run the thing on day one without provisioning anything. Swap to
    `PgVectorStore` when the corpus or the concurrency justifies it -- the
    interface is identical, so it is a one-line change in `build_store`.
    """

    def __init__(self, path: str | Path = "./data/index.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()
        self._bm25: BM25 | None = None

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                text         TEXT NOT NULL,
                source       TEXT NOT NULL,
                ordinal      INTEGER NOT NULL,
                heading_path TEXT DEFAULT '',
                metadata     TEXT DEFAULT '{}',
                vector       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self.conn.commit()

    # -- writes ---------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        rows = [
            (c.chunk_id, c.doc_id, c.text, c.source, c.ordinal, c.heading_path,
             json.dumps(c.metadata), json.dumps(list(v)))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, text, source, ordinal, heading_path, metadata, vector) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET text=excluded.text, vector=excluded.vector, "
            "metadata=excluded.metadata, heading_path=excluded.heading_path",
            rows,
        )
        self.conn.commit()
        self._bm25 = None
        return len(rows)

    def delete_document(self, doc_id: str) -> int:
        cur = self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        self._bm25 = None
        return cur.rowcount

    # -- reads ----------------------------------------------------------
    def _rows(self, where: dict[str, Any] | None = None) -> list[sqlite3.Row]:
        rows = self.conn.execute("SELECT * FROM chunks").fetchall()
        if not where:
            return rows
        cols = set(rows[0].keys()) if rows else set()
        out = []
        for r in rows:
            meta = json.loads(r["metadata"])
            if all((r[k] if k in cols else meta.get(k)) == v for k, v in where.items()):
                out.append(r)
        return out

    @staticmethod
    def _to_chunk(r: sqlite3.Row) -> Chunk:
        return Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], source=r["source"],
                     ordinal=r["ordinal"], heading_path=r["heading_path"],
                     metadata=json.loads(r["metadata"]))

    def _reject_access(self, access) -> None:
        """Fail closed.

        SQLiteStore is the single-tenant backend: one client, one corpus, no
        per-user visibility. If a caller hands it an AccessScope, the caller
        believes rows are being filtered by identity and they are not. Silently
        ignoring it would turn "Pera cannot see Zika's specs" into a comment in
        someone's design doc, so refuse instead and name the fix.
        """
        if access is not None:
            raise NotImplementedError(
                "SQLiteStore cannot enforce per-user access control. Set "
                "store_backend to 'pgvector' for the multi-user tier -- see "
                "app/store/access.py."
            )

    def search(self, vector, k: int = 10, where=None, access=None) -> list[ScoredChunk]:
        self._reject_access(access)
        q = list(vector)
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        scored: list[ScoredChunk] = []
        for r in self._rows(where):
            v = json.loads(r["vector"])
            dot = sum(a * b for a, b in zip(q, v, strict=True))
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            score = dot / (qn * vn)
            scored.append(ScoredChunk(self._to_chunk(r), score, {"vector": score}))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int = 10, where=None, access=None) -> list[ScoredChunk]:
        self._reject_access(access)
        chunks = [self._to_chunk(r) for r in self._rows(where)]
        if not chunks:
            return []
        if where is None:
            if self._bm25 is None:
                self._bm25 = BM25([c.text for c in chunks])
            bm = self._bm25
        else:
            bm = BM25([c.text for c in chunks])
        scores = bm.score(query)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda t: t[1], reverse=True)[:k]
        return [ScoredChunk(c, s, {"bm25": s}) for c, s in ranked if s > 0]

    def all_chunks(self, access=None) -> list[Chunk]:
        self._reject_access(access)
        return [self._to_chunk(r) for r in self._rows()]

    def count(self, access=None) -> int:
        self._reject_access(access)
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def close(self) -> None:
        """Release the sqlite3 connection.

        Not needed for the long-lived store the CLI/API/tests hold for a process's
        whole lifetime -- the OS reclaims it at exit either way. It matters for a
        short-lived store backed by a temp directory (see eval/ablations.py): on
        Windows, deleting a directory while a file inside it is still open raises
        PermissionError, unlike POSIX which allows unlinking an open file.
        """
        self.conn.close()
