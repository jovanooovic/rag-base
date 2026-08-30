from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.config import Settings
from ..core.providers import build_embeddings
from ..store.access import DocumentACL
from ..store.base import VectorStore
from .chunking import Chunk, chunk_documents
from .loaders import Document, load_path


@dataclass
class IngestReport:
    documents: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_unchanged: int = 0
    total_chunks_in_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# Bump when a change must reach chunks whose *text* is unchanged -- the
# fingerprint is text-only, so new metadata (effective_date, say) would
# otherwise be skipped forever on an existing index and the feature would look
# broken on every machine that had ingested before.
#
# The alternative, folding metadata into the fingerprint, is worse: it would
# put ingestion timestamps in the hash and re-embed the entire corpus nightly,
# which is the exact cost this class exists to avoid. A version bump forces one
# reindex, deliberately, at a moment someone chose.
STATE_VERSION = 2


def _fingerprint(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode()).hexdigest()[:16]


class Ingestor:
    """Idempotent ingestion with content-hash change detection.

    Re-running ingestion on an unchanged corpus must cost nothing. Clients
    re-sync nightly; embedding the whole corpus every night is the difference
    between a $12/month bill and a $900/month one, and it is the first thing
    they notice on the invoice.
    """

    def __init__(self, store: VectorStore, settings: Settings, embeddings=None,
                 state_path: str | Path | None = None):
        self.store = store
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(settings)
        self.state_path = Path(state_path or Path(settings.data_dir) / "ingest_state.json")
        self.state: dict[str, str] = {}
        if self.state_path.is_file():
            raw = json.loads(self.state_path.read_text())
            # Older state files are a bare {chunk_id: fingerprint} mapping with
            # no version, which is exactly the shape that needs discarding.
            if isinstance(raw, dict) and raw.get("version") == STATE_VERSION:
                self.state = dict(raw.get("chunks", {}))

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"version": STATE_VERSION, "chunks": self.state}, indent=2))

    def ingest_documents(self, docs: Iterable[Document], *, batch_size: int = 64,
                         chunk_kwargs: dict[str, Any] | None = None,
                         acl: "DocumentACL | None" = None) -> IngestReport:
        docs = list(docs)
        chunk_kwargs = dict(chunk_kwargs or {})
        if chunk_kwargs.get("strategy") == "semantic":
            chunk_kwargs.setdefault("embeddings", self.embeddings)
        chunks = chunk_documents(docs, **chunk_kwargs)
        if acl is not None:
            # Stamped after chunking so it lands on every chunk of the
            # document, including ones a splitter created.
            for c in chunks:
                c.metadata.update(acl.as_metadata())
        report = IngestReport(documents=len(docs), chunks_seen=len(chunks))

        pending: list[Chunk] = []
        for c in chunks:
            fp = _fingerprint(c)
            if self.state.get(c.chunk_id) == fp:
                report.chunks_skipped_unchanged += 1
                continue
            self.state[c.chunk_id] = fp
            pending.append(c)

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            # Embed heading + body: the heading is often the only place the
            # topic word appears, and losing it tanks recall on nested docs.
            payload = [f"{c.heading_path}\n{c.text}".strip() for c in batch]
            vectors = self.embeddings.embed(payload)
            self.store.upsert(batch, vectors)
            report.chunks_embedded += len(batch)

        self._save_state()
        report.total_chunks_in_index = self.store.count()
        return report

    def ingest_path(self, path: str | Path, **kw: Any) -> IngestReport:
        return self.ingest_documents(load_path(path), **kw)
