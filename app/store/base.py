from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..ingest.chunking import Chunk
from .access import AccessScope


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Where the score came from -- kept so eval reports can attribute wins to
    # the vector leg, the keyword leg, or the reranker.
    signals: dict[str, float] = field(default_factory=dict)


class VectorStore(Protocol):
    """`where` is the caller's own filter; `access` is the security boundary.

    They are separate parameters on purpose -- see app/store/access.py. A store
    that cannot enforce an AccessScope must raise when given one rather than
    ignore it: a filter that silently does nothing is worse than one that is
    absent, because the caller believes it is there.
    """
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...
    def search(self, vector: Sequence[float], k: int = 10,
               where: dict[str, Any] | None = None,
               access: AccessScope | None = None) -> list[ScoredChunk]: ...
    def keyword_search(self, query: str, k: int = 10,
                       where: dict[str, Any] | None = None,
                       access: AccessScope | None = None) -> list[ScoredChunk]: ...
    def all_chunks(self, access: AccessScope | None = None) -> list[Chunk]: ...
    def count(self, access: AccessScope | None = None) -> int: ...
    def delete_document(self, doc_id: str) -> int: ...
