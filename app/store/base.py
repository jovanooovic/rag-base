from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..ingest.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Where the score came from -- kept so eval reports can attribute wins to
    # the vector leg, the keyword leg, or the reranker.
    signals: dict[str, float] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...
    def search(self, vector: Sequence[float], k: int = 10,
               where: dict[str, Any] | None = None) -> list[ScoredChunk]: ...
    def keyword_search(self, query: str, k: int = 10,
                       where: dict[str, Any] | None = None) -> list[ScoredChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...
    def delete_document(self, doc_id: str) -> int: ...
