from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from .loaders import Document

CHUNKING_STRATEGIES = ("structure-first", "fixed-512", "fixed-1024", "recursive-overlap", "semantic")


class _EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    ordinal: int
    heading_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_text(self) -> str:
        """What the model sees as the chunk's provenance."""
        loc = f" > {self.heading_path}" if self.heading_path else ""
        return f"{self.source}{loc}"


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (heading_path, body) sections.

    Chunking on structure first, size second, is the single highest-leverage
    retrieval decision in this codebase. A chunk that stops mid-clause retrieves
    badly no matter how good the embedding model is.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]

    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            out.append(("", preamble))

    stack: list[str] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        stack = stack[: level - 1] + [title]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end(): end].strip()
        if body:
            out.append((" > ".join(stack), body))
    return out


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 120,
    strategy: str = "structure-first",
    embeddings: _EmbeddingClient | None = None,
) -> list[tuple[str, str]]:
    """Return [(heading_path, chunk_text)] using one of CHUNKING_STRATEGIES.

    `structure-first` (the default, unchanged from before `strategy` existed) is the
    production default -- see `_chunk_structure_first` for why. The other strategies
    exist for the ablation harness (`eval/ablations.py`) to measure whether that
    choice actually earns its keep on a given corpus, not because any of them is
    assumed better.
    """
    if strategy == "structure-first":
        return _chunk_structure_first(text, target_chars=target_chars,
                                      overlap_chars=overlap_chars, min_chars=min_chars)
    if strategy in ("fixed-512", "fixed-1024"):
        size = 512 if strategy == "fixed-512" else 1024
        return _chunk_fixed(text, target_chars=size, overlap_chars=overlap_chars)
    if strategy == "recursive-overlap":
        return _chunk_recursive_overlap(text, target_chars=target_chars, overlap_chars=overlap_chars)
    if strategy == "semantic":
        if embeddings is None:
            raise ValueError("strategy='semantic' requires an embeddings client")
        return _chunk_semantic(text, embeddings, target_chars=target_chars)
    raise ValueError(f"unknown chunking strategy {strategy!r}; choose one of {CHUNKING_STRATEGIES}")


def _chunk_structure_first(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 120,
) -> list[tuple[str, str]]:
    """Algorithm: sections -> paragraphs -> sentences, packing greedily up to
    `target_chars` and never splitting inside a sentence unless a single
    sentence is itself over target. Overlap is carried as whole trailing
    sentences, not a raw character slice, so the overlap is readable context.
    Never packs across a section boundary -- each heading starts a fresh chunk.
    """
    chunks: list[tuple[str, str]] = []
    for heading, body in _sections(text):
        buf: list[str] = []
        size = 0

        def flush() -> None:
            nonlocal buf, size
            joined = "\n\n".join(buf).strip()
            if joined:
                chunks.append((heading, joined))
            buf, size = [], 0

        for para in _paragraphs(body):
            if len(para) > target_chars:
                flush()
                sent_buf: list[str] = []
                sent_size = 0
                for sent in _sentences(para):
                    if sent_size + len(sent) > target_chars and sent_buf:
                        chunks.append((heading, " ".join(sent_buf).strip()))
                        tail, tail_size = [], 0
                        for s in reversed(sent_buf):
                            if tail_size + len(s) > overlap_chars:
                                break
                            tail.insert(0, s)
                            tail_size += len(s)
                        sent_buf, sent_size = list(tail), tail_size
                    sent_buf.append(sent)
                    sent_size += len(sent) + 1
                if sent_buf:
                    chunks.append((heading, " ".join(sent_buf).strip()))
                continue

            if size + len(para) > target_chars and buf:
                flush()
            buf.append(para)
            size += len(para) + 2
        flush()

    # Fold away runt chunks -- they pollute retrieval with high-similarity noise.
    merged: list[tuple[str, str]] = []
    for heading, body in chunks:
        if merged and len(body) < min_chars and merged[-1][0] == heading:
            merged[-1] = (heading, merged[-1][1] + "\n\n" + body)
        else:
            merged.append((heading, body))
    return merged


def _chunk_fixed(text: str, *, target_chars: int, overlap_chars: int) -> list[tuple[str, str]]:
    """Pure fixed-size character windows, ignoring document structure entirely --
    the ablation harness's naive baseline. This can and will cut mid-sentence or
    mid-word; that is the point of comparing it against the structure-aware
    strategies rather than a flaw to work around here.
    """
    body = text.strip()
    if not body:
        return []
    step = max(1, target_chars - overlap_chars)
    out: list[tuple[str, str]] = []
    for start in range(0, len(body), step):
        piece = body[start:start + target_chars].strip()
        if piece:
            out.append(("", piece))
        if start + target_chars >= len(body):
            break
    return out


def _strip_headings(text: str) -> str:
    return _HEADING.sub("", text)


def _chunk_recursive_overlap(text: str, *, target_chars: int, overlap_chars: int) -> list[tuple[str, str]]:
    """Sentence-level recursive packing across the whole document, carrying overlap
    forward as trailing sentences.

    Unlike `_chunk_structure_first`, this never resets at a section heading -- a
    chunk can span two sections if that is what it takes to fill target_chars. That
    is exactly the question the ablation harness exists to answer: does keeping
    strict section boundaries actually help retrieval, or does it just leave
    under-sized chunks at the end of short sections.
    """
    sentences = _sentences(_strip_headings(text))
    if not sentences:
        return []
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        if size + len(sent) > target_chars and buf:
            chunks.append(("", " ".join(buf).strip()))
            tail, tail_size = [], 0
            for s in reversed(buf):
                if tail_size + len(s) > overlap_chars:
                    break
                tail.insert(0, s)
                tail_size += len(s)
            buf, size = list(tail), tail_size
        buf.append(sent)
        size += len(sent) + 1
    if buf:
        chunks.append(("", " ".join(buf).strip()))
    return chunks


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _chunk_semantic(text: str, embeddings: _EmbeddingClient, *, target_chars: int,
                    similarity_threshold: float = 0.5) -> list[tuple[str, str]]:
    """Embed each sentence and start a new chunk wherever the cosine similarity to
    the previous sentence drops below `similarity_threshold` -- an adaptive boundary
    instead of a fixed character target, so a chunk ends where the topic actually
    shifts rather than where a character counter runs out. Still falls back to
    closing a chunk at target_chars regardless, so one on-topic run can't grow
    unbounded and blow the context budget downstream.
    """
    sentences = _sentences(_strip_headings(text))
    if not sentences:
        return []
    vectors = embeddings.embed(sentences)
    chunks: list[tuple[str, str]] = []
    buf = [sentences[0]]
    size = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = _cosine(vectors[i - 1], vectors[i])
        sent = sentences[i]
        if buf and (sim < similarity_threshold or size + len(sent) > target_chars):
            chunks.append(("", " ".join(buf).strip()))
            buf, size = [], 0
        buf.append(sent)
        size += len(sent) + 1
    if buf:
        chunks.append(("", " ".join(buf).strip()))
    return chunks


def chunk_documents(docs: Iterable[Document], **kw: Any) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for i, (heading, body) in enumerate(chunk_text(doc.text, **kw)):
            out.append(Chunk(
                chunk_id=f"{doc.doc_id}#{i}",
                doc_id=doc.doc_id,
                text=body,
                source=doc.source,
                ordinal=i,
                heading_path=heading,
                metadata=dict(doc.metadata),
            ))
    return out
