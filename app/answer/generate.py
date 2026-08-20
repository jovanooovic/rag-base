from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.providers import Message
from ..store.base import ScoredChunk

ANSWER_SYSTEM = """You answer questions using ONLY the numbered sources provided.

Rules, in priority order:
1. If the sources do not contain the answer, reply exactly:
   NOT_IN_SOURCES
   followed by one sentence saying what information would be needed.
   Never answer from your own knowledge. Never guess.
2. Cite every factual claim with the source number in square brackets, like [2].
   A sentence with no citation must contain no facts.
3. If sources disagree, say so explicitly and cite both.
4. Be concise. Do not restate the question."""

NOT_FOUND_MARKER = "NOT_IN_SOURCES"


@dataclass
class Citation:
    number: int
    chunk_id: str
    source: str
    heading_path: str
    text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    used_citations: list[int] = field(default_factory=list)
    answered: bool = True
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def unsupported(self) -> bool:
        """True when the model wrote prose but cited nothing.

        Worth surfacing in the API response: it is the signal that correlates
        best with hallucination in practice.
        """
        return self.answered and not self.used_citations


def build_context(hits: Sequence[ScoredChunk], *, max_chars: int = 8000) -> tuple[str, list[Citation]]:
    """Render retrieved chunks as a numbered source block.

    Numbering starts at 1 because models cite [1]-based lists far more reliably
    than [0]-based ones -- a small thing that measurably reduces broken citations.
    """
    parts: list[str] = []
    citations: list[Citation] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        block = f"[{i}] ({hit.chunk.citation_text})\n{hit.chunk.text}"
        if used + len(block) > max_chars and citations:
            break
        used += len(block)
        parts.append(block)
        citations.append(Citation(i, hit.chunk.chunk_id, hit.chunk.source,
                                  hit.chunk.heading_path, hit.chunk.text))
    return "\n\n".join(parts), citations


def answer_question(
    llm,
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    history: Sequence[Message] | None = None,
    max_context_chars: int = 8000,
    trace=None,
) -> Answer:
    if not hits:
        return Answer(text=f"{NOT_FOUND_MARKER} No documents matched this question.",
                      answered=False)

    context, citations = build_context(hits, max_chars=max_context_chars)
    messages: list[Message] = [Message.system(ANSWER_SYSTEM), Message.system(f"Sources:\n{context}")]
    if history:
        messages += [m for m in list(history)[-6:] if m.role in ("user", "assistant")]
    messages.append(Message.user(question))

    span = trace.span("answer") if trace else _null()
    with span:
        resp = llm.chat(messages)

    text = resp.text.strip()
    answered = not text.upper().startswith(NOT_FOUND_MARKER)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                   if 1 <= int(n) <= len(citations)})
    return Answer(
        text=text,
        citations=[c for c in citations if c.number in used] or citations,
        used_citations=used,
        answered=answered,
        usage={"tokens_in": resp.usage.tokens_in, "tokens_out": resp.usage.tokens_out,
               "cost_usd": resp.usage.cost_usd},
    )


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
