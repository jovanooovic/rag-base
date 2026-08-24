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
2. If the sources contain more than one answer that depends on something the
   question does not specify -- which product, which membership tier, which
   region, which order -- and picking one would be a guess, reply exactly:
   NEEDS_CLARIFICATION
   followed by one short question asking for the missing detail. This is not
   the same as sources disagreeing (rule 4): here every source is correct for
   its own case, the question just doesn't say which case applies.
3. Cite every factual claim with the source number in square brackets, like [2].
   A sentence with no citation must contain no facts.
4. If sources actually contradict each other on the same case, say so
   explicitly and cite both -- don't ask the user to resolve that for you.
5. Be concise. Do not restate the question."""

NOT_FOUND_MARKER = "NOT_IN_SOURCES"
NEEDS_CLARIFICATION_MARKER = "NEEDS_CLARIFICATION"


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
    needs_clarification: bool = False
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
    needs_clarification = text.upper().startswith(NEEDS_CLARIFICATION_MARKER)
    answered = not (text.upper().startswith(NOT_FOUND_MARKER) or needs_clarification)
    if needs_clarification:
        # Unlike NOT_IN_SOURCES (always swapped for a canned refusal downstream),
        # the clarifying question itself is what the user should see -- strip the
        # protocol marker, keep the question.
        text = text[len(NEEDS_CLARIFICATION_MARKER):].strip()
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                   if 1 <= int(n) <= len(citations)})
    return Answer(
        text=text,
        citations=[c for c in citations if c.number in used] or citations,
        used_citations=used,
        answered=answered,
        needs_clarification=needs_clarification,
        usage={"tokens_in": resp.usage.tokens_in, "tokens_out": resp.usage.tokens_out,
               "cost_usd": resp.usage.cost_usd},
    )


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
