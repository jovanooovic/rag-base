from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..store.base import ScoredChunk
from .generate import Answer

# Redact before text leaves the process -- into a log, a trace file, or a
# third-party model. Clients in finance, health and legal ask for this in the
# first call; having it already written is worth real money in a bid.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3}[ -]?\d{3,4}[ -]?\d{0,4}(?!\d)")),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


def redact(text: str) -> str:
    for label, pattern in PII_PATTERNS:
        text = pattern.sub(f"<{label}>", text)
    return text


@dataclass
class GuardrailResult:
    ok: bool
    reason: str = ""
    answer: Answer | None = None


def check_answer(
    answer: Answer,
    hits: Sequence[ScoredChunk],
    *,
    min_top_score: float | None = None,
    require_citations: bool = True,
    refusal_text: str = ("I could not find this in the available documents. "
                         "Please rephrase, or add the relevant document to the knowledge base."),
) -> GuardrailResult:
    """Decide whether an answer is safe to return.

    The point is to fail closed. A RAG system that says "I don't know" is
    annoying; one that confidently invents a refund policy gets you fired and
    is what the client's previous contractor did.
    """
    if answer.needs_clarification:
        # Not a refusal: the model found relevant sources but the question
        # doesn't say which case applies. No citations to require and no
        # score floor to clear -- nothing has been claimed yet.
        return GuardrailResult(True, "", answer)

    if not answer.answered:
        return GuardrailResult(False, "model reported the sources do not contain the answer",
                               Answer(refusal_text, answered=False))

    if min_top_score is not None:
        # signals["confidence"], not .score. The raw score's scale depends on
        # which retrieval path ran -- roughly 0-10 from the LLM reranker, but
        # capped near 0.033 for pure RRF fusion -- so one configured threshold
        # meant two incompatible things, and the intake form's "strictest"
        # preset of 0.15 was simultaneously unreachable in one mode and
        # never-triggered in the other. Every scorer now also publishes a
        # normalised 0-1 confidence, and the floor is defined on that.
        scored = [h for h in hits if "confidence" in h.signals]
        if hits and not scored:
            # Fail closed, but say which failure this is. Silently treating a
            # missing signal as 0.0 makes a mis-wired retriever look exactly
            # like a corpus that has nothing to say.
            return GuardrailResult(False,
                                   "retrieval produced no confidence signal, so min_top_score "
                                   "cannot be evaluated (custom retriever?)",
                                   Answer(refusal_text, answered=False))
        top = max((h.signals["confidence"] for h in scored), default=0.0)
        if top < min_top_score:
            return GuardrailResult(False,
                                   f"top retrieval confidence {top:.3f} below floor {min_top_score}",
                                   Answer(refusal_text, answered=False))

    if require_citations and not answer.used_citations:
        return GuardrailResult(False, "answer contained no citations",
                               Answer(refusal_text, answered=False))

    return GuardrailResult(True, "", answer)
