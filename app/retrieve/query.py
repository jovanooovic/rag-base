from __future__ import annotations

from typing import Sequence

from ..core.providers import Message

REWRITE_SYSTEM = """Rewrite the user's question into 1-3 standalone search queries.

Rules:
- Resolve pronouns and references using the conversation history.
- Keep exact identifiers (order numbers, SKUs, error codes, names) verbatim.
- One query per line. No numbering, no commentary."""


def rewrite(llm, question: str, history: Sequence[Message] | None = None, *, max_queries: int = 3) -> list[str]:
    """Turn a follow-up like "and what about refunds?" into a searchable query.

    Multi-turn is where most demo RAG systems fall over: the second question
    embeds terribly on its own because the subject lives in the first turn.
    Every client chatbot posting eventually hits this.
    """
    convo = ""
    if history:
        convo = "\n".join(f"{m.role}: {m.content}" for m in list(history)[-6:] if m.role in ("user", "assistant"))
    resp = llm.chat([
        Message.system(REWRITE_SYSTEM),
        Message.user((f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Question: {question}"),
    ])
    lines = [line.strip(" -*\t") for line in resp.text.splitlines() if line.strip()]
    queries = [line for line in lines if len(line) > 2][:max_queries]
    return queries or [question]
