from __future__ import annotations

from typing import Any, Iterator


def bootstrap_from_corpus(chunks, llm, *, per_chunk: int = 1, limit: int = 30) -> Iterator[dict[str, Any]]:
    """Generate a starter golden set from the client's own documents.

    Use this to get to a measurable baseline on day one, then hand the JSONL to
    the client's subject-matter expert to correct. Machine-written questions are
    a scaffold, not a ground truth -- say so explicitly when you hand it over,
    because a golden set the client never reviewed will flatter you and then
    fail in production.
    """
    from ..core.providers import Message as M
    system = M.system(
        "Write one specific factual question that the passage fully answers. "
        "The question must be answerable ONLY from this passage: no pronouns, "
        "no 'according to the text'. Output the question and nothing else.")
    for chunk in list(chunks)[:limit]:
        for _ in range(per_chunk):
            q = llm.chat([system, M.user(chunk.text[:1500])]).text.strip()
            if q:
                yield {
                    "id": f"auto-{chunk.chunk_id}",
                    "question": q,
                    "gold_doc_ids": [chunk.source],
                    "type": "factoid",
                    "difficulty": "easy",
                    "notes": "auto-generated, needs review",
                }
