from __future__ import annotations

import json
import re
from typing import Sequence

from ..core.providers import Message
from ..store.base import ScoredChunk

RERANK_SYSTEM = """You score how well each passage answers a question.

Return ONLY a JSON array of objects: [{"id": <int>, "score": <0-10>}, ...]
- 10: passage directly and completely answers the question
- 5: passage is on-topic and contains part of the answer
- 0: passage is irrelevant, or merely shares vocabulary with the question

Score every passage you are given. Do not explain."""


class LLMReranker:
    """Listwise reranker using the chat model you already pay for.

    A dedicated cross-encoder (bge-reranker, Cohere rerank) is better and
    cheaper at volume, and `CrossEncoderReranker` below is the drop-in slot for
    it. This LLM version exists because it needs no extra vendor, no extra key,
    and no GPU -- which means it works on day one of a client engagement, and
    the eval harness can prove whether the upgrade is worth buying.
    """

    def __init__(self, llm, *, batch_size: int = 12, max_chars: int = 700):
        self.llm = llm
        self.batch_size = batch_size
        self.max_chars = max_chars

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int = 5) -> list[ScoredChunk]:
        out: list[ScoredChunk] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = list(candidates[start:start + self.batch_size])
            passages = "\n\n".join(
                f"[{i}] {sc.chunk.text[:self.max_chars]}" for i, sc in enumerate(batch)
            )
            resp = self.llm.chat([
                Message.system(RERANK_SYSTEM),
                Message.user(f"Question: {query}\n\nPassages:\n{passages}"),
            ])
            scores = _parse_scores(resp.text, len(batch))
            for i, sc in enumerate(batch):
                sc.signals["rerank"] = scores[i]
                # Fused score kept as a tiebreaker so a model that returns all
                # zeros degrades to the retrieval order instead of to random.
                sc.score = scores[i] + min(sc.score, 0.999) * 0.001
                out.append(sc)
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:top_k]


def _parse_scores(text: str, n: int) -> list[float]:
    """Tolerant parse. A reranker must never take the pipeline down."""
    scores = [0.0] * n
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            for item in json.loads(match.group(0)):
                i = int(item["id"])
                if 0 <= i < n:
                    scores[i] = float(item["score"])
            return scores
        except (ValueError, KeyError, TypeError):
            pass
    for i, s in re.findall(r"\[?(\d+)\]?\D{0,12}?(\d+(?:\.\d+)?)", text):
        idx = int(i)
        if 0 <= idx < n:
            scores[idx] = float(s)
    return scores


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    """Slot for a real cross-encoder once a client's volume justifies it.

        pip install sentence-transformers
        reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")

    Measure it against LLMReranker with `make eval` (or the ablation harness in
    `eval/ablations.py`) before you recommend it. Sometimes it wins by 15 points
    of nDCG; sometimes it does not move at all, and then you have saved the
    client an inference server.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder  # type: ignore
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int = 5) -> list[ScoredChunk]:
        pairs = [(query, sc.chunk.text) for sc in candidates]
        for sc, score in zip(candidates, self.model.predict(pairs)):
            sc.score = float(score)
            sc.signals["rerank"] = float(score)
        return sorted(candidates, key=lambda s: s.score, reverse=True)[:top_k]
