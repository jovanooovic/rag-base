from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.core.providers import LLMClient

CORRECTNESS_SYSTEM = """You grade a candidate ANSWER against a GOLD answer for the same question.

Score on this rubric, one sentence per level:
0: the candidate is wrong, contradicts the gold answer, or answers a different question.
1: the candidate is mostly wrong but touches the right topic.
2: the candidate is partially correct but misses or garbles a required fact.
3: the candidate is correct but is missing a minor detail the gold answer includes.
4: the candidate is fully correct and equivalent in substance to the gold answer.

Return ONLY JSON: {"score": <0-4 integer>, "reason": "<one sentence>"}"""

FAITHFULNESS_SYSTEM = """You are grading whether an ANSWER is supported by SOURCES.

Decompose the answer into its atomic factual claims, then check each claim against the
sources. A claim is supported only if the sources state it; being plausible is not enough.

Return ONLY JSON: {"supported": <fraction of claims supported, 0.0-1.0>,
"claims": [{"claim": "...", "supported": true|false}]}"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant JSON-object extraction. A judge must never crash the run over a stray
    code fence or a trailing sentence."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


class JudgeCache:
    """SQLite cache keyed by hash(prompt + model + seed + vote), so re-running a suite
    whose judge calls have not changed costs nothing and CI stays cheap."""

    def __init__(self, path: str | Path = "./eval/.judge_cache.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.commit()

    @staticmethod
    def key(prompt: str, model: str, seed: int, vote: int) -> str:
        return hashlib.sha256(f"{model}|{seed}|{vote}|{prompt}".encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()


@dataclass
class JudgeVerdict:
    value: dict[str, Any]
    raw_votes: list[dict[str, Any]]
    disagreement_rate: float
    model: str


class Judge:
    """LLM-as-judge with a pinned model, temperature 0, majority-of-3 voting, and a
    local cache.

    A judge whose model floats with whatever the provider aliases to "latest" cannot
    support a number you put in a portfolio: the same suite run next month would score
    differently for reasons that have nothing to do with your system. Every verdict
    below records the exact model string it used.
    """

    def __init__(self, base_settings: Settings | None = None, *, model: str,
                 llm: LLMClient | None = None, votes: int = 3, seed: int = 0,
                 cache: JudgeCache | None = None) -> None:
        self.model = model
        self.votes = votes
        self.seed = seed
        if llm is not None:
            self.llm = llm
        else:
            if base_settings is None:
                raise ValueError("base_settings is required unless llm is provided")
            from app.core.providers import build_llm
            from app.core.trace import Trace
            judge_settings = replace(base_settings, llm_model=model, temperature=0.0)
            self.llm = build_llm(judge_settings, Trace(enabled=False))
        self.cache = cache or JudgeCache()

    def _vote(self, system: str, user: str, vote_idx: int) -> dict[str, Any]:
        from app.core.providers import Message
        prompt = f"{system}\n---\n{user}"
        key = self.cache.key(prompt, self.model, self.seed, vote_idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        resp = self.llm.chat([Message.system(system), Message.user(user)], temperature=0.0)
        parsed = _parse_json_object(resp.text) or {"_unparseable": True, "raw": resp.text[:500]}
        self.cache.set(key, parsed)
        return parsed

    def verdict(self, system: str, user: str) -> JudgeVerdict:
        votes = [self._vote(system, user, i) for i in range(self.votes)]
        sigs = [json.dumps(v, sort_keys=True) for v in votes]
        majority_sig, majority_count = Counter(sigs).most_common(1)[0]
        majority = votes[sigs.index(majority_sig)]
        disagreement = 1 - majority_count / len(votes)
        return JudgeVerdict(value=majority, raw_votes=votes, disagreement_rate=disagreement, model=self.model)

    def score_correctness(self, *, question: str, gold_answer: str, candidate: str) -> JudgeVerdict:
        user = f"QUESTION:\n{question}\n\nGOLD ANSWER:\n{gold_answer}\n\nCANDIDATE ANSWER:\n{candidate}"
        return self.verdict(CORRECTNESS_SYSTEM, user)

    def faithfulness(self, *, answer_text: str, retrieved_context: str) -> JudgeVerdict:
        user = f"SOURCES:\n{retrieved_context}\n\nANSWER:\n{answer_text}"
        return self.verdict(FAITHFULNESS_SYSTEM, user)


def _bucket(score: Any) -> str:
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "unparseable"
    return str(max(0, min(4, s)))


def calibrate(judge: Judge, subset_path: str | Path) -> dict[str, Any]:
    """Run `python -m core.eval calibrate` against a human-labelled subset.

    The subset file is JSONL: {case_id, question, gold_answer, candidate_answer,
    human_verdict}. `human_verdict` is null until a person fills it in -- this function
    reports kappa as null and says so explicitly rather than computing a number against
    absent data, which is exactly the "I measured whether my LLM judge is trustworthy"
    claim the README makes.
    """
    from .stats import cohens_kappa, confusion_matrix
    rows = [json.loads(line) for line in Path(subset_path).read_text().splitlines() if line.strip()]
    labelled = [r for r in rows if r.get("human_verdict") is not None]
    if not labelled:
        return {"kappa": None, "n_labelled": 0, "n_total": len(rows),
                "note": "no human verdicts yet -- see CONTRIBUTING.md for the calibration workflow"}

    pairs = []
    for r in labelled:
        v = judge.score_correctness(question=r["question"], gold_answer=r["gold_answer"],
                                    candidate=r["candidate_answer"])
        pairs.append((_bucket(v.value.get("score")), str(r["human_verdict"])))

    kappa = cohens_kappa(pairs)
    matrix = confusion_matrix(pairs)
    return {"kappa": round(kappa, 4), "n_labelled": len(labelled), "n_total": len(rows),
            "confusion_matrix": matrix, "judge_model": judge.model}
