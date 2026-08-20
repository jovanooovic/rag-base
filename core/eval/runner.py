from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .types import Case, Prediction

SystemUnderTest = Callable[[Case], Prediction]


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def content_hash(dataset_bytes: bytes, config: dict[str, Any], sha: str) -> str:
    """Names a results file from dataset + config + git SHA so two runs are never
    silently conflated."""
    payload = dataset_bytes + json.dumps(config, sort_keys=True).encode() + sha.encode()
    return hashlib.sha256(payload).hexdigest()[:12]


async def run_suite(cases: Sequence[Case], system: SystemUnderTest, *,
                    concurrency: int = 4) -> list[Prediction]:
    """Run every case through `system`, bounded to `concurrency` in flight.

    `system` is a plain sync callable -- the RAG pipeline (and any agent runtime that
    reuses this package) is synchronous today, so the concurrency here comes from a
    thread pool via asyncio.to_thread, not from the system itself being async.
    """
    sem = asyncio.Semaphore(concurrency)

    async def run_one(case: Case) -> Prediction:
        async with sem:
            started = time.time()
            try:
                return await asyncio.to_thread(system, case)
            except Exception as exc:  # noqa: BLE001 - one case failing must not kill the run
                return Prediction(case_id=case.id, output={}, error=f"{type(exc).__name__}: {exc}",
                                  latency_ms=round((time.time() - started) * 1000, 1))

    return list(await asyncio.gather(*(run_one(c) for c in cases)))


def write_results(out_dir: str | Path, run_hash: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashed = out / f"{run_hash}.json"
    latest = out / "latest.json"
    text = json.dumps(payload, indent=2)
    hashed.write_text(text)
    latest.write_text(text)
    return hashed, latest
