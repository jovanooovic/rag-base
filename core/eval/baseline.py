from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Scorecard


def load_baseline(path: str | Path) -> dict[str, float] | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    return {m["name"]: m["value"] for m in data.get("metrics", [])}


def write_baseline(path: str | Path, scorecard: Scorecard) -> None:
    from .report import to_json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_json(scorecard), indent=2))


def compare(scorecard: Scorecard, baseline: dict[str, float] | None, *,
           regression_threshold: float = 0.02) -> dict[str, Any]:
    """Per-metric deltas against baseline. `regressions` lists metrics that moved the
    wrong direction by more than `regression_threshold` -- this is what CI gates on.

    Deltas are still reported for every metric, including ungated ones (latency, cost)
    -- only gating is skipped, not visibility.
    """
    regressions: list[str] = []
    deltas: dict[str, float] = {}
    if baseline:
        for m in scorecard.metrics:
            prev = baseline.get(m.name)
            if prev is None:
                continue
            delta = m.value - prev
            deltas[m.name] = round(delta, 4)
            if not m.regression_gated:
                continue
            bad = (delta < -regression_threshold) if m.higher_is_better else (delta > regression_threshold)
            if bad:
                regressions.append(m.name)
    return {"deltas": deltas, "regressions": regressions}
