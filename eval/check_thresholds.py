"""Enforce absolute quality floors on top of baseline regression gating.

`core.eval run --fail-on-regression` only catches a metric moving the wrong direction
relative to the committed baseline -- a slow multi-PR slide never trips it if each
individual PR stays within the regression threshold. Some metrics have a floor that
must never be crossed regardless of where the baseline has drifted to; faithfulness
and refusal_accuracy are the two this repo's eval policy names explicitly.

    python -m eval.check_thresholds eval/report/latest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = {"faithfulness": 0.85, "refusal_accuracy": 0.90}


def check(scorecard: dict, thresholds: dict[str, float] = THRESHOLDS) -> list[str]:
    by_name = {m["name"]: m["value"] for m in scorecard.get("metrics", [])}
    failures = []
    for name, floor in thresholds.items():
        value = by_name.get(name)
        if value is not None and value < floor:
            failures.append(f"{name}={value:.4f} is below the required floor of {floor}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scorecard_json")
    args = ap.parse_args(argv)

    scorecard = json.loads(Path(args.scorecard_json).read_text())
    failures = check(scorecard)

    if failures:
        print("FAIL: absolute quality floor violated")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: " + ", ".join(f"{k}>={v}" for k, v in THRESHOLDS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
