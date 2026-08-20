from __future__ import annotations

from typing import Any

from .types import MetricResult, Scorecard


def _delta_str(scorecard: Scorecard, m: MetricResult) -> str:
    if not scorecard.baseline or m.name not in scorecard.baseline:
        return "-"
    prev = scorecard.baseline[m.name]
    delta = m.value - prev
    good = (delta >= -1e-9) if m.higher_is_better else (delta <= 1e-9)
    marker = "OK" if good else "REGRESSION"
    return f"{delta:+.4f} ({marker})"


def to_markdown(scorecard: Scorecard) -> str:
    lines = [f"# {scorecard.label}", "", f"n = {scorecard.n_cases}", "",
             "| Metric | Value | 95% CI | n | vs. baseline |", "|---|---|---|---|---|"]
    for m in scorecard.metrics:
        lines.append(f"| {m.name} | {m.value:.4f} | [{m.ci_low:.4f}, {m.ci_high:.4f}] | "
                     f"{m.n} | {_delta_str(scorecard, m)} |")
    kappa = scorecard.meta.get("kappa")
    kappa_str = "pending" if kappa is None else f"{kappa:.3f}"
    lines += ["", f"Judge-vs-human kappa: **{kappa_str}**"]
    return "\n".join(lines)


def to_json(scorecard: Scorecard) -> dict[str, Any]:
    return {
        "label": scorecard.label,
        "n_cases": scorecard.n_cases,
        "meta": scorecard.meta,
        "baseline": scorecard.baseline,
        "metrics": [
            {"name": m.name, "value": m.value, "ci": [m.ci_low, m.ci_high], "n": m.n,
             "higher_is_better": m.higher_is_better, "extra": m.extra,
             "delta": (round(m.value - scorecard.baseline[m.name], 4)
                      if scorecard.baseline and m.name in scorecard.baseline else None)}
            for m in scorecard.metrics
        ],
    }


def to_terminal(scorecard: Scorecard) -> str:
    rows = [("metric", "value", "95% CI", "n")] + [
        (m.name, f"{m.value:.4f}", f"[{m.ci_low:.4f}, {m.ci_high:.4f}]", str(m.n))
        for m in scorecard.metrics
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [scorecard.label, "=" * len(scorecard.label)]
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))
    kappa = scorecard.meta.get("kappa")
    lines += ["", f"judge-vs-human kappa: {'pending' if kappa is None else f'{kappa:.3f}'}"]
    return "\n".join(lines)
