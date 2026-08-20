from __future__ import annotations

import random
from typing import Any, Sequence


def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 10_000,
                 confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`.

    Seeded so a re-run of the same suite against the same predictions reports the
    identical interval -- an interval that jitters between runs is not one a client
    should be shown.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples))
    return (means[lo_idx], means[hi_idx])


def cohens_kappa(pairs: Sequence[tuple[Any, Any]]) -> float:
    """Cohen's kappa for two raters over categorical verdicts."""
    if not pairs:
        return 0.0
    n = len(pairs)
    labels = sorted({v for pair in pairs for v in pair})
    row_totals = dict.fromkeys(labels, 0)
    col_totals = dict.fromkeys(labels, 0)
    agree = 0
    for a, b in pairs:
        row_totals[a] += 1
        col_totals[b] += 1
        if a == b:
            agree += 1
    p_o = agree / n
    p_e = sum(row_totals[label] * col_totals[label] for label in labels) / (n * n)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(pairs: Sequence[tuple[Any, Any]]) -> dict[str, dict[str, int]]:
    """{judge_label: {human_label: count}}."""
    matrix: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        row = matrix.setdefault(str(a), {})
        row[str(b)] = row.get(str(b), 0) + 1
    return matrix
