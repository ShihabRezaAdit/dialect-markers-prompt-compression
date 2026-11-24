"""Paired cluster bootstrap, bootstrap p-values, Holm and equivalence tests
(internal plan Section 23).

A resample draws clusters (SQuAD paragraphs, CoQA stories, tweet pairs) with
replacement; everything belonging to a cluster (renderings, conditions, models)
moves together because statistics are computed from per-cluster arrays indexed by
the same resampled cluster indices. Every bootstrap uses default_rng(86), so all
cells of a family see the same resamples.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

B_DEFAULT = 2000
SEED = 86


def resample_indices(n_clusters: int, B: int = B_DEFAULT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(B, n_clusters))


def bootstrap(stat: Callable[[np.ndarray], float], n_clusters: int, B: int = B_DEFAULT,
              seed: int = SEED) -> tuple[float, np.ndarray]:
    """stat(idx) computes the statistic on clusters idx (an int array, repeats allowed)."""
    est = stat(np.arange(n_clusters))
    idx = resample_indices(n_clusters, B, seed)
    draws = np.array([stat(row) for row in idx], dtype=float)
    return float(est), draws


def mean_stat(values: np.ndarray) -> Callable[[np.ndarray], float]:
    """Mean over clusters of a per-cluster value (e.g. Delta_s(p))."""
    v = np.asarray(values, dtype=float)
    return lambda idx: float(np.nanmean(v[idx]))


def item_mean_stat(sums: np.ndarray, counts: np.ndarray) -> Callable[[np.ndarray], float]:
    """Mean over items when clusters hold several items: sum(item values)/sum(items)."""
    s, c = np.asarray(sums, float), np.asarray(counts, float)
    return lambda idx: float(s[idx].sum() / c[idx].sum())


def percentile_ci(draws: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    d = draws[~np.isnan(draws)]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def bootstrap_p(draws: np.ndarray) -> float:
    """Two-sided p = 2 min(P*[theta* <= 0], P*[theta* >= 0]), capped at 1."""
    d = draws[~np.isnan(draws)]
    if d.size == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down: returns adjusted p and reject flag per test."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = (not stop) and p <= alpha / (m - i)
        if not reject:
            stop = True
        out[k] = {"p": p, "p_holm": adj, "reject": reject}
    return out


def tost_equivalent(draws: np.ndarray, margin: float) -> tuple[bool, tuple[float, float]]:
    """Two one-sided tests at 0.05 via the 90% percentile interval inside +/- margin."""
    lo, hi = percentile_ci(draws, 0.90)
    return (lo > -margin and hi < margin), (lo, hi)


def signal(est: float, draws: np.ndarray, floor: float, level: float) -> dict:
    """Pilot signal for one cell: |est| >= floor and the level interval excludes 0."""
    lo, hi = percentile_ci(draws, level)
    excludes = lo > 0 or hi < 0
    return {"estimate": est, "ci_level": level, "ci": [lo, hi], "floor": floor,
            "met": bool(abs(est) >= floor and excludes)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)
