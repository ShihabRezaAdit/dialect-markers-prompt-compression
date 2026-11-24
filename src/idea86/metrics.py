"""The only definitions of Delta_s, r, Delta_a, Delta_gamma and Delta_phi
(internal plan Section 22). Every analysis imports these functions; each returns
per-cluster arrays so that stats.bootstrap can resample clusters.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import survival as sv
from .score import answer_survives, f1_score, is_flip


# ---------------------------------------------------------------- Delta_s (paired span survival)

def delta_s_text(sae: str, dia: str, comp_sae: str, comp_dia: str,
                 partial_as_kept: bool = False) -> float | None:
    """Delta_s for one SAE/dialect pair compressed by the same compressor at the same rate."""
    st, dt = sv.tokens(sae), sv.tokens(dia)
    ops = sv.diff_ops(st, dt)
    s_status = sv.word_survival(st, sv.tokens(comp_sae))
    d_status = sv.word_survival(dt, sv.tokens(comp_dia))
    return sv.paired_span_delta(ops, s_status, d_status, partial_as_kept)


# ---------------------------------------------------------------- r (marker retention ratio)

def ratio_counts_text(sae: str, dia: str, comp_dia: str, fs: sv.FreqStrata,
                      partial_as_kept: bool = False) -> np.ndarray:
    st, dt = sv.tokens(sae), sv.tokens(dia)
    ops = sv.diff_ops(st, dt)
    d_status = sv.word_survival(dt, sv.tokens(comp_dia))
    return sv.ratio_counts(dt, ops, d_status, fs, partial_as_kept)


def ratio_stat(counts: np.ndarray):
    """counts: array (n_clusters, 4, N_STRATA). Returns stat(idx) for the bootstrap."""
    return lambda idx: sv.ratio_from_counts(counts[idx].sum(axis=0))


def load_lexicon(path) -> list[tuple[str, ...]]:
    """REAL-AAE feature lexicon (Wu et al., Table 8), one entry per line as
    `entry<TAB>class`; entries may be multi-word. Lines starting with # are ignored."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = line.split("\t")[0]
            toks = tuple(t.norm for t in sv.tokens(entry))
            if toks:
                out.append(toks)
    return out


def lexicon_marker_positions(toks: list, lexicon: list[tuple[str, ...]]) -> set[int]:
    norms = [t.norm for t in toks]
    hits = set()
    for entry in lexicon:
        n = len(entry)
        for i in range(len(norms) - n + 1):
            if tuple(norms[i:i + n]) == entry:
                hits.update(range(i, i + n))
    return hits


def lexicon_ratio_counts(text: str, comp: str, lexicon: list[tuple[str, ...]], fs: sv.FreqStrata) -> np.ndarray:
    """UD-TwitterAAE fallback: markers are word tokens inside a lexicon match (no SAE side);
    every other word is a non-marker. Same strata and formula as r."""
    t = sv.tokens(text)
    keep = sv.survived(sv.word_survival(t, sv.tokens(comp)))
    marks = lexicon_marker_positions(t, lexicon)
    c = np.zeros((4, sv.N_STRATA))
    for j, tok in enumerate(t):
        k = sv.stratum(j, len(t), tok.norm, fs)
        row = 0 if j in marks else 2
        c[row, k] += keep[j]
        c[row + 1, k] += 1
    return c


# ---------------------------------------------------------------- Delta_a (answer survival)

def answer_survival_arrays(questions: list[dict], comp: dict, dialect: str,
                           cluster_key: str = "ppos") -> tuple[np.ndarray, np.ndarray, list]:
    """Per-cluster sums of [surv(SAE question) - surv(dialect question)] and item counts.

    comp[(qpos, g)] is the compressed SAE passage conditioned on the question in
    rendering g. Questions with a missing compression in either rendering are skipped.
    """
    sums, counts = defaultdict(float), defaultdict(int)
    for q in questions:
        a, b = comp.get((q["pos"], "S")), comp.get((q["pos"], dialect))
        if a is None or b is None:
            continue
        d = float(answer_survives(q["answers"], a)) - float(answer_survives(q["answers"], b))
        sums[q[cluster_key]] += d
        counts[q[cluster_key]] += 1
    keys = sorted(sums)
    return np.array([sums[k] for k in keys]), np.array([counts[k] for k in keys]), keys


def discordance(questions: list[dict], comp: dict, dialect: str) -> float:
    n = disc = 0
    for q in questions:
        a, b = comp.get((q["pos"], "S")), comp.get((q["pos"], dialect))
        if a is None or b is None:
            continue
        n += 1
        disc += answer_survives(q["answers"], a) != answer_survives(q["answers"], b)
    return disc / n if n else float("nan")


# ---------------------------------------------------------------- Delta_gamma and Delta_phi

def gap_items(questions: list[dict], ans: dict, dialect: str, cond: str) -> dict[int, float]:
    """Item difference d_i = [F1_S(C) - F1_D(C)] - [F1_S(U) - F1_D(U)], against SAE golds.
    ans[(qpos, rendering, condition)] is the normalized model answer."""
    out = {}
    for q in questions:
        keys = [(q["pos"], "S", cond), (q["pos"], dialect, cond), (q["pos"], "S", "U"), (q["pos"], dialect, "U")]
        if any(k not in ans for k in keys):
            continue
        g = q["answers"]
        out[q["pos"]] = ((f1_score(ans[keys[0]], g) - f1_score(ans[keys[1]], g))
                         - (f1_score(ans[keys[2]], g) - f1_score(ans[keys[3]], g)))
    return out


def flip_items(questions: list[dict], ans: dict, dialect: str, cond: str) -> dict[int, float]:
    """Per question: flip(D) - flip(S) under condition C relative to U."""
    out = {}
    for q in questions:
        keys = [(q["pos"], "S", cond), (q["pos"], "S", "U"), (q["pos"], dialect, cond), (q["pos"], dialect, "U")]
        if any(k not in ans for k in keys):
            continue
        out[q["pos"]] = float(is_flip(ans[keys[2]], ans[keys[3]])) - float(is_flip(ans[keys[0]], ans[keys[1]]))
    return out


def cluster_sums(item_values: dict[int, float], qpos_to_cluster: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    sums, counts = defaultdict(float), defaultdict(int)
    for qpos, v in item_values.items():
        c = qpos_to_cluster[qpos]
        sums[c] += v
        counts[c] += 1
    keys = sorted(sums)
    return np.array([sums[k] for k in keys]), np.array([counts[k] for k in keys])
