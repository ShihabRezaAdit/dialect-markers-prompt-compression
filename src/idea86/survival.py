"""Tokens, diff ops, rule typing, word survival, paired span survival and the
marker retention ratio (internal plan Section 11, steps 1 to 6).

Pure Python and NumPy: runs in env-stats and is fully unit-tested.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")

KEPT, PARTIAL, DELETED = "kept", "partial", "deleted"


@dataclass(frozen=True)
class Tok:
    text: str      # as written
    norm: str      # case-folded
    start: int     # character offsets in the source string, end exclusive
    end: int


def tokens(text: str) -> list[Tok]:
    return [Tok(m.group(0), m.group(0).casefold(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------- step 2: diff ops

@dataclass(frozen=True)
class Op:
    tag: str      # equal, replace, insert, delete
    i1: int       # SAE token range [i1, i2)
    i2: int
    j1: int       # dialect token range [j1, j2)
    j2: int


def diff_ops(sae: list[Tok], dia: list[Tok]) -> list[Op]:
    sm = SequenceMatcher(None, [t.norm for t in sae], [t.norm for t in dia], autojunk=False)
    return [Op(*o) for o in sm.get_opcodes()]


def marker_positions(ops: list[Op]) -> set[int]:
    """Dialect token indices inside replace or insert ops (marker words)."""
    out = set()
    for o in ops:
        if o.tag in ("replace", "insert"):
            out.update(range(o.j1, o.j2))
    return out


# ---------------------------------------------------------------- step 3: feature type

def op_sae_range(o: Op, sae: list[Tok], sae_len: int) -> tuple[int, int]:
    """SAE character range [cs, ce) of an op; an insert is a point at the next SAE token."""
    if o.i2 > o.i1:
        return sae[o.i1].start, sae[o.i2 - 1].end
    pos = sae[o.i1].start if o.i1 < len(sae) else sae_len
    return pos, pos


def type_op(o: Op, sae: list[Tok], sae_len: int, rules: list[dict]) -> str:
    """Primary feature type: the overlapping rule-log entry with the smallest start.

    Rule spans are [start, end_incl]. A replace/delete op overlaps a rule if their
    character ranges intersect. An insert op (a point p) is covered by a rule with
    start <= p <= end_incl + 1, so an insertion right after a rewritten span is typed
    by that rule. No overlapping entry -> "unlogged".
    """
    cs, ce = op_sae_range(o, sae, sae_len)
    hits = []
    for r in rules:
        rs, re_ = r["start"], r["end_incl"]
        if cs == ce:
            ok = rs <= cs <= re_ + 1
        else:
            ok = rs < ce and re_ >= cs
        if ok:
            hits.append(r)
    if not hits:
        return "unlogged"
    return min(hits, key=lambda r: (r["start"], r["end_incl"]))["type"]


def rules_covered(rules: list[dict], ops: list[Op], sae: list[Tok], sae_len: int) -> tuple[int, int]:
    """For gate G3: how many rule-log entries overlap at least one non-equal diff op."""
    spans = [op_sae_range(o, sae, sae_len) for o in ops if o.tag != "equal"]
    covered = 0
    for r in rules:
        rs, re_ = r["start"], r["end_incl"]
        if any((rs <= cs <= re_ + 1) if cs == ce else (rs < ce and re_ >= cs) for cs, ce in spans):
            covered += 1
    return covered, len(rules)


# ---------------------------------------------------------------- step 4: word survival

def word_survival(orig: list[Tok], comp: list[Tok]) -> list[str]:
    """Status of every original word in the compressed text: kept, partial or deleted.

    Kept: inside an equal block of the alignment. Partial: not kept, inside a replace
    block, and a compressed token in that block shares at least two characters with it
    as a substring either way (a split or shortened token). Otherwise deleted.
    """
    status = [DELETED] * len(orig)
    sm = SequenceMatcher(None, [t.norm for t in orig], [t.norm for t in comp], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                status[i] = KEPT
        elif tag == "replace":
            cands = [comp[j].norm for j in range(j1, j2)]
            for i in range(i1, i2):
                w = orig[i].norm
                if any(len(c) >= 2 and (c in w or w in c) for c in cands):
                    status[i] = PARTIAL
    return status


def survived(status: list[str], partial_as_kept: bool = False) -> np.ndarray:
    keep = {KEPT, PARTIAL} if partial_as_kept else {KEPT}
    return np.array([s in keep for s in status], dtype=float)


# ---------------------------------------------------------------- step 5: paired span survival

def paired_span_delta(ops: list[Op], sae_status: list[str], dia_status: list[str],
                      partial_as_kept: bool = False) -> float | None:
    """Delta_s(p): mean over replace ops of s_D(o) - s_S(o); None if no replace op."""
    s_sae = survived(sae_status, partial_as_kept)
    s_dia = survived(dia_status, partial_as_kept)
    diffs = [s_dia[o.j1:o.j2].mean() - s_sae[o.i1:o.i2].mean() for o in ops if o.tag == "replace"]
    return float(np.mean(diffs)) if diffs else None


# ---------------------------------------------------------------- step 6: marker retention ratio

N_STRATA = 3 * 11  # 3 position terciles x (10 frequency deciles + 1 unseen)


class FreqStrata:
    """Corpus-frequency deciles of log(1 + count) over all SAE word occurrences of a
    corpus; a word not seen in the SAE corpus goes to an eleventh stratum."""

    def __init__(self, sae_texts: list[str]):
        self.counts = Counter(t.norm for s in sae_texts for t in tokens(s))
        occ = np.array([math.log1p(self.counts[t.norm]) for s in sae_texts for t in tokens(s)])
        self.cuts = np.quantile(occ, np.linspace(0.1, 0.9, 9)) if len(occ) else np.zeros(9)

    def decile(self, word_norm: str) -> int:
        c = self.counts.get(word_norm, 0)
        if c == 0:
            return 10
        return int(np.searchsorted(self.cuts, math.log1p(c), side="right"))


def stratum(idx: int, n: int, word_norm: str, fs: FreqStrata) -> int:
    tercile = min(2, (3 * idx) // max(n, 1))
    return tercile * 11 + fs.decile(word_norm)


def ratio_counts(dia: list[Tok], ops: list[Op], dia_status: list[str], fs: FreqStrata,
                 partial_as_kept: bool = False) -> np.ndarray:
    """Per-text counts, shape (4, N_STRATA): marker kept, marker total, non-marker kept,
    non-marker total. Non-marker words are dialect words in equal ops. Summed over the
    texts of a cluster, these are what the bootstrap resamples."""
    keep = survived(dia_status, partial_as_kept)
    markers = marker_positions(ops)
    equal = set()
    for o in ops:
        if o.tag == "equal":
            equal.update(range(o.j1, o.j2))
    c = np.zeros((4, N_STRATA))
    n = len(dia)
    for j, t in enumerate(dia):
        k = stratum(j, n, t.norm, fs)
        if j in markers:
            c[0, k] += keep[j]
            c[1, k] += 1
        elif j in equal:
            c[2, k] += keep[j]
            c[3, k] += 1
    return c


def ratio_from_counts(c: np.ndarray) -> float:
    """r = sum_k w_k s^M_k / sum_k w_k s^N_k with w_k the marker count in stratum k;
    strata with no marker or no non-marker word are dropped."""
    ok = (c[1] > 0) & (c[3] > 0)
    if not ok.any():
        return float("nan")
    w = c[1][ok]
    sm = c[0][ok] / c[1][ok]
    sn = c[2][ok] / c[3][ok]
    den = (w * sn).sum()
    return float((w * sm).sum() / den) if den > 0 else float("nan")


def dropped_strata(c: np.ndarray) -> int:
    return int(((c[1] > 0) & (c[3] == 0)).sum())


# ---------------------------------------------------------------- achieved keep rate

def achieved_keep_rate(original: str, compressed: str) -> float:
    n = len(tokens(original))
    return len(tokens(compressed)) / n if n else float("nan")
