"""SQuAD normalization, EM, F1, answer survival, flips, and ReDial answer extraction
(internal plan Sections 12 and 22). Normalization follows the official SQuAD v1.1
evaluation script."""
from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: list[str]) -> float:
    return max(_f1(pred, g) for g in golds)


def answer_survives(golds: list[str], passage: str) -> bool:
    """A gold answer survives if at least one normalized gold answer is a substring of
    the normalized passage (also the rendering answer-preservation test)."""
    p = normalize_answer(passage)
    return any(normalize_answer(g) and normalize_answer(g) in p for g in golds)


def extract_answer(output: str) -> str:
    """SQuAD prompts: the first line of the output with surrounding whitespace removed."""
    return output.strip().split("\n", 1)[0].strip() if output.strip() else ""


def is_flip(ans_compressed: str, ans_uncompressed: str) -> bool:
    return normalize_answer(ans_compressed) != normalize_answer(ans_uncompressed)


# ---------------------------------------------------------------- ReDial (E6)

_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_redial(output: str, task: str, labels: list[str] | None = None) -> tuple[str | None, str]:
    """Return (answer, how): how is 'tag', 'fallback' or 'none'.

    The answer is the text inside the last <answer> tags. Without tags, math falls back
    to the last number and logic to the last occurrence of one of its labels.
    """
    tags = _TAG.findall(output)
    if tags:
        return tags[-1].strip(), "tag"
    if task == "math":
        nums = _NUM.findall(output)
        return (nums[-1], "fallback") if nums else (None, "none")
    if task == "logic" and labels:
        low = output.lower()
        best, best_pos = None, -1
        for lab in labels:
            pos = low.rfind(lab.lower())
            if pos > best_pos:
                best, best_pos = lab, pos
        return (best, "fallback") if best is not None else (None, "none")
    return None, "none"


def redial_correct(answer: str | None, gold: str, task: str) -> bool:
    if answer is None:
        return False
    if task == "math":
        def num(x):
            m = _NUM.findall(x)
            return float(m[-1].replace(",", "")) if m else None
        a, g = num(answer), num(gold)
        return a is not None and g is not None and abs(a - g) < 1e-9
    return answer.strip().lower() == gold.strip().lower()
