"""AAE grammatical feature lexicon used to type REAL-AAE diff ops (internal plan Section 11, step 8).

Source: Wu et al. (2026), "LLMs Silently Correct AAE" (cited in the plans as wu2026silently), Appendix
C.4, Table 8 "AAE grammatical feature lexicon adapted from Harris et al. (2022)", printed page 14
(page 14 of the PDF in Literature Review/Papers/2026_Wu_LLMs_Silently_Correct_AAE.pdf). Transcribed
entry by entry, in the table's order and spelling (straight and curly apostrophes as printed), and
checked against a 300-dpi image of the page. The table contains slurs because it was built from
tweets; they are kept verbatim so that the lexicon matches the source.

Typing rule (decisions log): a replace op of a REAL-AAE pair is typed by a lexicon entry whose word
tokens occur in the AAE tweet at positions overlapping the op's AAE tokens. Entries are compared with
straight and curly apostrophes treated alike. If several entries match, the longest (in tokens) wins;
ties go to the class listed first (Auxiliary, Aspectual, Preverbal, Syntactic). An op with no match is
typed "other".
"""
from __future__ import annotations

from .survival import Op, Tok, tokens

SOURCE = {"paper": "Wu et al. 2026, LLMs Silently Correct AAE", "table": "Table 8 (Appendix C.4)", "page": 14,
          "adapted_from": "Harris et al. 2022"}

CLASSES = ("Auxiliary", "Aspectual", "Preverbal", "Syntactic")

LEXICON = {
    "Auxiliary": ["we was", "they was", "finna", "tryna", "imma", "i’mma", "bitches was", "niggas was", "yall was",
                  "y’all was", "you was", "wanna", "gonna", "ima", "ion", "u was", "iont"],
    "Aspectual": ["I be", "he be", "they be", "she be", "I been", "he been", "she been", "they been", "it be",
                  "niggas be", "bitches be", "yall be", "y’all be", "you be", "u be"],
    "Preverbal": ["aint", "ain’t", "steady", "stay", "he done", "she done", "they done", "yall done", "y’all done",
                  "you done", "u done"],
    "Syntactic": ["cant nobody", "can’t nobody", "he don’t", "she don’t", "don’t never", "he dont", "she dont",
                  "dont never", "yall don’t", "y’all don’t", "aint nothing", "ain’t nothing", "aint nobody",
                  "ain’t nobody", "yall dont"],
}


def _forms(entry: str) -> list[tuple[str, ...]]:
    """Token sequences (case-folded) of an entry, with curly and straight apostrophes treated alike."""
    out = []
    for v in {entry, entry.replace("’", "'"), entry.replace("'", "’")}:
        seq = tuple(t.norm for t in tokens(v))
        if seq and seq not in out:
            out.append(seq)
    return out


ENTRIES = [(c, e, f) for c in CLASSES for e in LEXICON[c] for f in _forms(e)]


def matches(dia: list[Tok]) -> list[tuple[int, int, str, str]]:
    """Every occurrence of a lexicon entry in a token list: (start, end, class, entry)."""
    norms = [t.norm for t in dia]
    out = []
    for c, e, f in ENTRIES:
        n = len(f)
        for i in range(len(norms) - n + 1):
            if tuple(norms[i:i + n]) == f:
                out.append((i, i + n, c, e))
    return out


def type_op(o: Op, found: list[tuple[int, int, str, str]]) -> str:
    """Class of a replace or insert op from the matches of its AAE text, or "other"."""
    hits = [(s, e, c, x) for s, e, c, x in found if s < o.j2 and o.j1 < e]
    if not hits:
        return "other"
    hits.sort(key=lambda h: (-(h[1] - h[0]), CLASSES.index(h[2])))
    return hits[0][2]
