"""REAL-AAE lexicon (Wu et al. 2026, Table 8, page 14): transcription counts, matching and typing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea86 import lexicon as L  # noqa: E402
from idea86 import survival as sv  # noqa: E402


def test_table_counts_and_order():
    assert L.CLASSES == ("Auxiliary", "Aspectual", "Preverbal", "Syntactic")
    assert [len(L.LEXICON[c]) for c in L.CLASSES] == [17, 15, 11, 15]
    assert L.LEXICON["Auxiliary"][0] == "we was" and L.LEXICON["Auxiliary"][-1] == "iont"
    assert L.LEXICON["Aspectual"][-1] == "u be" and L.LEXICON["Preverbal"][-1] == "u done"
    assert L.LEXICON["Syntactic"][0] == "cant nobody" and L.LEXICON["Syntactic"][-1] == "yall dont"
    assert L.SOURCE["table"].startswith("Table 8") and L.SOURCE["page"] == 14
    for c in L.CLASSES:
        assert len(set(L.LEXICON[c])) == len(L.LEXICON[c])


def _type(sae, aae):
    s, d = sv.tokens(sae), sv.tokens(aae)
    found = L.matches(d)
    return [L.type_op(o, found) for o in sv.diff_ops(s, d) if o.tag == "replace"]


def test_typing_examples():
    assert _type("we were late", "we was late") == ["Auxiliary"]          # entry spans the op and its context
    assert _type("he is always working", "he be working") == ["Aspectual"]
    assert _type("I am going to eat", "I finna eat") == ["Auxiliary"]
    assert _type("there is nobody here", "ain't nobody here") == ["Syntactic"]   # longest entry wins over ain't
    assert _type("he isn't here", "he ain’t here") == ["Preverbal"]             # curly apostrophe as printed
    assert _type("the car is red", "the whip is red") == ["other"]


def test_case_and_apostrophe_forms():
    assert L.matches(sv.tokens("Y'ALL WAS there"))[0][2] == "Auxiliary"
    assert any(c == "Aspectual" for _, _, c, _ in L.matches(sv.tokens("i be chillin")))
