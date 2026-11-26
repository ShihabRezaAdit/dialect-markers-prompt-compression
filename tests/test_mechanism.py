"""E5a helpers: token-to-word mapping, Selective Context unit mapping, MeetingBank counts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea86 import mechanism as M  # noqa: E402


def test_word_scores_sum_and_mean():
    text = "She always be studying."
    spans = [(0, 3), (3, 10), (10, 13), (13, 18), (18, 22), (22, 23)]   # " studying" split in two, "." alone
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
    assert M.word_scores(text, spans, vals, "sum") == [1.0, 2.0, 3.0, 9.0]
    assert M.word_scores(text, spans, vals, "mean") == [1.0, 2.0, 3.0, 4.5]


def test_word_without_tokens_is_none():
    assert M.word_scores("a b", [(0, 1)], [0.5], "mean") == [0.5, None]


def test_sc_units_map_back_to_words():
    text = "She  always be studying. He ain't there."
    context = "She always be studying. He ain't there."
    units = ["She always", " be", " studying.", "He", " ain't there."]
    info = [1.0, 2.0, 3.0, 4.0, 5.0]
    ws = M.unit_word_scores(text, context, units, info)
    assert ws == [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0]


def test_sc_mismatch_raises():
    try:
        M.unit_word_scores("a b c", "a c", ["a", " c"], [1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("a mismatch between the cleaned text and the original was accepted")


def test_meetingbank_counts():
    r = M.meetingbank_counts([("The cat is here", "cat here"), ("the dog", "the dog")])
    assert r["documents"] == 2
    assert r["words"]["the"] == [2, 1] and r["words"]["cat"] == [1, 1] and r["words"]["is"] == [1, 0]
