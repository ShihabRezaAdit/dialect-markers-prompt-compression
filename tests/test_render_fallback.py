"""Sentence fallback for passages that Multi-VALUE cannot render whole (decisions log):
the joined passage keeps the SAE sentence order and the text between sentences exactly, rule
spans move to passage offsets, and every passage is tagged with its rendering method."""
import re

from idea86.render import join_sentences, render_passage
from idea86.survival import diff_ops, tokens


def spans_strip(text):
    out = []
    for m in re.finditer(r"[^.!?]+[.!?]", text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        out.append((s, e))
    return out


class FakeDialect:
    """Fails on any text with more than one sentence (like the Spacy/Stanza mismatch);
    otherwise rewrites 'is' -> 'be' and logs the rule at SAE character offsets."""

    def __init__(self, fail_always=False):
        self.fail_always = fail_always
        self.seeds = []
        self.executed_rules, self.modification_counter = {}, {}

    def set_seed(self, seed):
        self.seeds.append(seed)

    def transform(self, text):
        if self.fail_always or len(re.findall(r"[.!?]", text)) > 1:
            raise AssertionError("Spacy and Stanza word mismatch")
        self.executed_rules = {(m.start(), m.end() - 1): {"value": "be", "type": "zero_copula"}
                               for m in re.finditer(r"\bis\b", text)}
        self.modification_counter = {"zero_copula": len(self.executed_rules), "total": len(self.executed_rules)}
        return re.sub(r"\bis\b", "be", text)


TEXT = "The river is long.  It is wide in 1990, e.g. here.\nThe city grew."


def test_join_keeps_order_and_spacing():
    sp = spans_strip(TEXT)
    assert join_sentences(TEXT, sp, [TEXT[s:e] for s, e in sp]) == TEXT
    reps = [f"<{i}>" for i in range(len(sp))]
    out = join_sentences(TEXT, sp, reps)
    seps = [TEXT[: sp[0][0]]] + [TEXT[sp[i][1]: sp[i + 1][0]] for i in range(len(sp) - 1)] + [TEXT[sp[-1][1]:]]
    assert out == "".join(seps[i] + reps[i] for i in range(len(sp))) + seps[-1]
    assert [int(x) for x in re.findall(r"<(\d+)>", out)] == list(range(len(sp)))   # order kept
    assert "  " in out and "\n" in out                                            # spacing kept
    try:
        join_sentences(TEXT, sp, reps[:-1])
        raise AssertionError("length mismatch not caught")
    except ValueError:
        pass


def test_fallback_render_is_exact_twin():
    d = FakeDialect()
    r = render_passage(d, TEXT, 7, spans_fn=spans_strip)
    assert r["ok"] and r["method"] == "sentence_fallback" and "mismatch" in r["whole_error"]
    assert r["text"] == re.sub(r"\bis\b", "be", TEXT)          # order, spacing and line break kept
    assert d.seeds[-1] == 86 + 7 and r["n_sentences"] == len(spans_strip(TEXT))
    for rule in r["rules"]:                                     # spans point at 'is' in the SAE passage
        assert TEXT[rule["start"]: rule["end_incl"] + 1] == "is"
    assert len(r["rules"]) == 2 and r["counter"]["total"] == 2
    ops = diff_ops(tokens(TEXT), tokens(r["text"]))             # only the two rewrites differ
    assert [o.tag for o in ops].count("replace") == 2 and all(o.tag in ("equal", "replace") for o in ops)


def test_whole_and_failed_methods():
    one = "The river is long."
    r = render_passage(FakeDialect(), one, 1, spans_fn=spans_strip)
    assert r["ok"] and r["method"] == "whole" and r["text"] == "The river be long."
    f = render_passage(FakeDialect(fail_always=True), TEXT, 1, spans_fn=spans_strip)
    assert not f["ok"] and f["method"] == "failed" and f["attempts"] == 2
