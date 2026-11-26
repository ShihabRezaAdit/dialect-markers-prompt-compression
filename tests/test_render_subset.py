"""Feature subset, quotation protection and length check (decisions log, after the failed G5
and G3 in P0): switched-off features are removed and their rules dropped, an unknown feature
stops the run, quoted text is copied unchanged with the rest rejoined exactly, and a rendering
longer than the maximum ratio is excluded as too_long."""
import re

from idea86.render import disable_features, length_check, quote_spans, render_job, render_protected


class FakeFeatureDialect:
    """morphosyntax_transforms like value-nlp 0.3; decide_filter_rules returns rules by name."""

    def __init__(self):
        self.morphosyntax_transforms = {"zero_copula": 1.0, "null_relcl": 0.6, "null_relcl_that": 0.3}

    def decide_filter_rules(self):
        return {"zero_copula": {(0, 1): {}}, "null_relcl": {(2, 3): {}}}


def test_disable_removes_feature_and_its_rules():
    d = FakeFeatureDialect()
    assert disable_features(d, ["null_relcl"]) == ["null_relcl"]
    assert set(d.morphosyntax_transforms) == {"zero_copula", "null_relcl_that"}
    # a rule written under the disabled name (e.g. by null_relcl_that) is not applied
    assert set(d.decide_filter_rules()) == {"zero_copula"}


def test_unknown_feature_stops():
    d = FakeFeatureDialect()
    try:
        disable_features(d, ["no_such_feature"])
        raise AssertionError("unknown feature not caught")
    except SystemExit as e:
        assert "no_such_feature" in str(e)


def test_empty_list_changes_nothing():
    d = FakeFeatureDialect()
    orig = d.decide_filter_rules
    assert disable_features(d, []) == []
    assert d.decide_filter_rules == orig and len(d.morphosyntax_transforms) == 3


def fake_render(dialect, text, k):
    """'is' -> 'be', rule at offsets in `text`; fails on 'BOOM'."""
    if "BOOM" in text:
        return {"ok": False, "error": "x", "attempts": 2, "method": "failed"}
    rules = [{"start": m.start(), "end_incl": m.end() - 1, "value": "be", "type": "zero_copula"}
             for m in re.finditer(r"\bis\b", text)]
    return {"ok": True, "text": re.sub(r"\bis\b", "be", text), "rules": rules,
            "counter": {"zero_copula": len(rules)}, "seconds": 0.0, "attempts": 1, "method": "whole"}


TEXT = 'He is here.  She said "this is mine," and “it is old”; it is.\nDone "is'


def test_quote_spans():
    sp = quote_spans(TEXT)
    assert [TEXT[s:e] for s, e in sp] == ['"this is mine,"', "“it is old”"]   # unpaired " ignored


def test_quotes_copied_rest_rejoined_exactly():
    r = render_protected(None, TEXT, 0, fake_render)
    assert r["ok"] and r["quote_protected"] and r["n_quotes"] == 2
    assert r["text"] == 'He be here.  She said "this is mine," and “it is old”; it be.\nDone "be'
    # every rule span points at an "is" of the SAE text outside the quotes
    for x in r["rules"]:
        assert TEXT[x["start"]: x["end_incl"] + 1] == "is"
        assert not any(s <= x["start"] < e for s, e in quote_spans(TEXT))
    assert len(r["rules"]) == 3


def test_identity_render_gives_identical_text():
    ident = lambda d, t, k: {"ok": True, "text": t, "rules": [], "counter": {}, "method": "whole"}
    assert render_protected(None, TEXT, 0, ident)["text"] == TEXT


def test_no_quotes_is_plain_render():
    t = "It is fine."
    assert render_protected(None, t, 0, fake_render) == fake_render(None, t, 0)


def test_failed_part_fails_text():
    r = render_protected(None, 'BOOM is "x" here', 0, fake_render)
    assert not r["ok"] and r["method"] == "failed"


def test_length_check():
    sae = "one two three four five six seven eight nine ten"
    ok = {"ok": True, "text": sae + " a b c", "rules": [], "method": "whole"}
    assert length_check(ok, sae, 1.3)["ok"]                              # 13 / 10 = 1.3, kept
    long = {"ok": True, "text": sae + " a b c d", "rules": [], "method": "whole"}
    r = length_check(long, sae, 1.3)
    assert not r["ok"] and r["method"] == "too_long" and r["rendered_method"] == "whole"
    assert r["too_long_text"] == long["text"] and abs(r["length_ratio"] - 1.4) < 1e-9
    assert length_check(long, sae, None) == long                         # no ratio set: unchanged


def test_render_job_question_and_passage():
    class D:
        def set_seed(self, s):
            pass

        def transform(self, t):
            self.executed_rules, self.modification_counter = {}, {}
            return t + " " + t                                           # doubles the text

    q = render_job(D(), {"kind": "question", "text": "Who is it?", "k": 1},
                   {"protect_quotes": True, "max_length_ratio": 1.3})
    assert not q["ok"] and q["method"] == "too_long"
    q = render_job(D(), {"kind": "question", "text": "Who is it?", "k": 1}, {})
    assert q["ok"] and q["method"] == "whole"
