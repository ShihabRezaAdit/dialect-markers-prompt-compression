from idea86 import survival as sv


def test_tokens_regex_and_offsets():
    t = sv.tokens("Y'all ain't gonna win, 1,200 times!")
    assert [x.text for x in t] == ["Y'all", "ain't", "gonna", "win", "1", "200", "times"]
    assert t[0].norm == "y'all" and (t[0].start, t[0].end) == (0, 5)


def test_diff_ops_replace_insert_delete():
    sae = sv.tokens("You are always studying at the library")
    dia = sv.tokens("Y'all always be studying at library")
    tags = [(o.tag, o.i1, o.i2, o.j1, o.j2) for o in sv.diff_ops(sae, dia)]
    assert ("replace", 0, 2, 0, 1) in tags          # you are -> y'all
    assert ("insert", 3, 3, 2, 3) in tags           # habitual be
    assert ("delete", 5, 6, 5, 5) in tags           # the
    assert sv.marker_positions(sv.diff_ops(sae, dia)) == {0, 2}


def test_type_op_inclusive_end_and_insert_point():
    sae_text = "She is always studying"
    sae = sv.tokens(sae_text)
    dia = sv.tokens("She always be studying")
    ops = sv.diff_ops(sae, dia)
    # rule covers "is" = chars 4..5 (end inclusive)
    rules = [{"start": 4, "end_incl": 5, "value": "", "type": "zero_copula"},
             {"start": 14, "end_incl": 21, "value": "be studying", "type": "habitual_be"}]
    by_tag = {o.tag: o for o in ops}
    assert sv.type_op(by_tag["delete"], sae, len(sae_text), rules) == "zero_copula"
    assert sv.type_op(by_tag["insert"], sae, len(sae_text), rules) == "habitual_be"
    assert sv.type_op(by_tag["insert"], sae, len(sae_text), []) == "unlogged"
    cov, tot = sv.rules_covered(rules, ops, sae, len(sae_text))
    assert (cov, tot) == (2, 2)


def test_type_op_smallest_start_wins():
    sae_text = "they were not there"
    sae = sv.tokens(sae_text)
    dia = sv.tokens("they wasn't there")
    op = [o for o in sv.diff_ops(sae, dia) if o.tag == "replace"][0]
    rules = [{"start": 9, "end_incl": 11, "type": "b"}, {"start": 5, "end_incl": 8, "type": "a"}]
    assert sv.type_op(op, sae, len(sae_text), rules) == "a"


def test_word_survival_kept_partial_deleted():
    orig = sv.tokens("the remarkable cathedral was finished in 1345")
    comp = sv.tokens("remark cathedral finished 1345")
    st = sv.word_survival(orig, comp)
    assert st == ["deleted", "partial", "kept", "deleted", "kept", "deleted", "kept"]
    assert sv.survived(st).sum() == 3
    assert sv.survived(st, partial_as_kept=True).sum() == 4


def test_word_survival_repeated_words():
    orig = sv.tokens("the cat saw the dog")
    comp = sv.tokens("cat saw dog")
    assert sv.word_survival(orig, comp) == ["deleted", "kept", "kept", "deleted", "kept"]


def test_paired_span_delta():
    sae = sv.tokens("you are going home now")
    dia = sv.tokens("y'all going home now")
    ops = sv.diff_ops(sae, dia)
    s_status = sv.word_survival(sae, sv.tokens("going home"))          # you, are deleted
    d_status = sv.word_survival(dia, sv.tokens("y'all going home"))    # y'all kept
    assert sv.paired_span_delta(ops, s_status, d_status) == 1.0
    assert sv.paired_span_delta(sv.diff_ops(sae, sae), s_status, s_status) is None


def test_ratio_from_counts_and_drop():
    import numpy as np
    c = np.zeros((4, sv.N_STRATA))
    c[:, 0] = [1, 2, 4, 8]      # markers 0.5, non-markers 0.5
    c[:, 1] = [3, 3, 1, 2]      # markers 1.0, non-markers 0.5
    c[:, 2] = [1, 1, 0, 0]      # no non-markers: dropped
    r = sv.ratio_from_counts(c)
    assert abs(r - (2 * 0.5 + 3 * 1.0) / (2 * 0.5 + 3 * 0.5)) < 1e-12
    assert sv.dropped_strata(c) == 1


def test_freq_strata_unseen_is_eleventh():
    fs = sv.FreqStrata(["a a a b b c", "a d"])
    assert fs.decile("zzz") == 10
    assert 0 <= fs.decile("a") <= 9
    assert sv.stratum(0, 9, "a", fs) < 11 and sv.stratum(8, 9, "a", fs) >= 22


def test_achieved_keep_rate():
    assert sv.achieved_keep_rate("one two three four", "one four") == 0.5
