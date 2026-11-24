from idea86 import metrics as M


def test_delta_s_text():
    v = M.delta_s_text("you are going home", "y'all going home", "going home", "y'all going home")
    assert v == 1.0


def test_gap_and_flip_items():
    qs = [{"pos": 0, "answers": ["Denver"]}]
    ans = {(0, "S", "L1"): "Denver", (0, "A", "L1"): "Carolina", (0, "S", "U"): "Denver", (0, "A", "U"): "Denver"}
    assert M.gap_items(qs, ans, "A", "L1") == {0: 1.0}
    assert M.flip_items(qs, ans, "A", "L1") == {0: 1.0}


def test_answer_survival_arrays():
    qs = [{"pos": 0, "ppos": 0, "answers": ["x"]}, {"pos": 1, "ppos": 0, "answers": ["y"]}]
    comp = {(0, "S"): "x here", (0, "A"): "gone", (1, "S"): "y", (1, "A"): "y"}
    s, c, k = M.answer_survival_arrays(qs, comp, "A")
    assert list(s) == [1.0] and list(c) == [2]


def test_lexicon_markers():
    from idea86.survival import tokens
    lex = [("finna",), ("ion", "know")]
    assert M.lexicon_marker_positions(tokens("i finna go ion know"), lex) == {1, 3, 4}


def test_partial_words_not_kept_in_primary_but_kept_in_sensitivity():
    # SAE "you are" -> dialect "y'all" (one replace op). The compressed SAE text keeps only
    # "yo", a shortened "you" (partial); the compressed dialect text keeps "y'all".
    sae, dia = "you are going home", "y'all going home"
    comp_sae, comp_dia = "yo going home", "y'all going home"
    primary = M.delta_s_text(sae, dia, comp_sae, comp_dia)
    assert primary == M.delta_s_text(sae, dia, comp_sae, comp_dia, partial_as_kept=False)
    assert primary == 1.0           # partial "you" counts as deleted: s_S = 0, s_D = 1
    sensitivity = M.delta_s_text(sae, dia, comp_sae, comp_dia, partial_as_kept=True)
    assert sensitivity == 0.5       # partial "you" counted as kept: s_S = 0.5
