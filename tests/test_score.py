from idea86 import score as sc


def test_normalize_and_em_f1():
    assert sc.normalize_answer("The  Denver Broncos!") == "denver broncos"
    assert sc.exact_match("denver broncos", ["The Denver Broncos"]) == 1.0
    assert abs(sc.f1_score("Denver", ["Denver Broncos"]) - 2 / 3) < 1e-12
    assert sc.f1_score("Carolina", ["Denver Broncos"]) == 0.0


def test_answer_survives_and_extract():
    assert sc.answer_survives(["Santa Clara"], "played at Levi's Stadium in Santa Clara, California.")
    assert not sc.answer_survives(["Santa Clara"], "played at Levi's Stadium in California.")
    assert sc.extract_answer("  Santa Clara\nbecause ...") == "Santa Clara"
    assert sc.extract_answer("") == ""
    assert sc.is_flip("Denver", "the Denver") is False


def test_redial_extraction():
    assert sc.extract_redial("so <answer>12</answer> wait <answer>1,250</answer>", "math") == ("1,250", "tag")
    assert sc.extract_redial("the total is 40 then 42", "math") == ("42", "fallback")
    assert sc.extract_redial("no digits here", "math") == (None, "none")
    labels = ["True", "False", "Unknown"]
    assert sc.extract_redial("It is False, not True... final: Unknown", "logic", labels) == ("Unknown", "fallback")
    assert sc.redial_correct("1,250", "1250", "math")
    assert not sc.redial_correct(None, "3", "math")
    assert sc.redial_correct("true", "True", "logic")
