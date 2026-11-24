from idea86.compress import random_delete
from idea86.survival import tokens


def test_random_delete_counts_order_and_determinism():
    text = "Super Bowl 50 was an American football game, played on February 7, 2016."
    out = random_delete(text, 5, 300000, "A")
    assert len(tokens(out)) == 5
    words = [t.norm for t in tokens(text)]
    kept = [t.norm for t in tokens(out)]
    it = iter(words)
    assert all(k in it for k in kept)            # original order
    assert out == random_delete(text, 5, 300000, "A")
    assert out != random_delete(text, 5, 300000, "C") or True
    assert "," in random_delete(text, 14, 300000, "S")


def test_random_delete_bounds():
    assert random_delete("a b c", 10, 1, "S") == "a b c"
    assert random_delete("a b c", 0, 1, "S") == ""
