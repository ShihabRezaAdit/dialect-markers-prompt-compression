"""LLMLingua window (decisions log): iterative_size = min(200, Llama-2 tokens of the text being
compressed), passed to L1 and LL and recorded; keep rates are reported by text length."""
from idea86 import compress as K
from idea86.pilot import keep_by_length


class FakeLlama:
    def __init__(self):
        self.calls = []

    def get_token_length(self, text, add_special_tokens=True):
        return len(text.split())          # one "token" per word for the test

    def compress_prompt(self, context, **kw):
        self.calls.append(kw)
        out = " ".join(context[0].split()[::2])
        if kw.get("question") is not None:
            out += "\n\n" + kw["question"]
        return {"compressed_prompt": out, "origin_tokens": 1, "compressed_tokens": 1}


def fam():
    f = K.LLMLinguaFamily.__new__(K.LLMLinguaFamily)
    f._llama, f._xlmr = FakeLlama(), None
    return f


def test_window_is_min_200_and_text_length():
    f = fam()
    short, long_ = " ".join(["w"] * 37), " ".join(["w"] * 450)
    r1 = f.l1(short, 0.5)
    assert f._llama.calls[-1]["iterative_size"] == 37 and r1["iterative_size"] == 37
    r2 = f.l1(long_, 0.5)
    assert f._llama.calls[-1]["iterative_size"] == 200 and r2["iterative_size"] == 200
    r3 = f.ll(short, "Who?", 0.5)                 # window from the passage, not the question
    assert f._llama.calls[-1]["iterative_size"] == 37 and r3["iterative_size"] == 37
    assert r3["compressed_prompt"].endswith("w") and "Who?" not in r3["compressed_prompt"]
    for k in ("rate", "use_context_level_filter", "condition_in_question"):
        assert k in f._llama.calls[0]


def test_record_keeps_window_and_length_and_report_groups():
    job = {"job_id": "A:0:S:L1", "compressor": "L1", "text": "a b c d", "rho": 0.5}
    rec = K.record(job, {"compressed_prompt": "a c", "iterative_size": 4}, 0.1)
    assert rec["iterative_size"] == 4 and rec["n_words"] == 4 and rec["achieved_keep_rate"] == 0.5
    kb = keep_by_length({"x": rec, "y": {**rec, "n_words": 160, "iterative_size": 200, "achieved_keep_rate": 0.4}})
    assert kb["L1 | words 1-49"]["mean"] == 0.5 and kb["L1 | words 150+"]["mean"] == 0.4
    assert kb["L1 | window <200 (= text length)"]["n"] == 1 and kb["L1 | window 200"]["n"] == 1
