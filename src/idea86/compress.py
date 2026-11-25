"""Compressors L1, L2, LL, SC and the random control R, with budget records
(internal plan Section 10). Calls are exactly those fixed in the plan; every other
argument stays at the package default.

L1, L2 and LL run in env-compress (llmlingua==0.2.2); SC runs in env-sc
(selective-context==0.1.4). R needs no model and runs in env-stats. Imports are lazy
so each environment only needs its own package.

Run as a GPU job: ./run_pilot.sh compress submits one job per compressor.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .config import model_path
from .survival import achieved_keep_rate, tokens

SEED = 86
ITERATIVE_SIZE = 200   # internal plan: LLMLingua window; per text min(200, text length)
RENDER_INDEX = {"S": 0, "A": 1, "C": 2, "SAE": 0, "AAE": 1, "MAE": 0}


def _path(cfg, key: str) -> str:
    m = cfg["compress"]["models"][key]
    return model_path(m["repo"], m["revision"])


# ---------------------------------------------------------------- llmlingua family

def legacy_cache_shim(get_ppl, cache_cls):
    """Wrap llmlingua 0.2.2's PromptCompressor.get_ppl so that a key/value cache passed as a list of
    [key, value] pairs (llmlingua slices the cache this way between windows) is turned into the
    cache object transformers 4.4x expects. Older transformers converted it internally; 4.49 raises
    "'list' object has no attribute 'get_seq_length'" on the third window of a long text. The
    tensors are unchanged (decisions log: cache conversion)."""
    def wrapped(self, *args, past_key_values=None, **kwargs):
        if isinstance(past_key_values, (list, tuple)):
            past_key_values = cache_cls.from_legacy_cache(tuple((k, v) for k, v in past_key_values))
        return get_ppl(self, *args, past_key_values=past_key_values, **kwargs)
    wrapped._idea86_shim = True
    return wrapped


class LLMLinguaFamily:
    def __init__(self, path_llama2: str | None = None, path_xlmr: str | None = None):
        from llmlingua import PromptCompressor  # env-compress only
        from transformers import DynamicCache

        if not getattr(PromptCompressor.get_ppl, "_idea86_shim", False):
            PromptCompressor.get_ppl = legacy_cache_shim(PromptCompressor.get_ppl, DynamicCache)
        self._PC = PromptCompressor
        self.path_llama2, self.path_xlmr = path_llama2, path_xlmr
        self._llama = None
        self._xlmr = None

    @property
    def llama(self):
        if self._llama is None:
            self._llama = self._PC(model_name=self.path_llama2)
        return self._llama

    @property
    def xlmr(self):
        if self._xlmr is None:
            self._xlmr = self._PC(model_name=self.path_xlmr, use_llmlingua2=True)
        return self._xlmr

    def window(self, text: str) -> int:
        """iterative_size for one text: min(200, Llama-2 tokens of the text being compressed).

        llmlingua 0.2.2 assumes the text fills at least one window; for a shorter text it
        marks almost every token as kept (smoke run: 98-100% kept at rate 0.5). With the
        window equal to the text length the library compresses it as intended; texts of 200
        tokens or more use 200, as before (decisions log)."""
        n = self.llama.get_token_length(text, add_special_tokens=False)
        return max(1, min(ITERATIVE_SIZE, n))

    def l1(self, text: str, rho: float) -> dict:
        w = self.window(text)
        res = self.llama.compress_prompt([text], rate=rho, use_context_level_filter=False,
                                         use_token_level_filter=True, iterative_size=w,
                                         condition_in_question="none")
        return {**res, "iterative_size": w}

    def l2(self, text: str, rho: float) -> dict:
        return self.xlmr.compress_prompt([text], rate=rho, force_tokens=["\n", "?"])

    def ll(self, passage: str, question: str, rho: float = 0.5) -> dict:
        # Input fix (decisions log): llmlingua 0.2.2 asserts that the passage start survives a Llama
        # tokenizer round trip after the question; a passage that starts with whitespace fails it.
        # Leading and trailing whitespace is removed; every other passage is passed unchanged.
        stripped = passage != passage.strip()
        passage = passage.strip()
        w = self.window(passage)
        res = self.llama.compress_prompt([passage], question=question, rate=rho, iterative_size=w,
                                         condition_in_question="after_condition", condition_compare=True,
                                         rank_method="longllmlingua", use_context_level_filter=False,
                                         reorder_context="original", context_budget="+100",
                                         dynamic_context_compression_ratio=0.0)
        # llmlingua 0.2.2 appends "\n\n" + question when concate_question=True (default);
        # the stored Arm B passage must not contain the question.
        suffix = "\n\n" + question
        cp = res["compressed_prompt"]
        if not cp.endswith(suffix):
            raise RuntimeError("LongLLMLingua output does not end with the question; check llmlingua version")
        res = dict(res)
        res["compressed_prompt"] = cp[: -len(suffix)]
        res["question_stripped"] = True
        res["iterative_size"] = w
        res["input_stripped"] = stripped
        return res


# ---------------------------------------------------------------- Selective Context

def make_selective_context(path_gpt2: str):
    """SelectiveContext(model_type='gpt2', lang='en') with GPT-2 loaded from the pinned
    local snapshot. The package hard-codes the hub name 'gpt2' and has no revision
    argument, so _prepare_model is overridden to load the same model from the pinned
    path; nothing else changes."""
    from selective_context import SelectiveContext  # env-sc only
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    class PinnedSelectiveContext(SelectiveContext):
        def _prepare_model(self):
            self.tokenizer = GPT2Tokenizer.from_pretrained(path_gpt2)
            self.model = GPT2LMHeadModel.from_pretrained(path_gpt2)
            self.model.to(self.device)
            self.model.eval()
            self.get_self_information = self._get_self_info_via_gpt2

    return PinnedSelectiveContext(model_type="gpt2", lang="en")


def sc_compress(sc, text: str, rho: float, level: str = "phrase", max_tokens: int = 1024) -> dict:
    n_gpt2 = len(sc.tokenizer(text, add_special_tokens=False)["input_ids"]) + 1  # + <|endoftext|>
    if n_gpt2 > max_tokens:
        return {"compressed_prompt": None, "excluded": "over_gpt2_window", "gpt2_tokens": n_gpt2}
    context, _ = sc(text, reduce_ratio=1 - rho, reduce_level=level)
    return {"compressed_prompt": context, "gpt2_tokens": n_gpt2}


# ---------------------------------------------------------------- random control R

def random_delete(text: str, k_keep: int, k_text: int, rendering: str) -> str:
    """Keep k word tokens chosen uniformly without replacement, in original order;
    rng = default_rng(86 + k_text + 1_000_000 * i), i = 0, 1, 2 for S, A, C.
    Deleted words are removed from the string; punctuation stays; spaces are collapsed."""
    toks = tokens(text)
    k_keep = max(0, min(k_keep, len(toks)))
    rng = np.random.default_rng(SEED + k_text + 1_000_000 * RENDER_INDEX[rendering])
    keep = set(rng.choice(len(toks), size=k_keep, replace=False).tolist()) if toks else set()
    out, prev = [], 0
    for i, t in enumerate(toks):
        out.append(text[prev:t.start])
        if i in keep:
            out.append(t.text)
        prev = t.end
    out.append(text[prev:])
    s = "".join(out)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    return s.strip()


# ---------------------------------------------------------------- records and driver

def record(job: dict, res: dict, secs: float) -> dict:
    cp = res.get("compressed_prompt")
    rec = {**{k: v for k, v in job.items() if k not in ("text", "question_text")},
           "compressed": cp, "seconds": secs,
           "origin_tokens": res.get("origin_tokens"), "compressed_tokens": res.get("compressed_tokens"),
           "n_words": len(tokens(job["text"]))}
    if res.get("iterative_size") is not None:
        rec["iterative_size"] = res["iterative_size"]
    if res.get("input_stripped"):
        rec["input_stripped"] = True
    if "excluded" in res:
        rec["excluded"] = res["excluded"]
    if cp is not None:
        rec["achieved_keep_rate"] = achieved_keep_rate(job["text"], cp)
    return rec


def run_jobs(jobs: list[dict], compressor: str, cfg) -> list[dict]:
    rho_default = cfg["compress"]["rho"]
    out = []
    if compressor in ("L1", "L2", "LL"):
        fam = LLMLinguaFamily(_path(cfg, "llama2") if compressor in ("L1", "LL") else None,
                              _path(cfg, "xlmr") if compressor == "L2" else None)
        fn = {"L1": lambda j: fam.l1(j["text"], j.get("rho", rho_default)),
              "L2": lambda j: fam.l2(j["text"], j.get("rho", rho_default)),
              "LL": lambda j: fam.ll(j["text"], j["question_text"], j.get("rho", rho_default))}[compressor]
    elif compressor == "SC":
        sc = make_selective_context(_path(cfg, "gpt2"))
        fn = lambda j: sc_compress(sc, j["text"], j.get("rho", rho_default),  # noqa: E731
                                   cfg["compress"]["sc_reduce_level"], cfg["compress"]["sc_max_gpt2_tokens"])
    else:
        raise SystemExit(f"unknown model compressor {compressor}; R runs via random_control()")
    for j in jobs:
        t0 = time.perf_counter()
        res = fn(j)
        out.append(record(j, res, time.perf_counter() - t0))
    return out


def random_control(jobs: list[dict], matched: dict[str, dict]) -> list[dict]:
    """R for each job, matched to the word tokens kept by the reference compressor
    (L2 for Arm A and tweets, LL for Arm B) on the same text: matched[job_key] is that
    record."""
    out = []
    for j in jobs:
        ref = matched[j["match_key"]]
        k_keep = len(tokens(ref["compressed"])) if ref.get("compressed") is not None else None
        if k_keep is None:
            out.append(record(j, {"compressed_prompt": None, "excluded": "reference_missing"}, 0.0))
            continue
        cp = random_delete(j["text"], k_keep, j["k_text"], j["rendering"])
        out.append(record(j, {"compressed_prompt": cp}, 0.0))
    return out
