"""Multi-VALUE rendering with per-text seeds and rule logs (internal plan Section 8).

Runs in env-render (value-nlp==0.3). One dialect object per worker; before each text
the worker calls set_seed(86 + k). The rule log (`executed_rules`) is keyed by
(start, end) character offsets in the SAE string; in value-nlp 0.3 `end` is inclusive
(compile_from_rules continues from end + 1). We store it as a list of
{start, end_incl, value, type} so the convention is explicit.

Run as a Slurm CPU array (slurm/render_array.sbatch): each task renders a contiguous block
of the job list and writes its own shard.
"""
from __future__ import annotations

import time

SEED = 86


def _offline_patches() -> None:
    """value-nlp 0.3 downloads at import (spaCy model via pip, NLTK data) and builds a Stanza
    pipeline that fetches resources.json. Compute nodes have no internet, so when
    HF_HUB_OFFLINE=1 these become no-ops; `warmup` on the login node fills the caches first."""
    import os

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return
    import functools

    import nltk
    import spacy.cli
    import stanza

    if spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download = lambda *a, **k: None
    nltk.download = lambda *a, **k: True
    if not getattr(stanza.Pipeline, "_idea86_offline", False):
        orig = stanza.Pipeline
        patched = functools.partial(orig, download_method=None)
        patched._idea86_offline = True
        stanza.Pipeline = patched


def make_dialect(class_name: str, morphosyntax: bool = True):
    _offline_patches()
    from multivalue import Dialects  # env-render only

    return getattr(Dialects, class_name)(morphosyntax=morphosyntax)


def warmup(cfg) -> None:
    """Login node, internet: fill the spaCy, NLTK and Stanza caches and render one sentence
    per dialect."""
    import os

    import nltk
    import stanza

    nltk_dir = os.environ.get("NLTK_DATA")
    for pkg in ("cmudict", "wordnet", "punkt", "punkt_tab"):
        nltk.download(pkg, download_dir=nltk_dir, quiet=True)
    stanza.download("en", processors="tokenize,mwt,pos,lemma,depparse,coref", verbose=False)
    for code, cls in cfg["render"]["dialects"].items():
        d = make_dialect(cls, cfg["render"]["morphosyntax"])
        r = render_one(d, "She is always studying at the library because they were not there.", 0)
        print(f"[warmup] render {code} ({cls}): ok={r['ok']} -> {r.get('text', r.get('error'))}")


def rule_log(executed_rules: dict) -> list[dict]:
    out = []
    for (start, end), rule in sorted(executed_rules.items()):
        out.append({"start": int(start), "end_incl": int(end),
                    "value": rule.get("value"), "type": rule.get("type")})
    return out


def render_one(dialect, text: str, k: int) -> dict:
    """Render one text; retry once on error (plan: failed texts are recorded, not dropped silently)."""
    last_err = None
    for attempt in (1, 2):
        try:
            dialect.set_seed(SEED + k)
            t0 = time.perf_counter()
            out = dialect.transform(text)
            secs = time.perf_counter() - t0
            return {"ok": True, "text": out, "rules": rule_log(dialect.executed_rules),
                    "counter": {str(a): int(b) for a, b in dict(dialect.modification_counter).items()},
                    "seconds": secs, "attempts": attempt}
        except Exception as e:  # noqa: BLE001 - library errors are heterogeneous
            last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": last_err, "attempts": 2}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of `text` (NLTK Punkt, English model, from NLTK_DATA).
    Spans exclude the whitespace between sentences, which render_passage copies unchanged."""
    import nltk

    return list(nltk.data.load("tokenizers/punkt/english.pickle").span_tokenize(text))


def join_sentences(text: str, spans: list[tuple[int, int]], rendered: list[str]) -> str:
    """Rebuild a passage from rendered sentences, keeping the SAE text between and around the
    sentences (spacing, line breaks) exactly as it is, in the original order."""
    if len(spans) != len(rendered):
        raise ValueError("one rendering per sentence span is required")
    out, prev = [], 0
    for (s, e), r in zip(spans, rendered):
        if s < prev or e < s:
            raise ValueError("sentence spans must be ordered and non-overlapping")
        out.append(text[prev:s])
        out.append(r)
        prev = e
    out.append(text[prev:])
    return "".join(out)


COREF_ERROR = "Spacy and Stanza"   # value-nlp 0.3 alignment checks in create_coref_cluster


def transform_no_coref(dialect, text: str) -> str:
    """Transform one sentence with Multi-VALUE's coreference step turned off: the dialect's
    create_coref_cluster returns no clusters for this call only, so rules that read coreference
    clusters do not apply to this sentence (decisions log: no-coreference last resort)."""
    dialect.create_coref_cluster = lambda string: []
    try:
        return dialect.transform(text)
    finally:
        del dialect.create_coref_cluster   # the class method applies again


def coref_dependent_features(dialect) -> list[str]:
    """Feature methods of this dialect that read Multi-VALUE's coreference clusters, directly or
    through the methods they call (found from the source). These are the rules that cannot
    apply in a no-coreference sentence; the pilot report lists them per dialect."""
    import inspect
    import re as _re

    cls = type(dialect)
    src: dict[str, str] = {}
    for klass in cls.__mro__:
        for name, fn in vars(klass).items():
            if callable(fn) and name not in src:
                try:
                    src[name] = inspect.getsource(fn)
                except (OSError, TypeError):
                    src[name] = ""
    memo: dict[str, bool] = {}

    def uses(name: str, seen: frozenset = frozenset()) -> bool:
        if name in memo:
            return memo[name]
        body = src.get(name, "")
        hit = "coref_clusters" in body or any(
            c not in seen and c != name and uses(c, seen | {name})
            for c in set(_re.findall(r"self\.(\w+)\(", body)) if c in src)
        memo[name] = hit
        return hit

    return sorted(f for f in getattr(dialect, "morphosyntax_transforms", {}) if uses(f))


def render_passage(dialect, text: str, k: int, spans_fn=None) -> dict:
    """Render a passage whole; if that fails after its retry, render it sentence by sentence;
    a sentence that fails the coreference alignment check is rendered without coreference
    (decisions log: sentence fallback; no-coreference last resort).

    Fallback: set_seed(86 + k) once, then each Punkt sentence is transformed in order and the
    results are joined with the SAE text between sentences kept exactly (join_sentences). Rule
    spans are shifted to passage offsets. A sentence whose transform raises the spaCy/Stanza
    alignment error is transformed again with coreference off (transform_no_coref). The whole
    fallback is retried once; if it still fails the passage is recorded as failed.
    `method` is "whole", "sentence_fallback", "no_coref" (at least one no-coreference
    sentence) or "failed"; fallback records keep the whole-text error in `whole_error` and the
    indices of no-coreference sentences in `no_coref_sentences`.
    """
    r = render_one(dialect, text, k)
    if r["ok"]:
        return {**r, "method": "whole"}
    spans = (spans_fn or sentence_spans)(text)
    last_err = None
    for attempt in (1, 2):
        try:
            dialect.set_seed(SEED + k)
            t0 = time.perf_counter()
            outs, rules, counter, no_coref = [], [], {}, []
            for i, (s, e) in enumerate(spans):
                try:
                    outs.append(dialect.transform(text[s:e]))
                except Exception as ex:  # noqa: BLE001
                    if COREF_ERROR not in str(ex):
                        raise
                    outs.append(transform_no_coref(dialect, text[s:e]))
                    no_coref.append(i)
                for x in rule_log(dialect.executed_rules):
                    rules.append({**x, "start": x["start"] + s, "end_incl": x["end_incl"] + s})
                for a, b in dict(dialect.modification_counter).items():
                    counter[str(a)] = counter.get(str(a), 0) + int(b)
            out = {"ok": True, "text": join_sentences(text, spans, outs), "rules": rules, "counter": counter,
                   "seconds": time.perf_counter() - t0, "attempts": attempt,
                   "method": "no_coref" if no_coref else "sentence_fallback",
                   "n_sentences": len(spans), "no_coref_sentences": no_coref, "whole_error": r["error"]}
            if no_coref:
                out["coref_dependent_features"] = coref_dependent_features(dialect)
            return out
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": last_err, "whole_error": r["error"], "attempts": 2, "method": "failed",
            "n_sentences": len(spans)}


def build_jobs(items: dict, cfg: dict) -> list[dict]:
    """One job per (text, dialect). In the smoke run (items["smoke"] is true) the smoke
    seed offsets are used, so smoke texts never share seeds with pilot texts."""
    off = cfg["render"]["k_offsets"]
    smoke = bool(items.get("smoke"))
    groups = [("passages", "passage", off["smoke_passage" if smoke else "pilot_passage"], "context"),
              ("questions", "question", off["smoke_question" if smoke else "pilot_question"], "question")]
    jobs = []
    for key, kind, base, field in groups:
        for it in items[key]:
            for dcode, dclass in cfg["render"]["dialects"].items():
                jobs.append({"job_id": f"{kind}:{it['pos']}:{dcode}", "kind": kind, "pos": it["pos"],
                             "k": base + it["pos"], "dialect": dcode, "dialect_class": dclass,
                             "text": it[field]})
    return jobs


def run_shard(jobs: list[dict], task: int, ntasks: int, morphosyntax: bool = True) -> list[dict]:
    """Contiguous block of jobs for this task; one dialect object per class per worker.
    Questions are rendered whole; passages (and stories) use render_passage, which falls back
    to sentence-by-sentence rendering when the whole text fails."""
    n = len(jobs)
    lo, hi = task * n // ntasks, (task + 1) * n // ntasks
    objs: dict[str, object] = {}
    out = []
    for j in jobs[lo:hi]:
        if j["dialect_class"] not in objs:
            objs[j["dialect_class"]] = make_dialect(j["dialect_class"], morphosyntax)
        d = objs[j["dialect_class"]]
        if j["kind"] == "question":
            res = {**render_one(d, j["text"], j["k"])}
            res["method"] = "whole" if res["ok"] else "failed"
        else:
            res = render_passage(d, j["text"], j["k"])
        out.append({k: j[k] for k in ("job_id", "kind", "pos", "k", "dialect")} | res)
    return out
