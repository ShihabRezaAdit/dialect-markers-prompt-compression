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
    """Contiguous block of jobs for this task; one dialect object per class per worker."""
    n = len(jobs)
    lo, hi = task * n // ntasks, (task + 1) * n // ntasks
    objs: dict[str, object] = {}
    out = []
    for j in jobs[lo:hi]:
        if j["dialect_class"] not in objs:
            objs[j["dialect_class"]] = make_dialect(j["dialect_class"], morphosyntax)
        res = render_one(objs[j["dialect_class"]], j["text"], j["k"])
        out.append({k: j[k] for k in ("job_id", "kind", "pos", "k", "dialect")} | res)
    return out
