"""Descriptive summaries shared by the study stages: rendering methods per text kind and dialect,
dialect markers per passage, and achieved keep rates by text length (internal plan Sections 8, 10)."""
from __future__ import annotations

import numpy as np

LENGTH_GROUPS = [(1, 49), (50, 99), (100, 149), (150, 10**9)]
METHODS = ("whole", "sentence_fallback", "no_coref", "too_long", "failed")


def rendering_summary(rendered: dict) -> dict:
    """Per text kind and dialect: texts by rendering method, no-coreference sentences, quote-protected
    texts and the features that read coreference (could not apply in no-coreference sentences)."""
    out: dict = {}
    for r in rendered.values():
        key = f"{r['kind']}_{r['dialect']}"
        v = out.setdefault(key, {"n": 0, **{m: 0 for m in METHODS}, "no_coref_sentences": 0,
                                 "quote_protected": 0, "coref_dependent_features": []})
        v["n"] += 1
        m = r.get("method") or ("whole" if r.get("ok") else "failed")
        v[m if m in METHODS else ("whole" if r.get("ok") else "failed")] += 1
        v["no_coref_sentences"] += len(r.get("no_coref_sentences") or [])
        v["quote_protected"] += int(bool(r.get("quote_protected")))
        v["coref_dependent_features"] = sorted(set(v["coref_dependent_features"]) | set(r.get("coref_dependent_features") or []))
    for v in out.values():
        n = v["n"] or 1
        v["not_usable_rate"] = (v["failed"] + v["too_long"]) / n
    return dict(sorted(out.items()))


def marker_summary(passages: list[dict], rendered: dict, cfg, dialects) -> dict:
    """Per dialect over usable passages: rule-log entries (markers) per passage, shares with at least
    1 and 3, entries per rule type, and `disabled_fired` (switched-off features seen; must be empty)."""
    out = {}
    for d in dialects:
        cls = cfg["render"]["dialects"][d]
        off = set((cfg["render"].get("disabled_features") or {}).get(cls) or [])
        n, types = [], {}
        for pz in passages:
            r = rendered.get(f"passage:{pz['pos']}:{d}")
            if not r or not r.get("ok"):
                continue
            rules = r.get("rules") or []
            n.append(len(rules))
            for x in rules:
                types[x["type"]] = types.get(x["type"], 0) + 1
        a = np.array(n) if n else np.zeros(1)
        out[d] = {"dialect_class": cls, "passages": len(n), "mean": float(a.mean()), "median": float(np.median(a)),
                  "share_ge1": float((a >= 1).mean()), "share_ge3": float((a >= 3).mean()),
                  "entries": int(a.sum()), "features_off": len(off),
                  "types": dict(sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))),
                  "disabled_fired": sorted(set(types) & off)}
    return out


def keep_by_length(comp: dict) -> dict:
    """Mean achieved word keep rate per compressor (and keep rate, if recorded) and source-text length
    group, and for L1 and LL also by LLMLingua window (shorter than 200 tokens, or 200)."""
    out: dict = {}
    for v in comp.values():
        c, kr, n = v.get("compressor"), v.get("achieved_keep_rate"), v.get("n_words")
        if c is None or kr is None or n is None:
            continue
        tag = f"{c}" + (f" at {v['rho']}" if v.get("rho") is not None else "")
        lo, hi = next(g for g in LENGTH_GROUPS if g[0] <= n <= g[1])
        key = f"{tag}, words {lo}-{hi}" if hi < 10**9 else f"{tag}, words {lo}+"
        out.setdefault(key, []).append(kr)
        if v.get("iterative_size") is not None:
            wk = f"{tag}, window {'200' if v['iterative_size'] >= 200 else '<200 (= text length)'}"
            out.setdefault(wk, []).append(kr)
    return {k: {"mean": float(np.mean(x)), "n": len(x)} for k, x in sorted(out.items())}
