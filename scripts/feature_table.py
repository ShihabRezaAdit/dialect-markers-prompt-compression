"""Pilot P0 follow-up to the failed G5 and G3 (decisions log): evidence for choosing a
meaning-preserving Multi-VALUE feature subset, and post-hoc S1 sensitivity analyses.

  table     per dialect, every rule type that fired in the 100 audited passages: entries,
            audited pairs where it fired, CONTENT_PRESERVED = NO pairs where it fired, and NO
            pairs whose note quotes the SAE words of one of its entries (env-stats)
  features  the morphosyntactic features of AfricanAmericanVernacular, ColloquialSingaporeDialect
            and IndianDialect with their probabilities (env-render; loads value-nlp)
  s1post    post-hoc S1 on the current data: (a) only the audited pairs labelled
            CONTENT_PRESERVED = YES; (b) all pilot paragraphs, with replace ops typed by the
            rule types given in --exclude left out (env-stats). Exploratory; not a gate input.

Usage (cluster, project folder):  python scripts/feature_table.py table|features|s1post [--exclude a,b]
Writes runs/P0/audit/followup_<command>.md and prints it. Counts only; no item text.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "runs" / "P0"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def table() -> str:
    items = json.load(open(RUN / "items.json"))
    rend = json.load(open(RUN / "render" / "rendered.json"))
    labels = jl(RUN / "audit" / "labels_p0.jsonl")
    out = ["# Follow-up: rule types in the audited passages", "",
           "Per dialect: entries = rule-log entries; pairs = audited passages where the type fired; "
           "NO pairs = of those, CONTENT_PRESERVED = NO; quoted = NO pairs whose note quotes the SAE "
           "words of an entry of that type (a lower bound on the type being named as a cause).", ""]
    for d in ("A", "C"):
        ent, pairs, nos, quoted = Counter(), defaultdict(set), defaultdict(set), defaultdict(set)
        L = [r for r in labels if r["dialect"] == d]
        unmatched = []
        for r in L:
            key = r["item_key"]
            ppos = int(key.split(":")[1])
            sae = items["passages"][ppos]["context"]
            rules = rend[key].get("rules", [])
            note = (r.get("note") or "").lower()
            hit = False
            for x in rules:
                t = x["type"]
                ent[t] += 1
                pairs[t].add(r["pair_id"])
                if r["CONTENT_PRESERVED"] == "NO":
                    nos[t].add(r["pair_id"])
                    span = sae[x["start"]: x["end_incl"] + 1].strip().lower()
                    if len(span) >= 2 and span in note:
                        quoted[t].add(r["pair_id"])
                        hit = True
            if r["CONTENT_PRESERVED"] == "NO" and not hit:
                unmatched.append(r["pair_id"])
        n_no = sum(1 for r in L if r["CONTENT_PRESERVED"] == "NO")
        out += [f"## {'AAVE' if d == 'A' else 'CollSgE'} ({len(L)} pairs, {n_no} NO)", "",
                "| Rule type | Entries | Pairs | NO pairs | Quoted in NO notes |", "|---|---|---|---|---|"]
        for t in sorted(ent, key=lambda t: (-len(quoted[t]), -len(nos[t]), t)):
            out.append(f"| {t} | {ent[t]} | {len(pairs[t])} | {len(nos[t])} | {len(quoted[t])} |")
        out += ["", f"NO pairs where no entry's SAE words are quoted in the note: {len(unmatched)} "
                f"({', '.join(unmatched)})", ""]
    return "\n".join(out)


def features() -> str:
    from idea86.render import make_dialect
    out = ["# Follow-up: Multi-VALUE morphosyntactic features and probabilities", ""]
    for cls in ("AfricanAmericanVernacular", "ColloquialSingaporeDialect", "IndianDialect"):
        d = make_dialect(cls, True)
        mt = getattr(d, "morphosyntax_transforms", {})
        out += [f"## {cls} ({len(mt)} features)", "", "| Feature | Probability |", "|---|---|"]
        for k, v in sorted(mt.items(), key=lambda kv: (-float(kv[1]) if isinstance(kv[1], (int, float)) else 0, kv[0])):
            out.append(f"| {k} | {v} |")
        out.append("")
    return "\n".join(out)


def s1post(exclude: list[str]) -> str:
    import numpy as np
    from idea86 import survival as sv
    from idea86.stats import bootstrap, mean_stat, percentile_ci

    items = json.load(open(RUN / "items.json"))
    rend = json.load(open(RUN / "render" / "rendered.json"))
    comp = {}
    for c in ("L1", "L2", "SC", "R"):
        for r in jl(RUN / "compress" / f"{c}.jsonl"):
            comp[r["job_id"]] = r
    labels = jl(RUN / "audit" / "labels_p0.jsonl")
    yes = {(int(r["item_key"].split(":")[1]), r["dialect"]) for r in labels if r["CONTENT_PRESERVED"] == "YES"}
    ex = set(exclude)

    def ds(ppos, d, c, drop):
        sae = items["passages"][ppos]["context"]
        rr = rend.get(f"passage:{ppos}:{d}")
        cs, cd = comp.get(f"A:{ppos}:S:{c}"), comp.get(f"A:{ppos}:{d}:{c}")
        if not rr or not rr.get("ok") or not cs or not cd or cs.get("compressed") is None or cd.get("compressed") is None:
            return None
        st, dt = sv.tokens(sae), sv.tokens(rr["text"])
        ops = sv.diff_ops(st, dt)
        if drop:
            ops = [o for o in ops if not (o.tag == "replace" and sv.type_op(o, st, len(sae), rr.get("rules", [])) in drop)]
        s_s = sv.survived(sv.word_survival(st, sv.tokens(cs["compressed"])))
        s_d = sv.survived(sv.word_survival(dt, sv.tokens(cd["compressed"])))
        return sv.paired_span_delta(ops, s_s, s_d)

    out = ["# Follow-up: post-hoc S1 sensitivity (exploratory; 95% percentile intervals, B = 2000)", ""]
    for title, sel, drop in (("(a) audited pairs labelled CONTENT_PRESERVED = YES", "yes", set()),
                             (f"(b) all pilot paragraphs, replace ops typed {sorted(ex)} left out", "all", ex)):
        if sel == "all" and not ex:
            continue
        out += [f"## {title}", "", "| Cell | Delta_s | 95% interval | n |", "|---|---|---|---|"]
        for c in ("L1", "L2", "SC", "R"):
            for d in ("A", "C"):
                units = sorted(p for p, dd in yes if dd == d) if sel == "yes" else [p["pos"] for p in items["passages"]]
                v = np.array([x for x in (ds(p, d, c, drop) for p in units) if x is not None])
                if len(v) == 0:
                    out.append(f"| {c}_{d} | n/a | | 0 |")
                    continue
                est, draws = bootstrap(mean_stat(v), len(v), 2000)
                lo, hi = percentile_ci(draws, 0.95)
                out.append(f"| {c}_{d} | {est:.4f} | [{lo:.4f}, {hi:.4f}] | {len(v)} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["table", "features", "s1post"])
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    text = {"table": table, "features": features}.get(a.command, lambda: s1post([x for x in a.exclude.split(",") if x]))()
    path = RUN / "audit" / f"followup_{a.command}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
