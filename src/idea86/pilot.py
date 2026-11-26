"""P0 pilot driver: item lists, job files, collection, measurement, gates, signals,
projection and report (pilot plan; internal plan Section 14). Steps are called by the
command line (python -m idea86 <command>) through run_pilot.sh."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from . import stats as S
from . import survival as sv
from .audit import dialect_names, export as audit_export
from .config import load, provenance, read_json, read_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from .generate import squad_prompt
from .render import build_jobs
from .score import answer_survives, extract_answer, normalize_answer

DIALECTS = ("A", "C")
RENDERINGS = ("S", "A", "C")
ARM_A = ("L1", "L2", "SC", "R")
LEARNED_S1 = ("L1", "L2", "SC")


# ---------------------------------------------------------------- paths and loading

class Run:
    def __init__(self, cfg_path: str = "configs/p0.yaml", smoke: bool = False, cfg=None,
                 exclude_methods: tuple = ()):
        self.cfg = cfg if cfg is not None else load(cfg_path)
        self.cfg_hash = self.cfg.hash
        self.smoke = smoke
        self.root = self.cfg.run_dir(smoke)
        # Sensitivity analyses: treat texts rendered by these methods as missing
        # (e.g. ("no_coref",) or ("sentence_fallback", "no_coref")).
        self.exclude_methods = tuple(exclude_methods)

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def items(self) -> dict:
        return read_json(self.p("items.json"))

    def rendered(self) -> dict:
        r = read_json(self.p("render", "rendered.json"))
        if self.exclude_methods:
            r = {k: ({**v, "ok": False} if v.get("method") in self.exclude_methods else v) for k, v in r.items()}
        return r


def passage_text(items, rendered, ppos, g):
    if g == "S":
        return items["passages"][ppos]["context"]
    r = rendered.get(f"passage:{ppos}:{g}")
    return r["text"] if r and r["ok"] else None


def question_text(items, rendered, qpos, g):
    if g == "S":
        return items["questions"][qpos]["question"]
    r = rendered.get(f"question:{qpos}:{g}")
    return r["text"] if r and r["ok"] else None


# ---------------------------------------------------------------- step: items

def step_items(run: Run) -> None:
    """Freeze the P0 item lists (SQuAD train sample, audit pairs, smoke paragraphs, tweets)."""
    cfg = run.cfg
    revs = read_json(cfg.path("data_dir", "revisions.json"))
    paras = D.load_squad(cfg)
    smp = D.pilot_sample(paras, cfg["sample"]["pilot_paragraphs"], cfg["sample"]["questions_per_paragraph"],
                         cfg["sample"]["smoke_paragraphs"])
    smp["audit_pairs"] = D.audit_sample(cfg["sample"]["pilot_paragraphs"], cfg["sample"]["audit_pairs_per_dialect"],
                                        cfg["sample"]["questions_per_paragraph"])["pairs"]
    src = revs["tweets"]["source"]
    if src == "REAL-AAE":
        csv_path = cfg.path("data_dir", "real_aae", "real_aae.csv")
        if sha256_file(csv_path) != revs["tweets"]["sha256"]:
            raise SystemExit("REAL-AAE file changed since download")
        rows = D.read_real_aae(csv_path, cfg)
        first, last = cfg["sample"]["real_aae_pilot_rows"]
        idx = D.real_aae_sample(len(rows), first, last)
        tweets = {"source": src, "file_sha256": revs["tweets"]["sha256"],
                  "pairs": [{"pos": i, "row": r, "aae": rows[r]["aae_text"], "sae": rows[r]["sae_text"]}
                            for i, r in enumerate(idx)]}
    else:
        aa_file, wh_file = cfg["data"]["ud_twitteraae"]["files"]
        tweets = {"source": src, "files": {}}
        for name, label in ((aa_file, "AAE"), (wh_file, "MAE")):
            fp = cfg.path("data_dir", "ud_twitteraae", name)
            texts = D.read_conllu_texts(fp.read_text(encoding="utf-8"))
            pilot_idx, _ = D.ud_fallback_sample(texts, cfg["sample"]["ud_fallback_pilot_per_file"])
            tweets["files"][label] = {"file": name, "sha256": sha256_file(fp),
                                      "tweets": [{"pos": i, "row": r, "text": texts[r]} for i, r in enumerate(pilot_idx)]}
    smp["tweets"] = tweets
    smp["checksums"] = {
        "paragraph_numbers": sha256_text(json.dumps([p["pnum"] for p in smp["passages"]])),
        "question_ids": sha256_text(json.dumps([q["qid"] for q in smp["questions"]])),
        "audit_pairs": sha256_text(json.dumps(smp["audit_pairs"])),
        "tweet_rows": sha256_text(json.dumps(_tweet_rows(tweets))),
    }
    smp["provenance"] = provenance(run.cfg, "items")
    write_json(run.p("items.json"), smp)
    print(f"items: {len(smp['passages'])} paragraphs, {len(smp['questions'])} questions, tweets: {tweets['source']}")


def step_smoke_items(run_p0: "Run", run_smoke: "Run") -> None:
    """Smoke run items: the 10 smoke paragraphs (20 questions) and 10 tweets, run through
    every stage before P0. Smoke texts are not pilot or study items."""
    it = run_p0.items()
    t = it["tweets"]
    if t["source"] == "REAL-AAE":
        tw = {"source": t["source"], "pairs": t["pairs"][:10]}
    else:
        tw = {"source": t["source"], "files": {k: {**v, "tweets": v["tweets"][:5]} for k, v in t["files"].items()}}
    sm = {"smoke": True, "passages": it["smoke_passages"], "questions": it["smoke_questions"],
          "audit_pairs": [], "tweets": tw, "provenance": provenance(run_smoke.cfg, "smoke-items")}
    write_json(run_smoke.p("items.json"), sm)
    print(f"smoke items: {len(sm['passages'])} paragraphs, {len(sm['questions'])} questions")


def _tweet_rows(t):
    if t["source"] == "REAL-AAE":
        return [p["row"] for p in t["pairs"]]
    return {k: [x["row"] for x in v["tweets"]] for k, v in t["files"].items()}


# ---------------------------------------------------------------- step: rendering

def step_render_jobs(run: Run) -> None:
    jobs = build_jobs(run.items(), run.cfg)
    write_jsonl(run.p("render", "jobs.jsonl"), jobs)
    print(f"render jobs: {len(jobs)}" + ("" if run.smoke else " (planned 1,800)"))


def step_render_collect(run: Run) -> None:
    rendered, fails = {}, []
    for f in sorted(run.p("render", "shards").glob("shard_*.jsonl")):
        for r in read_jsonl(f):
            rendered[r["job_id"]] = r
            if not r["ok"]:
                fails.append(r["job_id"])
    jobs = read_jsonl(run.p("render", "jobs.jsonl"))
    missing = [j["job_id"] for j in jobs if j["job_id"] not in rendered]
    if missing:
        raise SystemExit(f"{len(missing)} render jobs missing, e.g. {missing[:3]}")
    write_json(run.p("render", "rendered.json"), rendered)
    write_json(run.p("render", "failures.json"), fails)
    summ = rendering_summary(rendered)
    write_json(run.p("render", "summary.json"), summ)
    print(f"rendered {len(rendered)}; failed {len(fails)}")
    for key, v in summ.items():
        print(f"  {key}: {v['n']} texts, whole {v['whole']}, sentence fallback {v['sentence_fallback']}, "
              f"no-coref {v['no_coref']} ({v['no_coref_sentences']} sentences), too long {v['too_long']}, "
              f"failed {v['failed']} (not usable {v['failure_rate']:.3f}); quote-protected {v['quote_protected']}")
        if v["coref_dependent_features"]:
            print(f"    features that read coreference (not applied in no-coref sentences): "
                  f"{', '.join(v['coref_dependent_features'])}")
    mk = marker_summary(run.items(), rendered, run.cfg)
    write_json(run.p("render", "markers.json"), mk)
    for d, v in mk.items():
        print(f"  markers {d} ({v['dialect_class']}): {v['passages']} passages, rule-log entries per passage "
              f"mean {v['mean']:.1f}, median {v['median']:.0f}; passages with >= 1 / >= 3: "
              f"{v['share_ge1']:.2f} / {v['share_ge3']:.2f}; {v['features_off']} features off, "
              f"{len(v['types'])} fired")
        print("    fired: " + ", ".join(f"{t} {n}" for t, n in v["types"].items()))
    bad = {d: v["disabled_fired"] for d, v in mk.items() if v["disabled_fired"]}
    if bad:
        raise SystemExit(f"switched-off features appear in the rule logs: {bad}")


def marker_summary(items: dict, rendered: dict, cfg) -> dict:
    """Per dialect, over the usable pilot passages: rule-log entries (markers) per passage, the
    share of passages with at least 1 and at least 3, and every rule type that fired with its
    entry count. `disabled_fired` lists switched-off features found in the logs (must be empty)."""
    out = {}
    for d in DIALECTS:
        cls = cfg["render"]["dialects"][d]
        off = set((cfg["render"].get("disabled_features") or {}).get(cls) or [])
        n, types = [], {}
        for pz in items["passages"]:
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


def rendering_summary(rendered: dict) -> dict:
    """Per text kind and dialect: texts rendered whole, by the sentence fallback, with at least
    one no-coreference sentence, or failed; the number of no-coreference sentences; and the
    Multi-VALUE features that read coreference and so could not apply in those sentences
    (decisions log: sentence fallback; no-coreference last resort)."""
    out: dict = {}
    for r in rendered.values():
        key = f"{r['kind']}_{r['dialect']}"
        v = out.setdefault(key, {"n": 0, "whole": 0, "sentence_fallback": 0, "no_coref": 0, "too_long": 0,
                                 "failed": 0, "no_coref_sentences": 0, "quote_protected": 0,
                                 "coref_dependent_features": []})
        v["n"] += 1
        m = r.get("method") or ("whole" if r.get("ok") else "failed")
        v[m if m in ("whole", "sentence_fallback", "no_coref", "too_long", "failed") else ("whole" if r.get("ok") else "failed")] += 1
        v["quote_protected"] += int(bool(r.get("quote_protected")))
        v["no_coref_sentences"] += len(r.get("no_coref_sentences") or [])
        v["coref_dependent_features"] = sorted(set(v["coref_dependent_features"]) | set(r.get("coref_dependent_features") or []))
    for v in out.values():
        n = v["n"] or 1
        v["failure_rate"] = (v["failed"] + v["too_long"]) / n   # texts not usable (failed or too long)
        v["too_long_share"] = v["too_long"] / n
        v["fallback_share"] = v["sentence_fallback"] / n
        v["no_coref_share"] = v["no_coref"] / n
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- step: audit export

def audit_round2_pairs(items: dict, rendered: dict, per_dialect: int, q_per_par: int = 2) -> dict:
    """Round-2 audit pairs (decisions log): a fresh default_rng(86) permutation of the pilot
    paragraphs not audited in round 1 (items["audit_pairs"]), with one of the two questions per
    position drawn from the same generator. Walking the permutation, paragraphs go to A until
    `per_dialect` usable renderings (rendered and not too long), then to C until `per_dialect`;
    a paragraph whose rendering in the dialect being filled is not usable is skipped and listed."""
    done = {a["ppos"] for a in items["audit_pairs"]}
    pool = [p["pos"] for p in items["passages"] if p["pos"] not in done]
    rng = np.random.default_rng(86)
    perm = rng.permutation(len(pool))
    qpick = rng.integers(0, q_per_par, size=len(pool))
    pairs, skipped, i = [], [], 0
    for d in DIALECTS:
        got = 0
        while got < per_dialect:
            if i >= len(perm):
                raise SystemExit(f"round-2 audit: not enough usable paragraphs for {d}")
            ppos = pool[int(perm[i])]
            r = rendered.get(f"passage:{ppos}:{d}")
            if r and r.get("ok"):
                pairs.append({"ppos": ppos, "dialect": d, "qpos": ppos * q_per_par + int(qpick[i])})
                got += 1
            else:
                skipped.append({"ppos": ppos, "dialect": d, "method": (r or {}).get("method")})
            i += 1
    return {"pairs": pairs, "skipped": skipped}


def audit_pairs(run: Run, items: dict, rendered: dict) -> list[dict]:
    """The audit pairs that G3 and G5 use: round 2 when sample.audit_round is 2 (pilot run),
    else round 1 (and always in the smoke run, which has no round 2)."""
    if run.cfg["sample"].get("audit_round", 1) == 2 and not run.smoke:
        return audit_round2_pairs(items, rendered, run.cfg["sample"]["audit_pairs_per_dialect"],
                                  run.cfg["sample"]["questions_per_paragraph"])["pairs"]
    return items["audit_pairs"]


def labels_path(run: Run) -> Path:
    return run.p("audit", "labels_p0_audit2.jsonl" if run.cfg["sample"].get("audit_round", 1) == 2
                 and not run.smoke else "labels_p0.jsonl")


def step_audit_export(run: Run, sheet: str) -> None:
    items, rendered = run.items(), run.rendered()
    pairs = []
    if sheet == "p0_audit2":
        if run.cfg["sample"].get("audit_round", 1) != 2 or run.smoke:
            raise SystemExit("p0_audit2 needs sample.audit_round: 2 and the pilot run (not --smoke)")
        a2 = audit_round2_pairs(items, rendered, run.cfg["sample"]["audit_pairs_per_dialect"],
                                run.cfg["sample"]["questions_per_paragraph"])
        write_json(run.p("audit", "round2_pairs.json"), a2)
        for a in a2["pairs"]:
            q = items["questions"][a["qpos"]]
            pairs.append({"item_key": f"passage:{a['ppos']}:{a['dialect']}", "dialect": a["dialect"],
                          "sae": items["passages"][a["ppos"]]["context"],
                          "dia": passage_text(items, rendered, a["ppos"], a["dialect"]),
                          "question": q["question"], "golds": q["answers"]})
        print(f"round 2: {len(a2['pairs'])} pairs; skipped (not usable): {len(a2['skipped'])}")
    elif sheet == "p0_audit":
        for a in items["audit_pairs"]:
            q = items["questions"][a["qpos"]]
            dia = passage_text(items, rendered, a["ppos"], a["dialect"])
            if dia is None:
                raise SystemExit(f"audit pair paragraph {a['ppos']} failed to render in {a['dialect']}")
            pairs.append({"item_key": f"passage:{a['ppos']}:{a['dialect']}", "dialect": a["dialect"],
                          "sae": items["passages"][a["ppos"]]["context"], "dia": dia,
                          "question": q["question"], "golds": q["answers"]})
    elif sheet == "warmup_audit":
        if not run.smoke:
            raise SystemExit("warmup_audit is exported from the smoke run (--smoke)")
        rng = np.random.default_rng(86)
        for ppos in sorted(rng.choice(len(items["passages"]), size=run.cfg["sample"]["warmup_pairs"],
                                      replace=False).tolist()):
            g = DIALECTS[ppos % 2]
            dia = passage_text(items, rendered, ppos, g)
            q = items["questions"][ppos * 2]
            if dia is not None:
                pairs.append({"item_key": f"smoke_passage:{ppos}:{g}", "dialect": g,
                              "sae": items["passages"][ppos]["context"], "dia": dia,
                              "question": q["question"], "golds": q["answers"]})
    else:
        raise SystemExit("P0 sheets: warmup_audit (smoke run), p0_audit, p0_audit2")
    audit_export(sheet, pairs, run.p("audit", "export", sheet), run.p("audit", "keys"),
                 names=dialect_names(run.cfg))
    print(f"exported {len(pairs)} pairs -> {run.p('audit', 'export', sheet)}")


# ---------------------------------------------------------------- step: compression jobs

def arm_a_texts(items, rendered):
    """(ppos, g) -> text for every pilot passage rendering that exists."""
    out = {}
    for p in items["passages"]:
        for g in RENDERINGS:
            t = passage_text(items, rendered, p["pos"], g)
            if t is not None:
                out[(p["pos"], g)] = t
    return out


def tweet_texts(items):
    t = items["tweets"]
    out = {}
    if t["source"] == "REAL-AAE":
        for p in t["pairs"]:
            out[(p["pos"], "AAE")] = p["aae"]
            out[(p["pos"], "SAE")] = p["sae"]
    else:
        for label, f in t["files"].items():
            for x in f["tweets"]:
                out[(x["pos"], label)] = x["text"]
    return out


def step_compress_jobs(run: Run) -> None:
    items, rendered = run.items(), run.rendered()
    rho = run.cfg["compress"]["rho"]
    off = run.cfg["render"]["k_offsets"]
    off = {**off, "real_aae_row": off["tweet_row"]}
    a_texts, t_texts = arm_a_texts(items, rendered), tweet_texts(items)
    kp = off["smoke_passage" if run.smoke else "pilot_passage"]
    kq = off["smoke_question" if run.smoke else "pilot_question"]
    jobs = {c: [] for c in ("L1", "L2", "SC", "LL")}
    for c in ("L1", "L2", "SC"):
        for (ppos, g), text in a_texts.items():
            jobs[c].append({"job_id": f"A:{ppos}:{g}:{c}", "arm": "A", "pos": ppos, "rendering": g,
                            "compressor": c, "rho": rho, "text": text, "k_text": kp + ppos})
        for (pos, g), text in t_texts.items():
            jobs[c].append({"job_id": f"T:{pos}:{g}:{c}", "arm": "T", "pos": pos, "rendering": g,
                            "compressor": c, "rho": rho, "text": text, "k_text": off["real_aae_row"] + pos})
    for q in items["questions"]:
        for g in RENDERINGS:
            qt = question_text(items, rendered, q["pos"], g)
            if qt is None:
                continue
            jobs["LL"].append({"job_id": f"B:{q['pos']}:{g}:LL", "arm": "B", "pos": q["pos"], "ppos": q["ppos"],
                               "rendering": g, "compressor": "LL", "rho": rho,
                               "text": items["passages"][q["ppos"]]["context"], "question_text": qt,
                               "k_text": kq + q["pos"]})
    for c, js in jobs.items():
        write_jsonl(run.p("compress", f"jobs_{c}.jsonl"), js)
        print(f"{c}: {len(js)} jobs")


def load_compressed(run: Run) -> dict:
    out = {}
    for c in ("L1", "L2", "SC", "LL", "R"):
        f = run.p("compress", f"{c}.jsonl")
        if f.exists():
            for r in read_jsonl(f):
                out[r["job_id"]] = r
    return out


def step_random_control(run: Run) -> None:
    from .compress import random_control

    comp = load_compressed(run)
    jobs = []
    for f, ref in (("jobs_L2.jsonl", "L2"), ("jobs_LL.jsonl", "LL")):
        for j in read_jsonl(run.p("compress", f)):
            jj = dict(j)
            jj["match_key"] = j["job_id"]
            jj["job_id"] = j["job_id"].rsplit(":", 1)[0] + ":R"
            jj["compressor"] = "R"
            jj.pop("question_text", None)
            jobs.append(jj)
    rows = random_control(jobs, comp)
    write_jsonl(run.p("compress", "R.jsonl"), rows)
    print(f"R: {len(rows)} texts")


# ---------------------------------------------------------------- step: prompts

def step_prompts(run: Run) -> None:
    items, rendered, comp = run.items(), run.rendered(), load_compressed(run)
    prompts = []
    for q in items["questions"]:
        for g in RENDERINGS:
            qt = question_text(items, rendered, q["pos"], g)
            if qt is None:
                continue
            for cond in ("U",) + ARM_A:
                if cond == "U":
                    passage = passage_text(items, rendered, q["ppos"], g)
                else:
                    r = comp.get(f"A:{q['ppos']}:{g}:{cond}")
                    passage = r["compressed"] if r else None
                if passage is not None:
                    prompts.append({"arm": "A", "qpos": q["pos"], "ppos": q["ppos"], "rendering": g,
                                    "condition": cond, "prompt": squad_prompt(passage, qt)})
            for cond in ("U", "LL"):
                passage = (items["passages"][q["ppos"]]["context"] if cond == "U"
                           else (comp.get(f"B:{q['pos']}:{g}:LL") or {}).get("compressed"))
                if passage is not None:
                    prompts.append({"arm": "B", "qpos": q["pos"], "ppos": q["ppos"], "rendering": g,
                                    "condition": cond, "prompt": squad_prompt(passage, qt)})
    write_jsonl(run.p("generate", "prompts.jsonl"), prompts)
    uniq = len({p["prompt"] for p in prompts})
    print(f"prompts: {len(prompts)} per model ({uniq} unique)" + ("" if run.smoke else "; planned 600 x 21 = 12,600"))


# ---------------------------------------------------------------- step: G1 re-run job files

def step_g1_jobs(run: Run) -> None:
    g = run.cfg["gates"]
    n = g["g1_render_pairs_per_dialect"]
    rjobs = read_jsonl(run.p("render", "jobs.jsonl"))
    npar = len(run.items()["passages"])
    n = min(n, npar)
    pick = set(np.random.default_rng(86).choice(npar, n, replace=False).tolist())
    g1r = [j for j in rjobs if j["kind"] == "passage" and j["pos"] in pick]
    write_jsonl(run.p("g1", "render_jobs.jsonl"), g1r)
    for c in ("L1", "L2", "SC", "LL"):
        js = read_jsonl(run.p("compress", f"jobs_{c}.jsonl"))
        js = [j for j in js if j["arm"] in ("A", "B")]
        idx = np.random.default_rng(86).choice(len(js), min(len(js), g["g1_compress_texts_per_compressor"]), replace=False)
        write_jsonl(run.p("g1", f"jobs_{c}.jsonl"), [js[i] for i in sorted(idx)])
    prompts = read_jsonl(run.p("generate", "prompts.jsonl"))
    idx = np.random.default_rng(86).choice(len(prompts), min(len(prompts), g["g1_generate_prompts_per_model"]), replace=False)
    write_jsonl(run.p("g1", "prompts.jsonl"), [prompts[i] for i in sorted(idx)])
    print(f"G1 job files: {len(g1r)} renders, 4 x 20 compressions, 200 prompts")


# ---------------------------------------------------------------- analysis

def _load_generations(run: Run, models: list[str]) -> dict:
    """ans[model][(arm, qpos, rendering, condition)] = normalized answer; also raw rows."""
    ans, rows = {}, {}
    for m in models:
        f = run.p("generate", f"{m}.jsonl")
        if not f.exists():
            continue
        rs = [r for r in read_jsonl(f) if not r.get("_meta")]
        rows[m] = rs
        ans[m] = {(r["arm"], r["qpos"], r["rendering"], r["condition"]): r["answer"] for r in rs}
    return {"ans": ans, "rows": rows}


def measure(run: Run) -> dict:
    items, rendered, comp = run.items(), run.rendered(), load_compressed(run)
    cfg = run.cfg
    res: dict = {"retention": {}, "keep_rates": {}, "answer_survival": {}, "ratio": {}}
    sae_corpus = [p["context"] for p in items["passages"]]
    fs = sv.FreqStrata(sae_corpus)

    # Arm A retention: Delta_s(p) and r per compressor and dialect
    res["retention_partial_kept"] = {}
    for c in ARM_A:
        for d in DIALECTS:
            vals, cnts, excluded, sens = {}, [], 0, []
            for p in items["passages"]:
                sae, dia = p["context"], passage_text(items, rendered, p["pos"], d)
                cs, cd = comp.get(f"A:{p['pos']}:S:{c}"), comp.get(f"A:{p['pos']}:{d}:{c}")
                if dia is None or not cs or not cd or cs.get("compressed") is None or cd.get("compressed") is None:
                    excluded += 1
                    continue
                # Primary: partial words count as deleted (internal plan Section 11, step 4).
                v = M.delta_s_text(sae, dia, cs["compressed"], cd["compressed"])
                if v is None:
                    excluded += 1
                else:
                    vals[p["pos"]] = v
                    # Sensitivity analysis only: partial words counted as kept.
                    sens.append(M.delta_s_text(sae, dia, cs["compressed"], cd["compressed"],
                                               partial_as_kept=True))
                cnts.append(M.ratio_counts_text(sae, dia, cd["compressed"], fs))
            res["retention"][f"{c}_{d}"] = {"values": vals, "excluded": excluded}
            if sens:
                est, draws = S.bootstrap(S.mean_stat(np.array(sens)), len(sens), cfg["stats"]["bootstrap_resamples"])
                res["retention_partial_kept"][f"{c}_{d}"] = {"delta_s": est, "ci95": S.percentile_ci(draws, 0.95),
                                                             "n_paragraphs": len(sens)}
            if cnts:
                cnt = np.array(cnts)
                est, draws = S.bootstrap(M.ratio_stat(cnt), len(cnt), cfg["stats"]["bootstrap_resamples"])
                res["ratio"][f"{c}_{d}"] = {"r": est, "ci95": S.percentile_ci(draws, 0.95)}
    # Tweets: REAL-AAE (Delta_s and r) or the UD-TwitterAAE fallback (r only)
    res["tweets"] = tweet_measures(run, comp)
    # achieved keep rates (G2)
    for c in ARM_A:
        for g in RENDERINGS:
            ks = [comp[k]["achieved_keep_rate"] for k in comp
                  if k.startswith("A:") and k.endswith(f":{g}:{c}") and comp[k].get("achieved_keep_rate") is not None]
            res["keep_rates"][f"{c}_{g}"] = {"mean": float(np.mean(ks)) if ks else None, "n": len(ks)}
    res["sc_excluded"] = sum(1 for k, v in comp.items() if k.endswith(":SC") and v.get("excluded"))
    res["keep_by_length"] = keep_by_length(comp)

    # Arm B answer survival
    for c in ("LL", "R"):
        cmap = {(int(k.split(":")[1]), k.split(":")[2]): v["compressed"]
                for k, v in comp.items() if k.startswith("B:") and k.endswith(f":{c}") and v.get("compressed") is not None}
        for d in DIALECTS:
            sums, counts, keys = M.answer_survival_arrays(items["questions"], cmap, d)
            est, draws = S.bootstrap(S.item_mean_stat(sums, counts), len(sums), cfg["stats"]["bootstrap_resamples"])
            res["answer_survival"][f"{c}_{d}"] = {"estimate": est, "draws": draws.tolist(),
                                                  "discordance": M.discordance(items["questions"], cmap, d),
                                                  "n_questions": int(counts.sum())}
            if c == "LL":
                res["answer_survival"][f"{c}_{d}"]["within_paragraph_corr"] = _icc_pairs(items["questions"], cmap, d)

    # Rendering answer preservation (G5, automatic half) and Arm A primary set
    pres = {}
    for d in DIALECTS:
        flags = {}
        for q in items["questions"]:
            dia = passage_text(items, rendered, q["ppos"], d)
            flags[q["pos"]] = bool(dia is not None and answer_survives(q["answers"], dia))
        pres[d] = flags
    res["answer_preserved_share"] = {d: float(np.mean(list(pres[d].values()))) for d in DIALECTS}
    res["primary_set"] = sorted(q for q in pres["A"] if pres["A"][q] and pres["C"][q])
    res["rendering"] = rendering_summary(read_json(run.p("render", "rendered.json")))
    return res


LENGTH_GROUPS = ((1, 49), (50, 99), (100, 149), (150, 10**9))


def keep_by_length(comp: dict) -> dict:
    """Mean achieved word keep rate per compressor and source-text length group (word tokens),
    and for L1 and LL also by LLMLingua window (shorter than 200 tokens, or 200) (decisions
    log: LLMLingua window)."""
    out: dict = {}
    for v in comp.values():
        c, kr, n = v.get("compressor"), v.get("achieved_keep_rate"), v.get("n_words")
        if c is None or kr is None or n is None:
            continue
        lo, hi = next(g for g in LENGTH_GROUPS if g[0] <= n <= g[1])
        key = f"{c}, words {lo}-{hi}" if hi < 10**9 else f"{c}, words {lo}+"
        out.setdefault(key, []).append(kr)
        if v.get("iterative_size") is not None:
            wk = f"{c}, window {'200' if v['iterative_size'] >= 200 else '<200 (= text length)'}"
            out.setdefault(wk, []).append(kr)
    return {k: {"mean": float(np.mean(x)), "n": len(x)} for k, x in sorted(out.items())}


def tweet_measures(run: Run, comp: dict) -> dict:
    items, cfg = run.items(), run.cfg
    B = cfg["stats"]["bootstrap_resamples"]
    tw, out = items["tweets"], {"source": items["tweets"]["source"]}
    if tw["source"] == "REAL-AAE":
        fs = sv.FreqStrata([p["sae"] for p in tw["pairs"]])
        for c in ARM_A:
            vals, cnts = [], []
            for p in tw["pairs"]:
                cs, ca = comp.get(f"T:{p['pos']}:SAE:{c}"), comp.get(f"T:{p['pos']}:AAE:{c}")
                if not cs or not ca or cs.get("compressed") is None or ca.get("compressed") is None:
                    continue
                v = M.delta_s_text(p["sae"], p["aae"], cs["compressed"], ca["compressed"])
                if v is not None:
                    vals.append(v)
                cnts.append(M.ratio_counts_text(p["sae"], p["aae"], ca["compressed"], fs))
            rec = {"n_pairs_with_replace": len(vals)}
            if vals:
                est, draws = S.bootstrap(S.mean_stat(np.array(vals)), len(vals), B)
                rec |= {"delta_s": est, "ci95": S.percentile_ci(draws, 0.95)}
            if cnts:
                est, draws = S.bootstrap(M.ratio_stat(np.array(cnts)), len(cnts), B)
                rec |= {"r": est, "r_ci95": S.percentile_ci(draws, 0.95)}
            out[c] = rec
    else:
        lex_path = cfg.path("data_dir", "real_aae_lexicon.tsv")
        if not lex_path.exists():
            raise SystemExit(f"UD-TwitterAAE fallback needs the REAL-AAE lexicon (Wu et al., Table 8) at {lex_path}")
        lex = M.load_lexicon(lex_path)
        files = tw["files"]
        fs = sv.FreqStrata([x["text"] for x in files["MAE"]["tweets"]])
        for c in ARM_A:
            cnts, keep = [], {"AAE": [], "MAE": []}
            for label in ("AAE", "MAE"):
                for x in files[label]["tweets"]:
                    r = comp.get(f"T:{x['pos']}:{label}:{c}")
                    if not r or r.get("compressed") is None:
                        continue
                    keep[label].append(r["achieved_keep_rate"])
                    if label == "AAE":
                        cnts.append(M.lexicon_ratio_counts(x["text"], r["compressed"], lex, fs))
            rec = {"keep_rate_AAE": float(np.mean(keep["AAE"])) if keep["AAE"] else None,
                   "keep_rate_MAE": float(np.mean(keep["MAE"])) if keep["MAE"] else None}
            if cnts:
                est, draws = S.bootstrap(M.ratio_stat(np.array(cnts)), len(cnts), B)
                rec |= {"r": est, "r_ci95": S.percentile_ci(draws, 0.95)}
            out[c] = rec
    return out


def _icc_pairs(questions, cmap, d):
    """Correlation of the two questions' paired differences within a paragraph."""
    by_p = defaultdict(list)
    for q in questions:
        a, b = cmap.get((q["pos"], "S")), cmap.get((q["pos"], d))
        if a is None or b is None:
            continue
        by_p[q["ppos"]].append(float(answer_survives(q["answers"], a)) - float(answer_survives(q["answers"], b)))
    x = np.array([v for v in by_p.values() if len(v) == 2])
    if len(x) < 3 or x[:, 0].std() == 0 or x[:, 1].std() == 0:
        return None
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])


def signals(run: Run, res: dict) -> dict:
    sg = run.cfg["signals"]
    B = run.cfg["stats"]["bootstrap_resamples"]
    out = {"S1": {}, "S2": {}}
    lvl1 = 1 - sg["s1_alpha"] / sg["s1_cells"]
    for c in LEARNED_S1:
        for d in DIALECTS:
            vals = np.array(list(res["retention"][f"{c}_{d}"]["values"].values()))
            est, draws = S.bootstrap(S.mean_stat(vals), len(vals), B)
            out["S1"][f"{c}_{d}"] = S.signal(est, draws, sg["s1_floor"], lvl1) | {
                "n_paragraphs": len(vals), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None}
    lvl2 = 1 - sg["s2_alpha"] / sg["s2_cells"]
    for d in DIALECTS:
        a = res["answer_survival"][f"LL_{d}"]
        out["S2"][d] = S.signal(a["estimate"], np.array(a["draws"]), sg["s2_floor"], lvl2) | {
            "discordance": a["discordance"], "n_questions": a["n_questions"]}
    s1 = any(v["met"] for v in out["S1"].values())
    s2 = any(v["met"] for v in out["S2"].values())
    out["S1_met"], out["S2_met"] = s1, s2
    out["recommendation"] = {(True, True): "Go", (True, False): "Go, retention focus",
                             (False, True): "Go, realistic-pipeline focus", (False, False): "No-go"}[(s1, s2)]
    return out


def gates(run: Run, res: dict) -> dict:
    g = run.cfg["gates"]
    items, rendered = run.items(), run.rendered()
    out = {}
    # G1: comparisons of re-runs, if present
    g1 = {}
    f = run.p("g1", "render_shards")
    if f.exists():
        rer = {r["job_id"]: r for s in sorted(f.glob("*.jsonl")) for r in read_jsonl(s)}
        same = sum(1 for k, r in rer.items() if r.get("ok") and rendered[k].get("text") == r["text"])
        g1["render"] = {"identical": same, "total": len(rer), "pass": same == len(rer) == 2 * g["g1_render_pairs_per_dialect"]}
    comp = load_compressed(run)
    tot = same = 0
    for c in ("L1", "L2", "SC", "LL"):
        fc = run.p("g1", f"{c}.jsonl")
        if fc.exists():
            for r in read_jsonl(fc):
                tot += 1
                same += r.get("compressed") == comp[r["job_id"]].get("compressed")
    if tot:
        g1["compress"] = {"identical": same, "total": tot, "pass": same == tot == 80}
    for m in run.cfg["generate"]["models"]:
        fg = run.p("g1", f"{m}.jsonl")
        fo = run.p("generate", f"{m}.jsonl")
        if fg.exists() and fo.exists():
            orig = {r["prompt_hash"]: r["answer"] for r in read_jsonl(fo) if not r.get("_meta")}
            rr = [r for r in read_jsonl(fg) if not r.get("_meta")]
            k = sum(normalize_answer(r["answer"]) == normalize_answer(orig[r["prompt_hash"]]) for r in rr)
            g1[f"generate_{m}"] = {"same": k, "total": len(rr), "pass": k >= g["g1_generate_min_same"]}
    out["G1"] = g1 | {"pass": bool(g1) and all(v["pass"] for v in g1.values())}
    # G2
    lo, hi = g["g2_band"]
    g2 = {}
    for c in ARM_A:
        means = {r: res["keep_rates"][f"{c}_{r}"]["mean"] for r in RENDERINGS}
        band = all(m is not None and lo <= m <= hi for m in means.values())
        diff = all(means[d] is not None and means["S"] is not None and abs(means[d] - means["S"]) <= g["g2_max_diff"]
                   for d in DIALECTS)
        g2[c] = {"means": means, "band_ok": band, "diff_ok": diff, "pass": band and diff}
    out["G2"] = g2 | {"pass": all(v["pass"] for v in g2.values())}
    # G3: rule-log spans covered by diff ops in the 100 audited paragraphs
    cov = tot = 0
    for a in audit_pairs(run, items, rendered):
        r = rendered.get(f"passage:{a['ppos']}:{a['dialect']}")
        if not r or not r["ok"]:
            continue
        sae = items["passages"][a["ppos"]]["context"]
        st, dt = sv.tokens(sae), sv.tokens(r["text"])
        c_, t_ = sv.rules_covered(r["rules"], sv.diff_ops(st, dt), st, len(sae))
        cov += c_
        tot += t_
    share = cov / tot if tot else float("nan")
    out["G3"] = {"covered": cov, "rules": tot, "share": share, "pass": tot > 0 and share >= g["g3_min_overlap"]}
    # G4: Delta_s under R pooled over A and C
    pooled = (list(res["retention"]["R_A"]["values"].values()) + list(res["retention"]["R_C"]["values"].values()))
    g4 = float(np.mean(pooled)) if pooled else float("nan")
    by_d = {d: float(np.mean(list(res["retention"][f"R_{d}"]["values"].values())))
            if res["retention"][f"R_{d}"]["values"] else float("nan") for d in DIALECTS}
    out["G4"] = {"delta_s": g4, "units": len(pooled), "pass": abs(g4) <= g["g4_max_abs"],
                 "by_dialect": by_d,
                 # report flag (not part of the gate): control not near zero in a dialect
                 "flag_by_dialect": {d: not (abs(v) <= g["g4_max_abs"]) for d, v in by_d.items()}}
    # G5: audit labels and automatic answer preservation
    lab = labels_path(run)
    g5 = {"answer_preserved_share": res["answer_preserved_share"]}
    if lab.exists():
        L = read_jsonl(lab)
        yes = {d: sum(1 for r in L if r["dialect"] == d and r["CONTENT_PRESERVED"] == "YES") for d in DIALECTS}
        n = {d: sum(1 for r in L if r["dialect"] == d) for d in DIALECTS}
        g5["content_yes"] = yes
        g5["n"] = n
        g5["pass_by_dialect"] = {d: yes[d] >= g["g5_min_content_yes"] and
                                 res["answer_preserved_share"][d] >= g["g5_min_answer_preserved"] for d in DIALECTS}
        g5["pass"] = all(g5["pass_by_dialect"].values())
    else:
        g5["pass"] = None
        g5["note"] = f"{lab.name} not imported yet"
    out["G5"] = g5
    return out


def accuracy(run: Run, res: dict) -> dict:
    """Preliminary (not powered) Delta_gamma and Delta_phi per model, with 95% intervals."""
    items = run.items()
    gen = _load_generations(run, list(run.cfg["generate"]["models"]))
    qmap = {q["pos"]: q["ppos"] for q in items["questions"]}
    prim = set(res["primary_set"])
    qs = [q for q in items["questions"] if q["pos"] in prim]
    out = {}
    for m, ans_all in gen["ans"].items():
        a = {(k[1], k[2], k[3]): v for k, v in ans_all.items() if k[0] == "A"}
        for c in ARM_A:
            for d in DIALECTS:
                for name, fn in (("gamma", M.gap_items), ("phi", M.flip_items)):
                    items_v = fn(qs, a, d, c)
                    if not items_v:
                        continue
                    sums, counts = M.cluster_sums(items_v, qmap)
                    est, draws = S.bootstrap(S.item_mean_stat(sums, counts), len(sums), run.cfg["stats"]["bootstrap_resamples"])
                    vals = np.array(list(items_v.values()))
                    out[f"{name}_{m}_{c}_{d}"] = {"estimate": est, "ci95": S.percentile_ci(draws, 0.95),
                                                  "item_sd": float(vals.std(ddof=1)), "n": len(vals)}
        rows = gen["rows"][m]
        out[f"length_stop_{m}"] = float(np.mean([r["length_stop"] for r in rows]))
    return out


def projection(run: Run, res: dict, sig: dict, acc: dict) -> dict:
    from scipy.stats import norm

    zb = norm.ppf(0.8)

    def mde(sd, n, deff, alpha):
        return (norm.ppf(1 - alpha / 2) + zb) * sd * math.sqrt(deff / n)

    out = {"delta_s_mde": {}, "delta_a_mde": {}, "delta_gamma_mde": {}}
    for k, v in sig["S1"].items():
        if v["sd"]:
            out["delta_s_mde"][k] = mde(v["sd"], 2067, 1, 0.05 / 6)
    m = 10570 / 2067
    for d in DIALECTS:
        a = res["answer_survival"][f"LL_{d}"]
        rho = a.get("within_paragraph_corr") or 0.3
        pi = a["discordance"]
        if pi and pi > 0:
            out["delta_a_mde"][d] = mde(math.sqrt(pi), 10570, 1 + (m - 1) * max(rho, 0), 0.05 / 2)
    share = len(res["primary_set"]) / max(1, len(run.items()["questions"]))
    sds = [v["item_sd"] for k, v in acc.items() if k.startswith("gamma_")]
    if sds:
        sd = float(np.median(sds))
        n = 10570 * share
        out["delta_gamma_mde"] = {"item_sd_median": sd, "answer_preserved_share": share,
                                  "four_models_alpha_0.05/48": mde(sd, n, 1 + (m * share - 1) * 0.2, 0.05 / 48)}
    # runtimes
    rendered = run.rendered()
    secs = [r["seconds"] for r in rendered.values() if r.get("ok")]
    if secs:
        per = float(np.mean(secs))
        wall_h = 26274 * per / 64 / 3600
        out["render"] = {"seconds_per_text": per, "E1_wall_hours_64_tasks": wall_h,
                         "rendering_time_rule_triggered": wall_h > 48}
    comp = load_compressed(run)
    for c in ("L1", "L2", "SC", "LL"):
        s = [v["seconds"] for v in comp.values() if v.get("compressor") == c and v.get("seconds")]
        if s:
            out[f"compress_seconds_{c}"] = float(np.mean(s))
    if "compress_seconds_LL" in out:
        h = 31710 * out["compress_seconds_LL"] / 3600
        out["arm_b_ll_gpu_hours"] = h
        out["arm_b_time_rule_triggered"] = h > 24
    return out


def control_adjusted(run: Run, res: dict) -> dict:
    """Per learned compressor and dialect: Delta_s(compressor) - Delta_s(R) per paragraph (the
    paragraphs that have both), mean with a 95% cluster-bootstrap interval. Descriptive (pilot
    report); not a signal and not a gate."""
    B = run.cfg["stats"]["bootstrap_resamples"]
    out = {}
    for c in LEARNED_S1:
        for d in DIALECTS:
            a, r = res["retention"][f"{c}_{d}"]["values"], res["retention"][f"R_{d}"]["values"]
            keys = sorted(set(a) & set(r), key=str)
            if not keys:
                out[f"{c}_{d}"] = {"estimate": None, "ci95": [None, None], "n_paragraphs": 0}
                continue
            v = np.array([a[k] - r[k] for k in keys], dtype=float)
            est, draws = S.bootstrap(S.mean_stat(v), len(v), B)
            out[f"{c}_{d}"] = {"estimate": est, "ci95": list(S.percentile_ci(draws, 0.95)), "n_paragraphs": len(v)}
    return out


def round1_gates(run: Run):
    """The original (round-1) gate results, kept in the report (decisions log)."""
    f = run.root.parent / "P0_round1" / "results" / "gates.json"
    return read_json(f) if (not run.smoke and f.exists()) else None


def write_report(run: Run, res, sig, gts, acc, proj, nf=None) -> None:
    L = ["# P0 pilot report", "",
         f"Config SHA-256: `{run.cfg_hash}`. Preliminary results; not confirmatory.", "",
         f"**Recommended outcome: {sig['recommendation']}** (S1 met: {sig['S1_met']}; S2 met: {sig['S2_met']}). "
         "The decision is taken with the advisor; any departure is logged.", "", "## Gates", ""]
    for k in ("G1", "G2", "G3", "G4", "G5"):
        L.append(f"- **{k}**: pass = {gts[k].get('pass')}  \n  `{json.dumps({x: y for x, y in gts[k].items() if x != 'pass'}, default=str)[:900]}`")
    L += ["", "## S1 retention (99.17% intervals)", "", "| Cell | Delta_s | Interval | n | SD | Met |", "|---|---|---|---|---|---|"]
    for k, v in sig["S1"].items():
        L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['n_paragraphs']} | {v['sd'] if v['sd'] is None else round(v['sd'], 3)} | {v['met']} |")
    L += ["", "## S2 answer survival (97.5% intervals)", "", "| Dialect | Delta_a | Interval | Discordance | n | Met |", "|---|---|---|---|---|---|"]
    for k, v in sig["S2"].items():
        L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['discordance']:.3f} | {v['n_questions']} | {v['met']} |")
    r1 = round1_gates(run)
    if r1:
        L += ["", "## Original gate results (round 1: ColloquialSingaporeDialect, all Multi-VALUE features)", "",
              "Kept as run (decisions log); G3 and G5 failed there and were re-checked on the new rendering "
              "with the same measures and thresholds.", ""]
        for k in ("G1", "G2", "G3", "G4", "G5"):
            if k in r1:
                L.append(f"- **{k}** (round 1): pass = {r1[k].get('pass')}  \n  "
                         f"`{json.dumps({x: y for x, y in r1[k].items() if x != 'pass'}, default=str)[:900]}`")
    ca = control_adjusted(run, res)
    L += ["", "## Compressor minus random control (descriptive; 95% intervals; not a signal)", "",
          "| Cell | Delta_s(c) - Delta_s(R) | 95% interval | n |", "|---|---|---|---|"]
    for k, v in ca.items():
        if v["estimate"] is None:
            L.append(f"| {k} | - | - | 0 |")
        else:
            L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n_paragraphs']} |")
    L.append("")
    for d, v in gts["G4"].get("by_dialect", {}).items():
        flag = gts["G4"]["flag_by_dialect"][d]
        L.append(f"- Random control in {d}: Delta_s(R) = {v:.4f}" +
                 (f" -- **flag: not near zero (|value| > {run.cfg['gates']['g4_max_abs']})**" if flag else " (near zero)"))
    summ = rendering_summary(read_json(run.p("render", "rendered.json")))
    L += ["", "## Rendering: whole, sentence fallback, no-coreference sentences, too long, failures", "",
          "| Texts | n | Whole | Sentence fallback | No-coref passages | No-coref sentences | Too long (> max ratio) | Failed | Not usable | Quote-protected |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summ.items():
        L.append(f"| {k} | {v['n']} | {v['whole']} | {v['sentence_fallback']} | {v['no_coref']} | "
                 f"{v['no_coref_sentences']} | {v.get('too_long', 0)} | {v['failed']} | {v['failure_rate']:.3f} | "
                 f"{v.get('quote_protected', 0)} |")
    L.append("")
    mk = marker_summary(run.items(), read_json(run.p("render", "rendered.json")), run.cfg)
    L += ["## Dialect markers per pilot passage (rule-log entries) and features that fired", "",
          "| Dialect | Class | Passages | Mean | Median | >= 1 | >= 3 | Features off | Types fired |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d, v in mk.items():
        L.append(f"| {d} | {v['dialect_class']} | {v['passages']} | {v['mean']:.1f} | {v['median']:.0f} | "
                 f"{v['share_ge1']:.2f} | {v['share_ge3']:.2f} | {v['features_off']} | {len(v['types'])} |")
    L.append("")
    for d, v in mk.items():
        L.append(f"- {d} fired: " + ", ".join(f"{t} {n}" for t, n in v["types"].items()))
    L.append("")
    for k, v in summ.items():
        if v["coref_dependent_features"]:
            L.append(f"- {k}: Multi-VALUE features that read coreference, so they could not apply in the "
                     f"no-coref sentences: {', '.join(v['coref_dependent_features'])}")
    for title, sa in (nf or {}).items():
        L += ["", f"## Sensitivity analysis: {title}", "",
              f"Recommended outcome under this analysis: {sa['sig']['recommendation']} "
              f"(S1 met: {sa['sig']['S1_met']}; S2 met: {sa['sig']['S2_met']}).", "",
              "| S1 cell | Delta_s | 99.17% interval | n | Met |", "|---|---|---|---|---|"]
        for k, v in sa["sig"]["S1"].items():
            L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['n_paragraphs']} | {v['met']} |")
        L += ["", "| S2 dialect | Delta_a | 97.5% interval | n | Met |", "|---|---|---|---|---|"]
        for k, v in sa["sig"]["S2"].items():
            L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['n_questions']} | {v['met']} |")
        L += ["", f"Answer-preserved share: {json.dumps(sa['res']['answer_preserved_share'])}", ""]
        for k, v in sa["acc"].items():
            if isinstance(v, dict):
                L.append(f"- {k}: {v['estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] (n {v['n']})")
    L += ["", "## Sensitivity analysis: partial words counted as kept (not used by S1 or G4)", "",
          "| Cell | Delta_s | 95% interval | n |", "|---|---|---|---|"]
    for k, v in res["retention_partial_kept"].items():
        L.append(f"| {k} | {v['delta_s']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n_paragraphs']} |")
    L += ["", "## Random control and ratio", ""]
    for k in ("R_A", "R_C"):
        vals = list(res["retention"][k]["values"].values())
        L.append(f"- Delta_s under {k}: {np.mean(vals):.4f} over {len(vals)} paragraphs")
    for k, v in res["ratio"].items():
        L.append(f"- r {k}: {v['r']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    L += ["", f"## Tweets ({res['tweets']['source']})", "", "```", json.dumps(res["tweets"], indent=1, default=str), "```"]
    L += ["", "## Achieved keep rate by compressor and text length", "",
          "| Compressor and group | Mean word keep rate | Texts |", "|---|---|---|"]
    for k, v in res.get("keep_by_length", {}).items():
        L.append(f"| {k} | {v['mean']:.3f} | {v['n']} |")
    L += ["", f"Selective Context texts over the GPT-2 window: {res['sc_excluded']}", "",
          "## Preliminary accuracy (labeled as not powered)", ""]
    for k, v in acc.items():
        if isinstance(v, dict):
            L.append(f"- {k}: {v['estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] (item SD {v['item_sd']:.3f}, n {v['n']})")
        else:
            L.append(f"- {k}: {v:.4f}")
    L += ["", "## Projection to the full study", "", "```", json.dumps(proj, indent=1, default=str), "```", "",
          "Audit table, worked examples and decisions-log proposals are added by the project lead."]
    path = run.p("report", "P0_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def step_analyze(run: Run) -> None:
    res = measure(run)
    sig = signals(run, res)
    gts = gates(run, res)
    acc = accuracy(run, res)
    proj = projection(run, res, sig, acc)
    slim = {k: v for k, v in res.items() if k != "answer_survival"}
    slim["answer_survival"] = {k: {x: y for x, y in v.items() if x != "draws"} for k, v in res["answer_survival"].items()}
    write_json(run.p("results", "measure.json"), slim)
    write_json(run.p("results", "signals.json"), sig)
    write_json(run.p("results", "gates.json"), gts)
    write_json(run.p("results", "accuracy.json"), acc)
    write_json(run.p("results", "projection.json"), proj)
    write_json(run.p("results", "control_adjusted.json"), control_adjusted(run, res))
    # Sensitivity analyses: the same measures without no-coref passages, and without any
    # passage rendered by a fallback (sentence fallback or no-coref)
    nf = {}
    for title, fname, methods in (("without no-coreference passages", "sensitivity_no_coref.json", ("no_coref",)),
                                  ("without sentence-fallback and no-coreference passages",
                                   "sensitivity_no_fallback.json", ("sentence_fallback", "no_coref"))):
        run_x = Run(cfg=run.cfg, smoke=run.smoke, exclude_methods=methods)
        res_x = measure(run_x)
        nf[title] = {"res": res_x, "sig": signals(run_x, res_x), "acc": accuracy(run_x, res_x)}
        write_json(run.p("results", fname),
                   {"excluded_methods": list(methods), "signals": nf[title]["sig"], "accuracy": nf[title]["acc"],
                    "answer_preserved_share": res_x["answer_preserved_share"],
                    "retention_excluded": {k: v["excluded"] for k, v in res_x["retention"].items()}})
    write_report(run, res, sig, gts, acc, proj, nf)
    print(f"recommendation: {sig['recommendation']}; "
          + "; ".join(f"{t}: {v['sig']['recommendation']}" for t, v in nf.items())
          + f"; report -> {run.p('report', 'P0_report.md')}")
