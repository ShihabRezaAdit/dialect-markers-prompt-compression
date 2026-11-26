"""Main-study driver, must-have tier (internal plan Sections 15, 16 and 19; advisor's priorities):
E1 rendering of SQuAD validation into AAVE and the 150-pair audit with 20 re-labeled pairs, and E2
retention on SQuAD AAVE and the 17,279 non-pilot REAL-AAE pairs at three keep rates with L1, L2,
SC and R. The mechanism analysis E5a is in mechanism.py.

Outputs go to runs/E1, runs/E2 and runs/E5. Nothing here reads pilot files (isolation.py).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from . import stats as S
from . import survival as sv
from .audit import columns_for, dialect_names, export as audit_export
from .config import Config, now, read_json, read_jsonl, sha256_text, write_json, write_jsonl
from .isolation import check_config
from .summaries import keep_by_length, marker_summary, rendering_summary

DIALECT = "A"                       # confirmatory dialect (Indian English only under X1)
LEARNED = ("L1", "L2", "SC")
ALL_C = ("L1", "L2", "SC", "R")
CORPORA = ("squad", "tweets")
TWEET_G = {"S": "SAE", "D": "AAE"}


class Study:
    def __init__(self, cfg: Config):
        check_config(cfg)
        self.cfg = cfg
        self.root = cfg.path("runs_dir")

    def p(self, exp: str, *parts) -> Path:
        return self.root.joinpath(exp, *parts)

    def items(self) -> dict:
        f = self.p("E1", "items.json")
        if not f.exists():
            raise SystemExit(f"{f} not found: run ./run_main.sh items first")
        return read_json(f)

    def rendered(self, exclude_methods: tuple = ()) -> dict:
        r = read_json(self.p("E1", "render", "rendered.json"))
        if exclude_methods:
            r = {k: ({**v, "ok": False} if v.get("method") in exclude_methods else v) for k, v in r.items()}
        return r


def usable(rendered: dict, ppos: int, d: str = DIALECT) -> str | None:
    r = rendered.get(f"passage:{ppos}:{d}")
    return r["text"] if r and r.get("ok") else None


# ---------------------------------------------------------------- E1: items

def build_items(paras: list[dict], tweet_rows: list[dict], study_idx: list[int], cfg) -> dict:
    """SQuAD validation paragraphs (pos = paragraph number) and questions (pos = split order), the
    REAL-AAE study pairs, and the audit draw: one default_rng(86) generator gives the paragraph
    permutation, a uniform number per paragraph that picks its question, and the 20 re-label
    positions among the 150 audited pairs."""
    passages, questions = [], []
    for p in paras:
        qpos = []
        for q in p["questions"]:
            qpos.append(len(questions))
            questions.append({"pos": len(questions), "ppos": p["pnum"], "qid": q["qid"],
                              "question": q["question"], "answers": q["answers"]})
        passages.append({"pos": p["pnum"], "title": p["title"], "context": p["context"], "qpos": qpos})
    cols = cfg["data"]["real_aae"]["text_columns"]
    tweets = [{"pos": i, "row": r, "aae": tweet_rows[r][cols[0]], "sae": tweet_rows[r][cols[1]]}
              for i, r in enumerate(study_idx)]
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(passages)).tolist()
    qpick = rng.random(len(passages)).tolist()
    n_audit, n_rel = cfg["sample"]["audit_pairs"], cfg["sample"]["relabel_pairs"]
    relabel = sorted(rng.choice(n_audit, size=n_rel, replace=False).tolist())
    audit = {"order": order, "qpick": qpick, "relabel_positions": relabel}
    items = {"passages": passages, "questions": questions, "tweets": tweets, "audit": audit}
    items["checksums"] = {
        "passages": sha256_text(json.dumps([p["pos"] for p in passages])),
        "questions": sha256_text(json.dumps([q["qid"] for q in questions])),
        "tweet_rows": sha256_text(json.dumps(study_idx)),
        "audit": sha256_text(json.dumps(audit)),
    }
    return items


def step_items(st: Study) -> None:
    from .data import load_squad, real_aae_rows_main

    cfg = st.cfg
    paras = load_squad(cfg)
    rows, idx = real_aae_rows_main(cfg)
    items = build_items(paras, rows, idx, cfg)
    write_json(st.p("E1", "items.json"), items)
    print(f"items: {len(items['passages'])} paragraphs, {len(items['questions'])} questions, "
          f"{len(items['tweets'])} REAL-AAE study pairs (planned 2,067, 10,570, 17,279)")


# ---------------------------------------------------------------- E1: rendering

def build_render_jobs(items: dict, cfg) -> list[dict]:
    off = cfg["render"]["k_offsets"]
    jobs = []
    for d, cls in cfg["render"]["dialects"].items():
        for p in items["passages"]:
            jobs.append({"job_id": f"passage:{p['pos']}:{d}", "kind": "passage", "pos": p["pos"],
                         "k": off["squad_passage"] + p["pos"], "dialect": d, "dialect_class": cls, "text": p["context"]})
        for q in items["questions"]:
            jobs.append({"job_id": f"question:{q['pos']}:{d}", "kind": "question", "pos": q["pos"],
                         "k": off["squad_question"] + q["pos"], "dialect": d, "dialect_class": cls, "text": q["question"]})
    return jobs


def step_render_jobs(st: Study) -> None:
    jobs = build_render_jobs(st.items(), st.cfg)
    write_jsonl(st.p("E1", "render", "jobs.jsonl"), jobs)
    print(f"render jobs: {len(jobs)} (planned 12,637 = 2,067 passages + 10,570 questions in AAVE)")


def step_render_collect(st: Study) -> None:
    rendered = {}
    for f in sorted(st.p("E1", "render", "shards").glob("shard_*.jsonl")):
        for r in read_jsonl(f):
            rendered[r["job_id"]] = r
    jobs = read_jsonl(st.p("E1", "render", "jobs.jsonl"))
    missing = [j["job_id"] for j in jobs if j["job_id"] not in rendered]
    if missing:
        raise SystemExit(f"{len(missing)} render jobs missing, e.g. {missing[:3]}")
    write_json(st.p("E1", "render", "rendered.json"), rendered)
    summ = rendering_summary(rendered)
    mk = marker_summary(st.items()["passages"], rendered, st.cfg, tuple(st.cfg["render"]["dialects"]))
    write_json(st.p("E1", "render", "summary.json"), summ)
    write_json(st.p("E1", "render", "markers.json"), mk)
    print(f"rendered {len(rendered)}; not usable {sum(not r['ok'] for r in rendered.values())}")
    for k, v in summ.items():
        print(f"  {k}: {v['n']} texts, whole {v['whole']}, sentence fallback {v['sentence_fallback']}, "
              f"no-coref {v['no_coref']} ({v['no_coref_sentences']} sentences), too long {v['too_long']}, "
              f"failed {v['failed']}; quote-protected {v['quote_protected']}")
    for d, v in mk.items():
        print(f"  markers {d} ({v['dialect_class']}): {v['passages']} passages, mean {v['mean']:.1f}, median "
              f"{v['median']:.0f}; >= 3: {v['share_ge3']:.2f}; {len(v['types'])} feature types fired")
    bad = {d: v["disabled_fired"] for d, v in mk.items() if v["disabled_fired"]}
    if bad:
        raise SystemExit(f"switched-off features appear in the rule logs: {bad}")


# ---------------------------------------------------------------- E1: audit

def audit_pairs(items: dict, rendered: dict, cfg) -> dict:
    """Walk the seed-86 paragraph permutation; a paragraph whose AAVE rendering is usable becomes a
    pair (with the question its uniform number picks) until 150 pairs; unusable ones are skipped and
    listed. The re-label pairs are the pairs at the drawn positions."""
    a = items["audit"]
    pairs, skipped = [], []
    for ppos in a["order"]:
        if len(pairs) == cfg["sample"]["audit_pairs"]:
            break
        dia = usable(rendered, ppos)
        if dia is None:
            skipped.append({"ppos": ppos, "method": (rendered.get(f"passage:{ppos}:{DIALECT}") or {}).get("method")})
            continue
        p = items["passages"][ppos]
        q = items["questions"][p["qpos"][int(a["qpick"][ppos] * len(p["qpos"]))]]
        pairs.append({"item_key": f"passage:{ppos}:{DIALECT}", "ppos": ppos, "qpos": q["pos"], "dialect": DIALECT,
                      "sae": p["context"], "dia": dia, "question": q["question"], "golds": q["answers"]})
    if len(pairs) < cfg["sample"]["audit_pairs"]:
        raise SystemExit("not enough usable paragraphs for the audit")
    relabel = [pairs[i] for i in a["relabel_positions"]]
    return {"pairs": pairs, "relabel": relabel, "skipped": skipped}


def step_audit_export(st: Study, sheet: str) -> None:
    cfg = st.cfg
    ap = audit_pairs(st.items(), st.rendered(), cfg)
    names = dialect_names(cfg)
    if sheet == "e1_audit":
        pairs, seed = ap["pairs"], cfg["seed"]
        write_json(st.p("E1", "audit", "audit_pairs.json"),
                   {k: [{x: y for x, y in p.items() if x in ("item_key", "ppos", "qpos")} for p in v] if k != "skipped" else v
                    for k, v in ap.items()})
    elif sheet == "e1_relabel":
        imp = st.p("E1", "audit", "e1_audit_import.json")
        if not imp.exists():
            raise SystemExit("import the e1_audit labels first; the re-label sheet comes at least 7 days later")
        t0 = dt.datetime.strptime(read_json(imp)["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        days = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds() / 86400
        if days < cfg["sample"]["relabel_min_days"]:
            raise SystemExit(f"only {days:.1f} days since the e1_audit import; the re-label needs at least "
                             f"{cfg['sample']['relabel_min_days']}")
        pairs, seed = ap["relabel"], cfg["sample"]["relabel_order_seed"]
    else:
        raise SystemExit("main-study sheets: e1_audit, e1_relabel")
    audit_export(sheet, pairs, st.p("E1", "audit", "export", sheet), st.p("E1", "audit", "keys"),
                 order_seed=seed, names=names)
    print(f"exported {len(pairs)} pairs ({len(ap['skipped'])} paragraphs skipped as not usable) -> "
          f"{st.p('E1', 'audit', 'export', sheet)}")


def step_audit_import(st: Study, sheet: str) -> None:
    from .audit import import_labels

    merged, errs = import_labels(st.p("E1", "audit", f"{sheet}.xlsx"), st.p("E1", "audit", "keys", f"{sheet}_keys.json"),
                                 columns_for(sheet))
    if errs:
        print(f"{len(errs)} problem(s); nothing imported. Fix these rows and upload the sheet again:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    write_jsonl(st.p("E1", "audit", f"labels_{sheet}.jsonl"), merged)
    write_json(st.p("E1", "audit", f"{sheet}_import.json"), {"utc": now(), "rows": len(merged)})
    print(f"[audit] imported {len(merged)} rows -> {st.p('E1', 'audit', f'labels_{sheet}.jsonl')}")


def audit_report(labels: list[dict], relabels: list[dict] | None = None) -> dict:
    """Shares with Wilson 95% intervals, counts under the applied and the strict D5 reading (every
    pair with TIME_HABIT_SHIFT = YES counted as a content change), fluency counts, and Cohen's kappa
    for CONTENT_PRESERVED on the re-labeled pairs."""
    n = len(labels)
    yes = sum(r["CONTENT_PRESERVED"] == "YES" for r in labels)
    strict = sum(r["CONTENT_PRESERVED"] == "YES" and r.get("TIME_HABIT_SHIFT", "NO") != "YES" for r in labels)
    ans = sum(r["ANSWER_PRESERVED"] == "YES" for r in labels)
    out = {"n": n, "content_yes": yes, "content_yes_wilson95": S.wilson(yes, n),
           "content_yes_strict_d5": strict, "content_yes_strict_wilson95": S.wilson(strict, n),
           "answer_yes": ans, "answer_yes_wilson95": S.wilson(ans, n),
           "fluency": {f: sum(r["FLUENCY"] == f for r in labels) for f in ("NONE", "MINOR", "SEVERE")},
           "time_habit_shift_yes": sum(r.get("TIME_HABIT_SHIFT") == "YES" for r in labels)}
    if relabels:
        first = {r["item_key"]: r["CONTENT_PRESERVED"] for r in labels}
        a = [first[r["item_key"]] for r in relabels]
        b = [r["CONTENT_PRESERVED"] for r in relabels]
        out["relabel"] = {"n": len(b), "agreement": float(np.mean([x == y for x, y in zip(a, b)])),
                          "kappa": S.cohen_kappa(a, b)}
    return out


def step_audit_report(st: Study) -> None:
    lab = read_jsonl(st.p("E1", "audit", "labels_e1_audit.jsonl"))
    if not lab:
        raise SystemExit("labels_e1_audit.jsonl not imported yet")
    rel = read_jsonl(st.p("E1", "audit", "labels_e1_relabel.jsonl")) or None
    rep = audit_report(lab, rel)
    write_json(st.p("E1", "results", "audit.json"), rep)
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- E2: compression

def text_units(items: dict, rendered: dict, cfg) -> list[dict]:
    """Every text E2 compresses: SQuAD SAE passages (all) and AAVE passages (usable), and the AAE and
    SAE sides of the REAL-AAE study pairs. `k_text` seeds the random control."""
    off = cfg["render"]["k_offsets"]
    out = []
    for p in items["passages"]:
        out.append({"corpus": "squad", "pos": p["pos"], "rendering": "S", "text": p["context"], "k_text": off["squad_passage"] + p["pos"]})
        dia = usable(rendered, p["pos"])
        if dia is not None:
            out.append({"corpus": "squad", "pos": p["pos"], "rendering": "A", "text": dia, "k_text": off["squad_passage"] + p["pos"]})
    for t in items["tweets"]:
        for g, field in (("SAE", "sae"), ("AAE", "aae")):
            out.append({"corpus": "tweets", "pos": t["pos"], "rendering": g, "text": t[field], "k_text": off["tweet_row"] + t["row"]})
    return out


def job_id(u: dict, c: str, rho: float) -> str:
    return f"{u['corpus']}:{u['pos']}:{u['rendering']}:{c}:{rho}"


def build_compress_jobs(items: dict, rendered: dict, cfg) -> dict[str, list[dict]]:
    units = text_units(items, rendered, cfg)
    jobs = {c: [] for c in cfg["compress"]["compressors"]}
    for c in jobs:
        for rho in cfg["compress"]["rates"]:
            for u in units:
                jobs[c].append({**u, "job_id": job_id(u, c, rho), "compressor": c, "rho": rho})
    return jobs


def step_compress_jobs(st: Study) -> None:
    jobs = build_compress_jobs(st.items(), st.rendered(), st.cfg)
    for c, js in jobs.items():
        write_jsonl(st.p("E2", "compress", f"jobs_{c}.jsonl"), js)
        print(f"{c}: {len(js)} jobs ({len(st.cfg['compress']['rates'])} keep rates)")


def shard(jobs: list, i: int, n: int) -> list:
    lo, hi = i * len(jobs) // n, (i + 1) * len(jobs) // n
    return jobs[lo:hi]


def step_compress_collect(st: Study) -> None:
    for c in st.cfg["compress"]["compressors"]:
        jobs = read_jsonl(st.p("E2", "compress", f"jobs_{c}.jsonl"))
        rows = {}
        for f in sorted(st.p("E2", "compress", c).glob("shard_*.jsonl")):
            for r in read_jsonl(f):
                rows[r["job_id"]] = r
        missing = [j["job_id"] for j in jobs if j["job_id"] not in rows]
        if missing:
            raise SystemExit(f"{c}: {len(missing)} compressions missing, e.g. {missing[:3]}; re-run that shard")
        write_jsonl(st.p("E2", "compress", f"{c}.jsonl"), [rows[j["job_id"]] for j in jobs])
        print(f"{c}: {len(rows)} compressions collected; excluded {sum(1 for r in rows.values() if r.get('excluded'))}")


def step_random_control(st: Study) -> None:
    from .compress import random_control

    l2 = {r["job_id"]: r for r in read_jsonl(st.p("E2", "compress", "L2.jsonl"))}
    if not l2:
        raise SystemExit("L2.jsonl missing: run compress-collect first")
    jobs = []
    for j in read_jsonl(st.p("E2", "compress", "jobs_L2.jsonl")):
        jobs.append({**j, "job_id": j["job_id"].replace(":L2:", ":R:"), "compressor": "R", "match_key": j["job_id"]})
    rows = random_control(jobs, l2)
    write_jsonl(st.p("E2", "compress", "R.jsonl"), rows)
    print(f"R: {len(rows)} texts, matched to L2")


def load_compressed(st: Study) -> dict:
    out = {}
    for c in ALL_C:
        for r in read_jsonl(st.p("E2", "compress", f"{c}.jsonl")):
            out[r["job_id"]] = r
    return out


# ---------------------------------------------------------------- E2: measurement

def pair_texts(items: dict, rendered: dict, corpus: str):
    """(pos, standard text, dialect text) for every usable pair of a corpus."""
    if corpus == "squad":
        for p in items["passages"]:
            dia = usable(rendered, p["pos"])
            if dia is not None:
                yield p["pos"], p["context"], dia
    else:
        for t in items["tweets"]:
            yield t["pos"], t["sae"], t["aae"]


def comp_of(comp: dict, corpus: str, pos: int, g: str, c: str, rho: float):
    r = comp.get(f"{corpus}:{pos}:{g}:{c}:{rho}")
    return r.get("compressed") if r else None


def retention(items, rendered, comp, corpus, c, rho, partial_as_kept=False) -> dict[int, float]:
    gS, gD = ("S", "A") if corpus == "squad" else ("SAE", "AAE")
    vals = {}
    for pos, sae, dia in pair_texts(items, rendered, corpus):
        cs, cd = comp_of(comp, corpus, pos, gS, c, rho), comp_of(comp, corpus, pos, gD, c, rho)
        if cs is None or cd is None:
            continue
        v = M.delta_s_text(sae, dia, cs, cd, partial_as_kept)
        if v is not None:
            vals[pos] = v
    return vals


def ratio_cell(items, rendered, comp, corpus, c, rho, B, seed) -> dict:
    gD = "A" if corpus == "squad" else "AAE"
    pairs = list(pair_texts(items, rendered, corpus))
    fs = sv.FreqStrata([s for _, s, _ in pairs])
    cnts = []
    for pos, sae, dia in pairs:
        cd = comp_of(comp, corpus, pos, gD, c, rho)
        if cd is not None:
            cnts.append(M.ratio_counts_text(sae, dia, cd, fs))
    if not cnts:
        return {"r": None}
    est, draws = S.bootstrap(M.ratio_stat(np.array(cnts)), len(cnts), B, seed)
    return {"r": est, "ci95": S.percentile_ci(draws, 0.95), "n": len(cnts)}


def cell(vals: dict, B: int, seed: int, margin: float) -> dict:
    v = np.array([vals[k] for k in sorted(vals)], dtype=float)
    est, draws = S.bootstrap(S.mean_stat(v), len(v), B, seed)
    eq, ci90 = S.tost_equivalent(draws, margin)
    return {"estimate": est, "ci95": S.percentile_ci(draws, 0.95), "ci90": ci90, "p": S.bootstrap_p(draws),
            "equivalent": eq, "n": len(v), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "share_positive": float((v > 0).mean()), "share_negative": float((v < 0).mean())}


def family(cells: dict, names: tuple, alpha: float) -> dict:
    h = S.holm({k: cells[k]["p"] for k in names}, alpha)
    return {k: {**cells[k], "holm": h[k]} for k in names}


def measure_e2(st: Study, exclude_methods: tuple = ()) -> dict:
    cfg, sc = st.cfg, st.cfg["stats"]
    B, seed, margin, alpha = sc["bootstrap_resamples"], sc["bootstrap_seed"], sc["equivalence_margin_delta_s"], sc["alpha"]
    items, rendered, comp = st.items(), st.rendered(exclude_methods), load_compressed(st)
    rho0 = cfg["compress"]["confirmatory_rate"]
    res = {"cells": {}, "ratio": {}, "keep_rates": {}, "partial_kept": {}}
    for corpus in CORPORA:
        for rho in cfg["compress"]["rates"]:
            for c in ALL_C:
                key = f"{corpus}|{c}|{rho}"
                vals = retention(items, rendered, comp, corpus, c, rho)
                res["cells"][key] = cell(vals, B, seed, margin) if vals else {"estimate": None, "n": 0}
                if rho == rho0:
                    pk = retention(items, rendered, comp, corpus, c, rho, partial_as_kept=True)
                    res["partial_kept"][key] = cell(pk, B, seed, margin) if pk else None
                    res["ratio"][key] = ratio_cell(items, rendered, comp, corpus, c, rho, B, seed)
    # confirmatory family F1 (SQuAD AAVE at the confirmatory rate) and the REAL-AAE replication family
    f1 = {c: res["cells"][f"squad|{c}|{rho0}"] for c in LEARNED}
    rep = {c: res["cells"][f"tweets|{c}|{rho0}"] for c in LEARNED}
    res["F1"] = family(f1, LEARNED, alpha) if all(v.get("n") for v in f1.values()) else None
    res["replication_real_aae"] = family(rep, LEARNED, alpha) if all(v.get("n") for v in rep.values()) else None
    rc = res["cells"][f"squad|R|{rho0}"]
    if rc.get("n"):
        lo, hi = rc["ci90"]
        res["H1c"] = {"estimate": rc["estimate"], "ci90": rc["ci90"], "supported": bool(-margin < lo and hi < margin)}
    if res["F1"]:
        res["H1a"] = {"supported": bool(res["F1"]["L1"]["holm"]["reject"] and res["F1"]["L1"]["estimate"] > 0)}
        res["H1b"] = {"supported": bool(res["F1"]["L2"]["holm"]["reject"]), "sign": np.sign(res["F1"]["L2"]["estimate"]).item()}
        res["H1d"] = {"supported": bool(res["F1"]["SC"]["holm"]["reject"]), "sign": np.sign(res["F1"]["SC"]["estimate"]).item()}
    # achieved keep rates per corpus, rendering, compressor and rate
    acc: dict = {}
    for r in comp.values():
        if r.get("achieved_keep_rate") is not None:
            acc.setdefault(f"{r['corpus']}|{r['rendering']}|{r['compressor']}|{r['rho']}", []).append(r["achieved_keep_rate"])
    res["keep_rates"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(acc.items())}
    res["keep_by_length"] = keep_by_length(comp)
    res["by_tweet_length"] = by_tweet_length(items, comp, rho0, B, seed, margin)
    res["by_feature_type"] = by_feature_type(items, rendered, comp, rho0)
    res["by_lexicon_class"] = by_lexicon_class(items, comp, rho0)
    res["excluded_methods"] = list(exclude_methods)
    return res


def by_tweet_length(items, comp, rho, B, seed, margin) -> dict:
    n_words = {t["pos"]: len(sv.tokens(t["aae"])) for t in items["tweets"]}
    med = float(np.median(list(n_words.values())))
    out = {"median_aae_words": med}
    for c in ALL_C:
        vals = retention(items, {}, comp, "tweets", c, rho)
        for name, keep in (("short", lambda n: n <= med), ("long", lambda n: n > med)):
            sub = {k: v for k, v in vals.items() if keep(n_words[k])}
            out[f"{c}|{name}"] = cell(sub, B, seed, margin) if sub else None
    return out


def by_feature_type(items, rendered, comp, rho, top: int = 15) -> dict:
    """Mean s_D - s_S over replace ops by primary feature type (SQuAD AAVE), per compressor."""
    acc: dict = {}
    for pos, sae, dia in pair_texts(items, rendered, "squad"):
        rr = rendered.get(f"passage:{pos}:{DIALECT}")
        st_, dt_ = sv.tokens(sae), sv.tokens(dia)
        ops = [o for o in sv.diff_ops(st_, dt_) if o.tag == "replace"]
        types = [sv.type_op(o, st_, len(sae), rr.get("rules", [])) for o in ops]
        for c in ALL_C:
            cs, cd = comp_of(comp, "squad", pos, "S", c, rho), comp_of(comp, "squad", pos, "A", c, rho)
            if cs is None or cd is None:
                continue
            ks = sv.survived(sv.word_survival(st_, sv.tokens(cs)))
            kd = sv.survived(sv.word_survival(dt_, sv.tokens(cd)))
            for o, t in zip(ops, types):
                acc.setdefault(t, {}).setdefault(c, []).append(kd[o.j1:o.j2].mean() - ks[o.i1:o.i2].mean())
    ranked = sorted(acc, key=lambda t: -len(acc[t].get("L1", [])))[:top]
    return {t: {c: {"mean": float(np.mean(v)), "ops": len(v)} for c, v in acc[t].items()} for t in ranked}


def by_lexicon_class(items, comp, rho) -> dict:
    """REAL-AAE: mean s_D - s_S over replace ops by lexicon class (Wu et al. 2026, Table 8; lexicon.py),
    per compressor, with the number of ops per class."""
    from .lexicon import matches, type_op

    acc: dict = {}
    counts: dict = {}
    for t in items["tweets"]:
        st_, dt_ = sv.tokens(t["sae"]), sv.tokens(t["aae"])
        ops = [o for o in sv.diff_ops(st_, dt_) if o.tag == "replace"]
        found = matches(dt_)
        types = [type_op(o, found) for o in ops]
        for ty in types:
            counts[ty] = counts.get(ty, 0) + 1
        for c in ALL_C:
            cs, cd = comp_of(comp, "tweets", t["pos"], "SAE", c, rho), comp_of(comp, "tweets", t["pos"], "AAE", c, rho)
            if cs is None or cd is None:
                continue
            ks = sv.survived(sv.word_survival(st_, sv.tokens(cs)))
            kd = sv.survived(sv.word_survival(dt_, sv.tokens(cd)))
            for o, ty in zip(ops, types):
                acc.setdefault(ty, {}).setdefault(c, []).append(kd[o.j1:o.j2].mean() - ks[o.i1:o.i2].mean())
    return {"ops": counts, "classes": {ty: {c: {"mean": float(np.mean(v)), "ops": len(v)} for c, v in d.items()} for ty, d in acc.items()}}


def step_analyze_e2(st: Study) -> None:
    res = measure_e2(st)
    write_json(st.p("E2", "results", "retention.json"), res)
    for title, methods in (("no_coref", ("no_coref",)), ("no_fallback", ("sentence_fallback", "no_coref"))):
        r2 = measure_e2(st, methods)
        write_json(st.p("E2", "results", f"sensitivity_{title}.json"), {k: r2[k] for k in ("F1", "cells", "excluded_methods")})
    write_report_e2(st, res)
    rho0 = st.cfg["compress"]["confirmatory_rate"]
    for c in ALL_C:
        v = res["cells"][f"squad|{c}|{rho0}"]
        if v.get("n"):
            print(f"SQuAD AAVE {c} at {rho0}: {v['estimate']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] n={v['n']}")
    print(f"report -> {st.p('E2', 'report', 'E2_report.md')}")


def write_report_e2(st: Study, res: dict) -> None:
    rho0 = st.cfg["compress"]["confirmatory_rate"]
    L = ["# E2 retention report (main study)", "", f"Config SHA-256: `{st.cfg.hash}`.", "",
         f"## Confirmatory family F1: SQuAD AAVE at keep rate {rho0} (Holm, three tests)", "",
         "| Compressor | Delta_s | 95% interval | p | Holm reject | Equivalent (+-0.03) | n |", "|---|---|---|---|---|---|---|"]
    for c, v in (res.get("F1") or {}).items():
        L.append(f"| {c} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['p']:.4f} | "
                 f"{v['holm']['reject']} | {v['equivalent']} | {v['n']} |")
    for h in ("H1a", "H1b", "H1d", "H1c"):
        if h in res:
            L.append(f"- {h}: {json.dumps(res[h])}")
    L += ["", "## Replication: REAL-AAE (Holm, three tests)", "",
          "| Compressor | Delta_s | 95% interval | p | Holm reject | n |", "|---|---|---|---|---|---|"]
    for c, v in (res.get("replication_real_aae") or {}).items():
        L.append(f"| {c} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['p']:.4f} | {v['holm']['reject']} | {v['n']} |")
    L += ["", "## All cells (dose-response; exploratory except the families above)", "",
          "| Corpus | Compressor | Rate | Delta_s | 95% interval | n | share > 0 / < 0 |", "|---|---|---|---|---|---|---|"]
    for k, v in res["cells"].items():
        corpus, c, rho = k.split("|")
        if v.get("n"):
            L.append(f"| {corpus} | {c} | {rho} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n']} | "
                     f"{v['share_positive']:.2f} / {v['share_negative']:.2f} |")
    L += ["", "## Retention ratio r at the confirmatory rate", "", "| Cell | r | 95% interval | n |", "|---|---|---|---|"]
    for k, v in res["ratio"].items():
        if v.get("r") is not None:
            L.append(f"| {k} | {v['r']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['n']} |")
    L += ["", "## REAL-AAE by tweet length (exploratory)", "", f"Median AAE words: {res['by_tweet_length']['median_aae_words']}", "",
          "| Compressor and group | Delta_s | 95% interval | n |", "|---|---|---|---|"]
    for k, v in res["by_tweet_length"].items():
        if isinstance(v, dict) and v.get("n"):
            L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n']} |")
    L += ["", "## Achieved keep rates", "", "| Corpus, rendering, compressor, rate | Mean | n |", "|---|---|---|"]
    for k, v in res["keep_rates"].items():
        L.append(f"| {k} | {v['mean']:.3f} | {v['n']} |")
    L += ["", "## REAL-AAE survival difference by lexicon class (Wu et al. 2026, Table 8; exploratory)", "",
          "| Class | Ops | " + " | ".join(ALL_C) + " |", "|---|---|" + "---|" * len(ALL_C)]
    lc = res.get("by_lexicon_class") or {"ops": {}, "classes": {}}
    for ty, n in sorted(lc["ops"].items(), key=lambda x: -x[1]):
        cells = [f"{lc['classes'].get(ty, {}).get(c, {}).get('mean', float('nan')):.3f}" for c in ALL_C]
        L.append(f"| {ty} | {n} | " + " | ".join(cells) + " |")
    L += ["", "## Survival difference by feature type (SQuAD AAVE, exploratory)", "", "```",
          json.dumps(res["by_feature_type"], indent=1), "```"]
    path = st.p("E2", "report", "E2_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
