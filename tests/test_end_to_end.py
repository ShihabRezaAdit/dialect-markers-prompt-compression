"""End-to-end dry run of every CPU step of P0 and the smoke run on synthetic data, with fake
renderings, compressions and answers (no GPU, no internet): job files, R control, prompts,
audit export, G1 job files, measurement, gates, signals, projection and report."""
import json
import re
import tempfile
from pathlib import Path

from idea86 import config as C
from idea86 import data as D
from idea86 import pilot as P
from idea86.survival import tokens


def fake_dialect(text, g):
    t = re.sub(r"\bis large\b", "be big", text) if g == "A" else re.sub(r"\bis long\b", "damn long", text)
    return t + (" lah" if g == "C" else "")


def half(text, offset=0):
    return " ".join(t.text for i, t in enumerate(tokens(text)) if (i + offset) % 2 == 0)


def make_cfg(tmp: Path):
    cfg = C.load("configs/p0.yaml")
    cfg["paths"] = {"data_dir": str(tmp / "data"), "runs_dir": str(tmp / "runs"), "prereg_dir": str(tmp / "prereg")}
    return cfg


def fake_render(r):
    P.step_render_jobs(r)
    jobs = C.read_jsonl(r.p("render", "jobs.jsonl"))
    # every 10th passage counts as rendered by the sentence fallback
    rows = [{**{k: j[k] for k in ("job_id", "kind", "pos", "k", "dialect")}, "ok": True,
             "text": fake_dialect(j["text"], j["dialect"]),
             "rules": [{"start": 9, "end_incl": 10, "value": "", "type": "zero_copula"}],
             "counter": {}, "seconds": 0.5,
             "method": "sentence_fallback" if j["kind"] == "passage" and j["pos"] % 10 == 0 else "whole"}
            for j in jobs]
    C.write_jsonl(r.p("render", "shards", "shard_000.jsonl"), rows)
    P.step_render_collect(r)


def pilot_run(tmp: Path):
    r = P.Run(cfg=make_cfg(tmp))
    recs = []
    for p in range(330):
        for j in range(3):
            recs.append({"id": f"q{p}_{j}", "title": f"T{p}",
                         "context": f"The city {p} is large and the river is long. It was founded in {1800 + p}.",
                         "question": f"When was city {p} founded?", "answers": {"text": [str(1800 + p)]}})
    smp = D.pilot_sample(D.squad_paragraphs(recs))
    smp["audit_pairs"] = D.audit_sample()["pairs"]
    smp["tweets"] = {"source": "REAL-AAE", "pairs": [
        {"pos": i, "row": i, "aae": f"he finna go to the store {i} fr", "sae": f"he is about to go to the store {i}"}
        for i in range(200)]}
    smp["checksums"] = {}
    C.write_json(r.p("items.json"), smp)
    fake_render(r)
    return r


def downstream(run):
    P.step_compress_jobs(run)
    for c in ("L1", "L2", "SC", "LL"):
        out = []
        for j in C.read_jsonl(run.p("compress", f"jobs_{c}.jsonl")):
            comp = half(j["text"], 1 if c == "L1" else 0)
            out.append({**{k: v for k, v in j.items() if k not in ("text", "question_text")},
                        "compressed": comp, "seconds": 0.1,
                        "achieved_keep_rate": len(tokens(comp)) / len(tokens(j["text"]))})
        C.write_jsonl(run.p("compress", f"{c}.jsonl"), out)
    P.step_random_control(run)
    P.step_prompts(run)
    prompts = C.read_jsonl(run.p("generate", "prompts.jsonl"))
    for m in ("M1", "M2"):
        rows = []
        for p in prompts:
            m_ = re.search(r"founded in (\d{4})", p["prompt"])
            ans = m_.group(1) if m_ else "unknown"
            rows.append({**{k: v for k, v in p.items() if k != "prompt"}, "model": m,
                         "prompt_hash": str(hash(p["prompt"])), "output": ans, "answer": ans,
                         "finish_reason": "stop", "length_stop": False})
        C.write_jsonl(run.p("generate", f"{m}.jsonl"), rows)
    P.step_g1_jobs(run)
    P.step_analyze(run)
    return prompts


def test_pilot_dry_run():
    with tempfile.TemporaryDirectory() as d:
        run = pilot_run(Path(d))
        assert len(C.read_jsonl(run.p("render", "jobs.jsonl"))) == 1800
        P.step_audit_export(run, "p0_audit")
        assert len(list(run.p("audit", "export", "p0_audit", "pairs").glob("*.html"))) == 100
        prompts = downstream(run)
        assert len(prompts) == 600 * 21
        sig = json.loads(run.p("results", "signals.json").read_text())
        gts = json.loads(run.p("results", "gates.json").read_text())
        assert set(sig["S1"]) == {f"{c}_{d_}" for c in ("L1", "L2", "SC") for d_ in ("A", "C")}
        assert all(v["n_paragraphs"] == 300 for v in sig["S1"].values())
        assert sig["recommendation"] in ("Go", "Go, retention focus", "Go, realistic-pipeline focus", "No-go")
        assert {"G1", "G2", "G3", "G4", "G5"} <= set(gts) and gts["G4"]["units"] == 600
        rep = run.p("report", "P0_report.md").read_text()
        assert "Recommended outcome" in rep and "partial words counted as kept" in rep
        assert "Rendering: failures and sentence fallback" in rep and "without sentence-fallback passages" in rep
        summ = json.loads(run.p("render", "summary.json").read_text())
        assert summ["passage_A"]["sentence_fallback"] == 30 and summ["passage_A"]["fallback_share"] == 0.1
        nf = json.loads(run.p("results", "sensitivity_no_fallback.json").read_text())
        assert all(v["n_paragraphs"] == 270 for v in nf["signals"]["S1"].values())
        meas = json.loads(run.p("results", "measure.json").read_text())
        assert set(meas["retention_partial_kept"]) == {f"{c}_{d_}" for c in ("L1", "L2", "SC", "R") for d_ in ("A", "C")}


def test_smoke_dry_run():
    with tempfile.TemporaryDirectory() as d:
        run = pilot_run(Path(d))
        sm = P.Run(cfg=run.cfg, smoke=True)
        P.step_smoke_items(run, sm)
        fake_render(sm)
        jobs = C.read_jsonl(sm.p("render", "jobs.jsonl"))
        assert len(jobs) == 60 and min(j["k"] for j in jobs) >= 500000
        P.step_audit_export(sm, "warmup_audit")
        assert len(list(sm.p("audit", "export", "warmup_audit", "pairs").glob("*.html"))) == 5
        assert len(downstream(sm)) == 20 * 21
        assert sm.p("report", "P0_report.md").exists()
