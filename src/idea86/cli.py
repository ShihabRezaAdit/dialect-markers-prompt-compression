"""Command line: python -m idea86 <command> [--config configs/p0.yaml] [--smoke]

Pilot order (README, RUNBOOK_ARCC2.md):
  download -> warmup-render / warmup-compress -> items -> test -> smoke run (smoke-items, then
  render, compress, random-control, prompts, generate, g1-jobs, analyze with --smoke)
  -> precommit -> render -> audit-export -> compress -> random-control -> prompts -> generate
  -> g1-jobs -> G1 re-runs -> (you label) -> audit-import -> analyze
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import config as C


def _cmd_test(cfg) -> None:
    r = subprocess.run([sys.executable, str(C.ROOT / "tests" / "run_tests.py")], cwd=C.ROOT)
    C.write_json(cfg.run_dir() / "unit_tests.json", {"passed": r.returncode == 0, "code_hash": C.code_hash(),
                                                     "git_commit": C.git_commit(), "utc": C.now()})
    if r.returncode:
        raise SystemExit("unit tests failed")


def _status(run) -> None:
    r = run.root
    cfg = run.cfg
    checks = [("data/revisions.json", cfg.path("data_dir", "revisions.json")),
              ("items.json", r / "items.json"), ("unit_tests.json", cfg.run_dir() / "unit_tests.json"),
              ("prereg/P0_precommit.md", cfg.path("prereg_dir", "P0_precommit.md")),
              ("render/jobs.jsonl", r / "render" / "jobs.jsonl"), ("render/rendered.json", r / "render" / "rendered.json")]
    checks += [(f"compress/{c}.jsonl", r / "compress" / f"{c}.jsonl") for c in ("L1", "L2", "SC", "LL", "R")]
    checks += [("generate/prompts.jsonl", r / "generate" / "prompts.jsonl")]
    checks += [(f"generate/{m}.jsonl", r / "generate" / f"{m}.jsonl") for m in cfg["generate"]["models"]]
    checks += [("audit/labels_p0.jsonl", r / "audit" / "labels_p0.jsonl"),
               ("audit/labels_p0_audit2.jsonl", r / "audit" / "labels_p0_audit2.jsonl"),
               ("results/signals.json", r / "results" / "signals.json"),
               ("report/P0_report.md", r / "report" / "P0_report.md")]
    for name, p in checks:
        print(f"  [{'x' if p.exists() else ' '}] {name}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="idea86", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[
        "download", "warmup-render", "warmup-compress", "items", "smoke-items", "test", "precommit",
        "render-jobs", "render", "render-collect", "audit-export", "audit-import", "compress-jobs",
        "compress", "random-control", "prompts", "generate", "g1-jobs", "analyze", "status"])
    ap.add_argument("--config", default="configs/p0.yaml")
    ap.add_argument("--smoke", action="store_true", help="operate on the smoke run (runs/P0_smoke)")
    ap.add_argument("--g1", action="store_true", help="render/compress/generate: the G1 determinism re-run")
    ap.add_argument("--task", type=int, help="render: array task index (default SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--ntasks", type=int, help="render: number of array tasks (default from config)")
    ap.add_argument("--compressor", choices=["L1", "L2", "LL", "SC"])
    ap.add_argument("--model", help="generate: model key (M1 or M2)")
    ap.add_argument("--sheet", choices=["warmup_audit", "p0_audit", "p0_audit2"])
    ap.add_argument("--force", action="store_true", help="precommit: overwrite (must be logged)")
    a = ap.parse_args(argv)
    cfg = C.load(a.config)

    if a.command == "download":
        from .data import download
        download(cfg)
        return
    if a.command == "test":
        _cmd_test(cfg)
        return
    if a.command == "precommit":
        from .precommit import write
        write(cfg, force=a.force)
        print("pre-commitment written to prereg/P0_precommit.md; commit and push it before rendering pilot items")
        return
    if a.command == "warmup-render":
        from .render import warmup
        warmup(cfg)
        return
    if a.command == "warmup-compress":
        import nltk
        for pkg in ("punkt", "punkt_tab"):
            nltk.download(pkg, download_dir=os.environ.get("NLTK_DATA"), quiet=True)
        print("[warmup] NLTK punkt data ready")
        return

    from . import pilot as P
    run = P.Run(cfg=cfg, smoke=a.smoke)

    def guard():
        if not a.smoke:
            from .precommit import verify
            verify(cfg)

    if a.command == "items":
        P.step_items(run)
    elif a.command == "smoke-items":
        P.step_smoke_items(P.Run(cfg=cfg), P.Run(cfg=cfg, smoke=True))
    elif a.command == "render-jobs":
        guard()
        P.step_render_jobs(run)
    elif a.command == "render":
        guard()
        from .render import run_shard
        task = a.task if a.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        ntasks = a.ntasks or cfg["render"]["array_tasks"]
        src = run.p("g1", "render_jobs.jsonl") if a.g1 else run.p("render", "jobs.jsonl")
        out = run.p("g1", "render_shards") if a.g1 else run.p("render", "shards")
        rows = run_shard(C.read_jsonl(src), task, ntasks, cfg["render"])
        C.write_jsonl(out / f"shard_{task:03d}.jsonl", rows)
        print(f"[render] task {task}/{ntasks}: {len(rows)} texts, {sum(not r['ok'] for r in rows)} failed")
    elif a.command == "render-collect":
        P.step_render_collect(run)
    elif a.command == "audit-export":
        if not a.sheet:
            ap.error("--sheet is required")
        P.step_audit_export(run, a.sheet)
    elif a.command == "audit-import":
        from .audit import import_labels
        sheet = a.sheet or "p0_audit"
        merged, errs = import_labels(run.p("audit", f"{sheet}.xlsx"), run.p("audit", "keys", f"{sheet}_keys.json"))
        if errs:
            print(f"{len(errs)} problem(s); nothing imported. Fix these rows and upload the sheet again:")
            for e in errs:
                print("  -", e)
            raise SystemExit(1)
        out = run.p("audit", "labels_p0.jsonl" if sheet == "p0_audit" else f"labels_{sheet}.jsonl")
        C.write_jsonl(out, merged)
        print(f"[audit] imported {len(merged)} rows -> {out}")
    elif a.command == "compress-jobs":
        P.step_compress_jobs(run)
    elif a.command == "compress":
        guard()
        if not a.compressor:
            ap.error("--compressor is required")
        from .compress import run_jobs
        src = run.p("g1", f"jobs_{a.compressor}.jsonl") if a.g1 else run.p("compress", f"jobs_{a.compressor}.jsonl")
        out = run.p("g1", f"{a.compressor}.jsonl") if a.g1 else run.p("compress", f"{a.compressor}.jsonl")
        rows = run_jobs(C.read_jsonl(src), a.compressor, cfg)
        C.write_jsonl(out, rows)
        C.write_json(run.p("provenance", f"compress_{a.compressor}{'_g1' if a.g1 else ''}.json"),
                     C.provenance(cfg, "compress", {"compressor": a.compressor, "n": len(rows)}))
        print(f"[compress] {a.compressor}: {len(rows)} texts -> {out}")
    elif a.command == "random-control":
        P.step_random_control(run)
    elif a.command == "prompts":
        P.step_prompts(run)
    elif a.command == "generate":
        guard()
        if a.model not in cfg["generate"]["models"]:
            ap.error(f"--model must be one of {list(cfg['generate']['models'])}")
        from .generate import run as gen
        src = run.p("g1", "prompts.jsonl") if a.g1 else run.p("generate", "prompts.jsonl")
        out = run.p("g1", f"{a.model}.jsonl") if a.g1 else run.p("generate", f"{a.model}.jsonl")
        rows = gen(C.read_jsonl(src), a.model, cfg)
        C.write_jsonl(out, rows)
        C.write_json(run.p("provenance", f"generate_{a.model}{'_g1' if a.g1 else ''}.json"),
                     C.provenance(cfg, "generate", {"model": a.model, "n": len(rows) - 1}))
        print(f"[generate] {a.model}: {len(rows) - 1} answers -> {out}")
    elif a.command == "g1-jobs":
        P.step_g1_jobs(run)
    elif a.command == "analyze":
        P.step_analyze(run)
    elif a.command == "status":
        _status(run)


if __name__ == "__main__":
    main()
