"""Main-study command line: python -m idea86 <command> [--config configs/main.yaml]

Order (README.md; run through ./run_main.sh):
  download -> warmup-render / warmup-compress -> test -> items -> precommit (commit it)
  E1: render-jobs -> render (CPU array) -> render-collect -> audit-export --sheet e1_audit
      -> (you label) -> audit-import --sheet e1_audit -> after >= 7 days: audit-export / audit-import
      --sheet e1_relabel -> audit-report
  E2: compress-jobs -> compress --compressor C --shard i --nshards n (GPU) -> compress-collect
      -> random-control -> analyze-e2
  E5: mech-score --scorer S --shard i --nshards n (GPU) -> mech-meetingbank -> mech-analyze

The pilot is archived separately (git tag pilot-final). Before any step this command installs the
isolation guard: no file in a pilot location can be opened by the main study.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import config as C
from .isolation import _inside, check_config, forbidden_roots, install_guard

COMMANDS = ["download", "warmup-render", "warmup-compress", "test", "items", "precommit", "render-jobs", "render",
            "render-collect", "audit-export", "audit-import", "audit-report", "compress-jobs", "compress",
            "compress-collect", "random-control", "analyze-e2", "mech-score", "mech-meetingbank", "mech-analyze", "status"]


def _cmd_test(cfg) -> None:
    r = subprocess.run([sys.executable, str(C.ROOT / "tests" / "run_tests.py")], cwd=C.ROOT)
    from .prereg import unit_tests_file
    C.write_json(unit_tests_file(cfg), {"passed": r.returncode == 0, "code_hash": C.code_hash(),
                                        "git_commit": C.git_commit(), "utc": C.now()})
    if r.returncode:
        raise SystemExit("unit tests failed")


def _status(st) -> None:
    cfg = st.cfg
    from .prereg import path as prereg_path, unit_tests_file
    checks = [("data/revisions_main.json", cfg.path("data_dir", "revisions_main.json")),
              ("setup/unit_tests.json", unit_tests_file(cfg)), ("E1/items.json", st.p("E1", "items.json")),
              ("prereg/prereg_main.md", prereg_path(cfg)),
              ("E1/render/jobs.jsonl", st.p("E1", "render", "jobs.jsonl")),
              ("E1/render/rendered.json", st.p("E1", "render", "rendered.json")),
              ("E1/audit/labels_e1_audit.jsonl", st.p("E1", "audit", "labels_e1_audit.jsonl")),
              ("E1/audit/labels_e1_relabel.jsonl", st.p("E1", "audit", "labels_e1_relabel.jsonl")),
              ("E1/results/audit.json", st.p("E1", "results", "audit.json"))]
    for c in cfg["compress"]["compressors"]:
        n = cfg["compress"]["shards"][c]
        done = len(list(st.p("E2", "compress", c).glob("shard_*.jsonl")))
        checks.append((f"E2/compress/{c} shards {done}/{n}", st.p("E2", "compress", f"{c}.jsonl")))
    checks += [("E2/compress/R.jsonl", st.p("E2", "compress", "R.jsonl")),
               ("E2/report/E2_report.md", st.p("E2", "report", "E2_report.md"))]
    for s in cfg["mechanism"]["scorers"]:
        n = cfg["mechanism"]["shards"][s]
        done = len(list(st.p("E5", "scores", s).glob("shard_*.jsonl")))
        checks.append((f"E5/scores/{s} shards {done}/{n}", st.p("E5", "scores", s, f"shard_{n - 1:03d}.jsonl")))
    checks += [("E5/meetingbank_counts.json", st.p("E5", "meetingbank_counts.json")),
               ("E5/report/E5_report.md", st.p("E5", "report", "E5_report.md"))]
    for name, p in checks:
        print(f"  [{'x' if p.exists() else ' '}] {name}")


def _shard_args(a, ap, n_default: int) -> tuple[int, int]:
    i = a.shard if a.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n = a.nshards or n_default
    if not 0 <= i < n:
        ap.error(f"--shard must be in 0..{n - 1}")
    return i, n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="idea86", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--task", type=int, help="render: array task index (default SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--ntasks", type=int, help="render: number of array tasks (default from config)")
    ap.add_argument("--compressor", choices=["L1", "L2", "SC"])
    ap.add_argument("--scorer", choices=["L1", "L2", "SC"])
    ap.add_argument("--shard", type=int, help="compress / mech-score: shard index (default SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--nshards", type=int, help="compress / mech-score: number of shards (default from config)")
    ap.add_argument("--sheet", choices=["e1_audit", "e1_relabel"])
    ap.add_argument("--force", action="store_true", help="precommit: overwrite (must be logged)")
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    if cfg.get("study") != "main":
        raise SystemExit(f"{a.config} is not the main-study configuration")
    if _inside(C.Path(cfg["_config_file"]), forbidden_roots(cfg)):
        raise SystemExit("isolation: the configuration file lies in a pilot location")
    check_config(cfg)
    install_guard(cfg)          # from here on no file in a pilot location can be opened

    if a.command == "download":
        from .data import download_main
        download_main(cfg)
        return
    if a.command == "test":
        _cmd_test(cfg)
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

    from . import mechanism as E5
    from . import study as E
    from .prereg import verify
    st = E.Study(cfg)

    if a.command == "items":
        E.step_items(st)
    elif a.command == "precommit":
        from .prereg import write
        write(cfg, force=a.force)
        print("pre-commitment written to prereg/prereg_main.md; commit and push it before rendering")
    elif a.command == "render-jobs":
        verify(cfg)
        E.step_render_jobs(st)
    elif a.command == "render":
        verify(cfg)
        from .render import run_shard
        task = a.task if a.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        ntasks = a.ntasks or cfg["render"]["array_tasks"]
        rows = run_shard(C.read_jsonl(st.p("E1", "render", "jobs.jsonl")), task, ntasks, cfg["render"])
        C.write_jsonl(st.p("E1", "render", "shards", f"shard_{task:03d}.jsonl"), rows)
        print(f"[render] task {task}/{ntasks}: {len(rows)} texts, {sum(not r['ok'] for r in rows)} failed")
    elif a.command == "render-collect":
        E.step_render_collect(st)
    elif a.command == "audit-export":
        if not a.sheet:
            ap.error("--sheet e1_audit or e1_relabel is required")
        E.step_audit_export(st, a.sheet)
    elif a.command == "audit-import":
        if not a.sheet:
            ap.error("--sheet e1_audit or e1_relabel is required")
        E.step_audit_import(st, a.sheet)
    elif a.command == "audit-report":
        E.step_audit_report(st)
    elif a.command == "compress-jobs":
        verify(cfg)
        E.step_compress_jobs(st)
    elif a.command == "compress":
        verify(cfg)
        if not a.compressor:
            ap.error("--compressor is required")
        from .compress import run_jobs
        i, n = _shard_args(a, ap, cfg["compress"]["shards"][a.compressor])
        jobs = E.shard(C.read_jsonl(st.p("E2", "compress", f"jobs_{a.compressor}.jsonl")), i, n)
        rows = run_jobs(jobs, a.compressor, cfg)
        C.write_jsonl(st.p("E2", "compress", a.compressor, f"shard_{i:03d}.jsonl"), rows)
        C.write_json(st.p("E2", "provenance", f"compress_{a.compressor}_{i:03d}.json"),
                     C.provenance(cfg, "compress", {"compressor": a.compressor, "shard": i, "nshards": n, "n": len(rows)}))
        print(f"[compress] {a.compressor} shard {i}/{n}: {len(rows)} texts")
    elif a.command == "compress-collect":
        E.step_compress_collect(st)
    elif a.command == "random-control":
        E.step_random_control(st)
    elif a.command == "analyze-e2":
        E.step_analyze_e2(st)
    elif a.command == "mech-score":
        verify(cfg)
        if not a.scorer:
            ap.error("--scorer is required")
        i, n = _shard_args(a, ap, cfg["mechanism"]["shards"][a.scorer])
        E5.step_score(st, a.scorer, i, n)
        C.write_json(st.p("E5", "provenance", f"score_{a.scorer}_{i:03d}.json"),
                     C.provenance(cfg, "mech-score", {"scorer": a.scorer, "shard": i, "nshards": n}))
    elif a.command == "mech-meetingbank":
        E5.step_meetingbank(st)
    elif a.command == "mech-analyze":
        E5.step_analyze(st)
    elif a.command == "status":
        _status(st)


if __name__ == "__main__":
    main()
