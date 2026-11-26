"""The main study may not read anything from the pilot (decisions log: pilot archived after P0).

1. Static: no main-study module refers to a pilot location or imports pilot code.
2. Configuration: every pilot location is forbidden, and a configured path inside one is refused.
3. Guard: once installed, opening or listing a file in a pilot location raises PermissionError.
4. End to end: the E1 audit, E2 analysis and E5 analysis are run in a fresh process on a small
   synthetic study while an audit hook records every file opened; the test fails if any of them
   lies in a pilot location, and fails if the guard does not stop a deliberate pilot read.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "idea86"
sys.path.insert(0, str(ROOT / "src"))

PILOT_REF = re.compile(r"runs/P0|P0_annotation|P0_results|pilot_archive|p0\.yaml|P0_precommit|pilot/|"
                       r"import pilot|from \. import pilot|from \.pilot|from \.precommit|import precommit")
PILOT_LOCATIONS = {"pilot", "runs/pilot_archive", "runs/P0", "runs/P0_smoke", "runs/P0_round1", "P0_annotation", "P0_results"}


def test_no_pilot_references_in_main_code():
    bad = []
    for p in sorted(SRC.glob("*.py")):
        if p.name == "isolation.py":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if PILOT_REF.search(line):
                bad.append(f"{p.name}:{n}: {line.strip()}")
    for f in ("run_main.sh", "slurm/gpu_job.sbatch", "slurm/render_array.sbatch", "configs/main.yaml"):
        for n, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
            if PILOT_REF.search(line) and "forbidden:" not in line:
                bad.append(f"{f}:{n}: {line.strip()}")
    assert not bad, "main-study code refers to pilot files:\n" + "\n".join(bad)


def test_config_forbids_every_pilot_location():
    from idea86 import config as C
    from idea86 import isolation as I

    cfg = C.load(ROOT / "configs" / "main.yaml")
    assert PILOT_LOCATIONS <= set(cfg["isolation"]["forbidden"])
    I.check_config(cfg)                       # the real configuration passes
    for bad in ("runs/pilot_archive/E2", "pilot/runs", "P0_results"):
        c2 = C.Config({**cfg, "paths": {**cfg["paths"], "runs_dir": bad}})
        try:
            I.check_config(c2)
        except SystemExit as e:
            assert "isolation" in str(e)
        else:
            raise AssertionError(f"check_config accepted runs_dir = {bad}")


GUARD_SCRIPT = r'''
import os, sys, json
sys.path.insert(0, sys.argv[1])
from idea86 import config as C, isolation as I
cfg = C.load(sys.argv[2])
cfg["isolation"]["forbidden"] = cfg["isolation"]["forbidden"] + [sys.argv[3]]
I.install_guard(cfg)
res = {}
for name, fn in [("open", lambda: open(os.path.join(sys.argv[3], "secret.json")).read()),
                 ("listdir", lambda: os.listdir(sys.argv[3])),
                 ("scandir", lambda: list(os.scandir(sys.argv[3]))),
                 ("pathlib", lambda: C.read_json(os.path.join(sys.argv[3], "secret.json"))),
                 ("repo_pilot", lambda: open(os.path.join(sys.argv[4], "pilot", "README.md")).read())]:
    try:
        fn(); res[name] = "read"
    except PermissionError:
        res[name] = "blocked"
    except FileNotFoundError:
        res[name] = "missing"
open(os.path.join(sys.argv[5], "ok.txt"), "w").write("fine")   # outside: allowed
print(json.dumps(res))
'''


def test_guard_blocks_pilot_reads():
    with tempfile.TemporaryDirectory() as t:
        forb, out = Path(t, "pilot_like"), Path(t, "out")
        forb.mkdir(); out.mkdir()
        (forb / "secret.json").write_text("{}")
        r = subprocess.run([sys.executable, "-c", GUARD_SCRIPT, str(ROOT / "src"), str(ROOT / "configs" / "main.yaml"),
                            str(forb), str(ROOT), str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert all(res[k] == "blocked" for k in ("open", "listdir", "scandir", "pathlib")), res
        assert res["repo_pilot"] in ("blocked", "missing"), res
        assert (out / "ok.txt").read_text() == "fine"


E2E_SCRIPT = r'''
import os, sys, json
sys.path.insert(0, sys.argv[1])
opened = []
def rec(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        opened.append(os.path.abspath(os.fsdecode(args[0])))
sys.addaudithook(rec)
sys.path.insert(0, sys.argv[3])
import fake_study
from idea86 import isolation as I
cfg = fake_study.make(sys.argv[2])
I.check_config(cfg)
I.install_guard(cfg)
fake_study.run_all(cfg)
roots = [str(r) for r in I.forbidden_roots(cfg)]
hits = [p for p in opened if any(p == r or p.startswith(r + os.sep) for r in roots)]
try:
    open(os.path.join(roots[0], "README.md")).read()
    probe = "read"
except PermissionError:
    probe = "blocked"
except FileNotFoundError:
    probe = "missing"
print(json.dumps({"pilot_files_opened": hits, "n_opened": len(opened), "probe": probe}))
'''


def test_main_analysis_reads_no_pilot_file():
    with tempfile.TemporaryDirectory() as t:
        r = subprocess.run([sys.executable, "-c", E2E_SCRIPT, str(ROOT / "src"), t, str(Path(__file__).parent)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["n_opened"] > 0
        assert res["pilot_files_opened"] == [], res["pilot_files_opened"]
        assert res["probe"] in ("blocked", "missing")
