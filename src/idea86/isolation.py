"""Separation of the main study from the pilot (decisions log: pilot archived after P0).

Nothing from the pilot may be read by the main study. The configuration lists the pilot
locations (`isolation.forbidden`, relative to the repository or absolute): the archived pilot
outputs (runs/pilot_archive), the pilot folder of the repository (pilot/) and, for safety, the
pilot's former run folders. Two guards enforce this:

* check_config: every configured path (data, runs, pre-commitment) must lie outside them.
* install_guard: a Python audit hook (sys.addaudithook) that refuses to open, list or stat-walk
  any file inside them for the rest of the process. The main-study command line installs it
  before any step runs. Audit hooks cannot be removed, so tests run it in a subprocess.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import ROOT

_GUARD = {"roots": None}


def forbidden_roots(cfg) -> list[Path]:
    roots = []
    for p in cfg["isolation"]["forbidden"]:
        q = Path(p)
        q = q if q.is_absolute() else ROOT / q
        roots.append(Path(os.path.abspath(q)))
    return roots


def _inside(path: Path, roots: list[Path]) -> bool:
    p = os.path.abspath(path)
    for r in roots:
        r = str(r)
        if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
            return True
    return False


def check_config(cfg) -> None:
    roots = forbidden_roots(cfg)
    for key, val in cfg["paths"].items():
        p = Path(val)
        p = p if p.is_absolute() else ROOT / p
        if _inside(p, roots):
            raise SystemExit(f"isolation: paths.{key} = {val} overlaps a pilot location {[str(r) for r in roots]}")


def install_guard(cfg) -> list[Path]:
    """Refuse every file access inside a pilot location for the rest of this process."""
    roots = forbidden_roots(cfg)
    if _GUARD["roots"] is not None:
        _GUARD["roots"].extend(r for r in roots if r not in _GUARD["roots"])
        return _GUARD["roots"]
    _GUARD["roots"] = list(roots)

    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "glob.glob", "shutil.copyfile"):
            target = args[0] if args else None
            if isinstance(target, int) or target is None:
                return
            try:
                path = os.fsdecode(target)
            except TypeError:
                return
            if _inside(Path(path), _GUARD["roots"]):
                raise PermissionError(f"isolation: the main study may not read pilot files ({path})")

    sys.addaudithook(hook)
    return _GUARD["roots"]
