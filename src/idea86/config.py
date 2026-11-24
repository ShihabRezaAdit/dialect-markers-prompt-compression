"""Configuration, file helpers and provenance.

Every run is described by the committed configs/p0.yaml; its hash, the code hash and the
git commit go into every provenance record and into the pre-commitment.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # the repository (Pilot_Experiment) folder
PKG_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- hashing and files

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- configuration

class Config(dict):
    @property
    def hash(self) -> str:
        return stable_hash({k: v for k, v in self.items() if not k.startswith("_")})

    def path(self, key: str, *parts: str) -> Path:
        p = Path(self["paths"][key])
        p = p if p.is_absolute() else ROOT / p
        return p.joinpath(*parts)

    def run_dir(self, smoke: bool = False) -> Path:
        d = self.path("runs_dir") / (self["experiment"] + ("_smoke" if smoke else ""))
        d.mkdir(parents=True, exist_ok=True)
        return d


REQUIRED = ["experiment", "seed", "paths", "plan_sources", "data", "sample", "render", "compress",
            "generate", "stats", "gates", "signals", "decision_rule"]


def load(path="configs/p0.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    cfg = Config(raw)
    cfg["_config_file"] = str(path)
    return cfg


def model_path(repo: str, revision: str) -> str:
    """Local snapshot of a pinned model revision in the Hugging Face cache (HF_HOME).
    With HF_HUB_OFFLINE=1 (compute nodes) this only reads the cache; `download` fills it."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, revision=revision)


# ---------------------------------------------------------------- provenance

def code_hash() -> str:
    paths = sorted(PKG_DIR.glob("*.py"))
    return sha256_text("".join(p.name + sha256_file(p) for p in paths))


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def provenance(cfg: Config, stage: str, extra: dict | None = None) -> dict:
    rec = {"stage": stage, "experiment": cfg["experiment"], "config_hash": cfg.hash, "code_hash": code_hash(),
           "git_commit": git_commit(), "host": socket.gethostname(), "platform": platform.platform(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "gpu": gpu_name(), "utc": now()}
    if extra:
        rec.update(extra)
    return rec
