"""P0 pre-commitment (pilot plan, Section 9).

prereg/P0_precommit.md is written and committed to git before any pilot item is rendered,
compressed or generated. It records the plan sources, item lists with checksums, the SQuAD
revision, the tweet source and its SHA-256, the five model revisions, the five environment
lock files, the annotation guide copy, the code and its passing unit-test log, the gate
thresholds, the signal rules, the decision rule and the seed. Rendering, compression and
generation of pilot items verify it and refuse to start if anything differs.
"""
from __future__ import annotations

import json
import re

from .config import ROOT, Config, code_hash, git_commit, now, read_json, sha256_file

LOCKS = ("env-render", "env-compress", "env-sc", "env-gen", "env-stats")


class PrecommitError(RuntimeError):
    pass


def path(cfg: Config):
    return cfg.path("prereg_dir", f"{cfg['experiment']}_precommit.md")


def _state(cfg: Config) -> dict:
    items_file = cfg.run_dir() / "items.json"
    if not items_file.exists():
        raise PrecommitError("items.json not found: run ./run_pilot.sh items first")
    items = read_json(items_file)
    revs = read_json(cfg.path("data_dir", "revisions.json"))
    locks = {}
    for name in LOCKS:
        p = ROOT / "envs" / "lock" / f"{name}.lock.txt"
        if not p.exists():
            raise PrecommitError(f"{p} missing: run ./run_pilot.sh setup and commit envs/lock/")
        locks[name] = sha256_file(p)
    if not cfg["plan_sources"].get("annotation_guide_pdf"):
        raise PrecommitError("plan_sources.annotation_guide_pdf is empty: record the SHA-256 of the "
                             "frozen annotation guide copy in configs/p0.yaml first")
    return {
        "experiment": cfg["experiment"],
        "seed": cfg["seed"],
        "config_hash": cfg.hash,
        "plan_sources_sha256": cfg["plan_sources"],
        "item_lists_sha256": items["checksums"],
        "squad": {k: cfg["data"]["squad"][k] for k in ("repo", "config", "revision", "split")},
        "tweets": {k: v for k, v in revs["tweets"].items() if k != "utc"},
        "model_revisions": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in revs["models"].items()},
        "env_locks_sha256": locks,
        "gates": cfg["gates"],
        "signals": cfg["signals"],
        "decision_rule": cfg["decision_rule"],
        "code_hash": code_hash(),
    }


def _unit_tests_ok(cfg: Config) -> dict:
    p = cfg.run_dir() / "unit_tests.json"
    if not p.exists():
        raise PrecommitError("unit tests have not been run: ./run_pilot.sh test")
    t = read_json(p)
    if not t.get("passed") or t.get("code_hash") != code_hash():
        raise PrecommitError("unit tests did not pass for the current code: ./run_pilot.sh test")
    return t


def write(cfg: Config, force: bool = False) -> dict:
    p = path(cfg)
    if p.exists() and not force:
        raise PrecommitError(f"{p} already exists; a change after the pre-commitment must be logged "
                             "in the decisions log")
    state = _state(cfg)
    tests = _unit_tests_ok(cfg)
    meta = {"written_utc": now(), "git_commit": git_commit(), "unit_tests": tests}
    body = [
        "# P0 pre-commitment (Idea 86)",
        "",
        f"Written {meta['written_utc']} at git commit `{meta['git_commit']}`. Thresholds, rules, samples "
        "and seeds below do not change after this file is committed; any change is logged with its "
        "reason in the decisions log and reported in the paper.",
        "",
        "Rendering, compression and generation of pilot items verify the block below and refuse to "
        "start if anything differs.",
        "",
        "```json",
        json.dumps({"state": state, "meta": meta}, indent=1, sort_keys=True),
        "```",
        "",
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(body), encoding="utf-8")
    return state


def verify(cfg: Config) -> None:
    p = path(cfg)
    if not p.exists():
        raise PrecommitError(f"{p} not found: run ./run_pilot.sh precommit and commit it before "
                             "rendering, compressing or generating pilot items")
    m = re.search(r"```json\n(.*?)\n```", p.read_text(encoding="utf-8"), re.S)
    saved = json.loads(m.group(1))["state"]
    cur = _state(cfg)
    diffs = sorted(k for k in set(saved) | set(cur) if saved.get(k) != cur.get(k))
    if diffs:
        raise PrecommitError(f"pre-commitment mismatch in {diffs}; refusing to run")
