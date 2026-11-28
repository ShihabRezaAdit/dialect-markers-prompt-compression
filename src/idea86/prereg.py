"""Main-study pre-commitment (internal plan Section 22; advisor's decisions after the pilot).

prereg/prereg_main.md is written and committed to git before any main-study text is rendered,
compressed or scored. It records the configuration hash, the plan sources and the frozen annotation
guide (SHA-256), the item lists with checksums, the data revisions (SQuAD, REAL-AAE file SHA-256,
MeetingBank-LLMCompressed), the pinned model revisions, the environment lock files, the code hash
with its passing unit-test log, the hypotheses, the confirmatory families and the priorities.
Rendering, compression and mechanism scoring verify it and refuse to start if anything differs.
Nothing from the pilot is recorded or read here.
"""
from __future__ import annotations

import json
import re

from .config import ROOT, Config, code_hash, git_commit, now, read_json, sha256_file

LOCKS = ("env-render", "env-compress", "env-sc", "env-gen", "env-stats")
FILE = "prereg_main.md"


class PreregError(RuntimeError):
    pass


def path(cfg: Config):
    return cfg.path("prereg_dir", FILE)


def unit_tests_file(cfg: Config):
    return cfg.run_dir("setup") / "unit_tests.json"


def _state(cfg: Config) -> dict:
    items_file = cfg.path("runs_dir", "E1", "items.json")
    if not items_file.exists():
        raise PreregError("runs/E1/items.json not found: run ./run_main.sh items first")
    revs_file = cfg.path("data_dir", "revisions_main.json")
    if not revs_file.exists():
        raise PreregError("data/revisions_main.json not found: run ./run_main.sh download first")
    revs = read_json(revs_file)
    bad = [k for k, v in cfg["plan_sources"].items() if not v or str(v).startswith("PLACEHOLDER")]
    if bad:
        raise PreregError(f"plan_sources {bad} are not filled in configs/main.yaml")
    locks = {}
    for name in LOCKS:
        p = ROOT / "envs" / "lock" / f"{name}.lock.txt"
        if not p.exists():
            raise PreregError(f"{p} missing: run ./run_main.sh setup and commit envs/lock/")
        locks[name] = sha256_file(p)
    return {
        "study": cfg["study"],
        "seed": cfg["seed"],
        "config_hash": cfg.hash,
        "plan_sources_sha256": cfg["plan_sources"],
        "item_lists_sha256": read_json(items_file)["checksums"],
        "data": {"squad": {k: revs["squad"][k] for k in ("repo", "config", "revision", "split", "paragraphs", "questions")},
                 "real_aae": revs["real_aae"],
                 "meetingbank": {k: revs["meetingbank"][k] for k in ("repo", "revision", "file", "rows", "sha256")}},
        "model_revisions": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in revs["models"].items()},
        "env_locks_sha256": locks,
        "hypotheses": cfg["hypotheses"],
        "priorities": cfg["priorities"],
        "code_hash": code_hash(),
    }


def _unit_tests_ok(cfg: Config) -> dict:
    p = unit_tests_file(cfg)
    if not p.exists():
        raise PreregError("unit tests have not been run: ./run_main.sh test")
    t = read_json(p)
    if not t.get("passed") or t.get("code_hash") != code_hash():
        raise PreregError("unit tests did not pass for the current code: ./run_main.sh test")
    return t


def write(cfg: Config, force: bool = False) -> dict:
    p = path(cfg)
    if p.exists() and not force:
        raise PreregError(f"{p} already exists; a change after the pre-commitment must be logged in the decisions log")
    state = _state(cfg)
    meta = {"written_utc": now(), "git_commit": git_commit(), "unit_tests": _unit_tests_ok(cfg)}
    body = [
        "# Main-study pre-commitment 1 of 2: must-have experiments (Idea 86)",
        "",
        f"Written {meta['written_utc']} at git commit `{meta['git_commit']}`, before any main-study text was "
        "rendered, compressed or scored. The configuration, prompts, item lists, hypotheses, families and the "
        "pipeline below are frozen; any change is logged with its reason in the decisions log and reported.",
        "",
        "This is the first of two main-study pre-commitments (decisions log: Gate 1 change). It records the "
        "SHA-256 of the internal execution plan's source (plan_sources_sha256 below), so the E3 hypotheses, "
        "the H3 rule, the four answer models and their pinned revisions, the prompt templates and every "
        "threshold are frozen by the plan text from this commit on. The second, generation pre-commitment "
        "freezes the E3 code and the result of the Mistral-7B-Instruct-v0.3 and Gemma-2-9B-it setup check; it "
        "is committed before any E3 answer is generated, and no accuracy or flip analysis of main-study data "
        "takes place before it.",
        "",
        "Confirmatory dialect: AAVE. Families: F1 (three Delta_s tests, Holm) on SQuAD at keep rate 0.5, with "
        "REAL-AAE as the replication family; F2 to F4 are confirmatory only if their experiment is run. H3 is "
        "supported if at least 6 of the 12 AAVE accuracy cells widen the gap significantly and none narrows it "
        "significantly; accuracy is measured on answer-preserved questions, with all questions as a secondary analysis.",
        "",
        "```json",
        json.dumps({"state": state, "meta": meta}, indent=1, sort_keys=True),
        "```",
        "",
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(body), encoding="utf-8")
    return state


def saved(cfg: Config) -> dict:
    p = path(cfg)
    if not p.exists():
        raise PreregError(f"{p} not found: run ./run_main.sh precommit and commit it before rendering, "
                          "compressing or scoring main-study texts")
    m = re.search(r"```json\n(.*?)\n```", p.read_text(encoding="utf-8"), re.S)
    return json.loads(m.group(1))["state"]


def verify(cfg: Config) -> None:
    old, cur = saved(cfg), _state(cfg)
    diffs = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
    if diffs:
        raise PreregError(f"pre-commitment mismatch in {diffs}; refusing to run")
