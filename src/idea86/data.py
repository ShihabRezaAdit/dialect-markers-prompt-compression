"""Dataset loading, count checks and the fixed P0 samples (internal plan Sections 7 and 8).

All sampling uses NumPy's default_rng(86). Each sample that the plan describes as
"shuffled with default_rng(86)" starts from a fresh generator, except the SQuAD
pilot sample, whose paragraph and question shuffles share one generator as the
plan states.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np

SEED = 86


# ---------------------------------------------------------------- SQuAD

def squad_paragraphs(records) -> list[dict]:
    """Group SQuAD rows into paragraphs numbered by first appearance.

    `records` is an iterable of dicts with keys id, title, context, question, answers
    (the Hugging Face `rajpurkar/squad` schema). A paragraph is a unique (title, context).
    Questions keep split order.
    """
    index: dict[tuple[str, str], int] = {}
    paras: list[dict] = []
    for r in records:
        key = (r["title"], r["context"])
        if key not in index:
            index[key] = len(paras)
            paras.append({"pnum": len(paras), "title": r["title"], "context": r["context"], "questions": []})
        answers = r["answers"]["text"] if isinstance(r["answers"], dict) else r["answers"]
        paras[index[key]]["questions"].append(
            {"qid": r["id"], "question": r["question"], "answers": list(answers)}
        )
    return paras


def load_squad(cfg: dict, split: str | None = None) -> list[dict]:
    """Load SQuAD v1.1 at the pinned revision (HF cache) and check the question count."""
    from datasets import load_dataset  # imported here so tests do not need `datasets`

    c = cfg["data"]["squad"]
    split = split or c["split"]
    ds = load_dataset(c["repo"], c["config"], revision=c["revision"], split=split)
    paras = squad_paragraphs(ds)
    n_q = sum(len(p["questions"]) for p in paras)
    expected = c["expected_questions"] if split == c["split"] else None
    if split == "validation":
        expected = 10570
        if len(paras) != 2067 or len({p["title"] for p in paras}) != 48:
            raise SystemExit(f"SQuAD validation: expected 2,067 paragraphs in 48 articles, got {len(paras)}")
    if expected is not None and n_q != expected:
        raise SystemExit(f"SQuAD {split}: expected {expected} questions, got {n_q}")
    return paras


def pilot_sample(paras: list[dict], n_paragraphs: int = 300, q_per_par: int = 2,
                 n_smoke: int = 10) -> dict:
    """P0 SQuAD sample (internal plan Section 8) plus the smoke-run paragraphs.

    Paragraphs with at least two questions are shuffled with default_rng(86); the first
    300 are the pilot. Within each kept paragraph its question IDs are shuffled with the
    same generator and the first two kept. The smoke run uses the next 10 paragraphs of
    the same shuffle (positions 300 to 309), so smoke items never overlap pilot items.
    """
    rng = np.random.default_rng(SEED)
    eligible = [p for p in paras if len(p["questions"]) >= q_per_par]
    order = rng.permutation(len(eligible))
    chosen = [eligible[i] for i in order[: n_paragraphs + n_smoke]]

    def take(par_list, q_rng):
        out_p, out_q = [], []
        for pos, p in enumerate(par_list):
            qperm = q_rng.permutation(len(p["questions"]))
            qs = [p["questions"][i] for i in qperm[:q_per_par]]
            out_p.append({"pos": pos, "pnum": p["pnum"], "title": p["title"], "context": p["context"],
                          "qids": [q["qid"] for q in qs]})
            for j, q in enumerate(qs):
                out_q.append({"pos": pos * q_per_par + j, "ppos": pos, "pnum": p["pnum"], **q})
        return out_p, out_q

    pilot_p, pilot_q = take(chosen[:n_paragraphs], rng)
    smoke_p, smoke_q = take(chosen[n_paragraphs:], rng)
    return {"passages": pilot_p, "questions": pilot_q,
            "smoke_passages": smoke_p, "smoke_questions": smoke_q}


def audit_sample(n_paragraphs: int = 300, per_dialect: int = 50, q_per_par: int = 2) -> dict:
    """Audit pairs: 50 pilot paragraphs for AAVE and 50 others for CollSgE, each shown
    with one of its two questions, all drawn with a fresh default_rng(86)."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_paragraphs)
    qpick = rng.integers(0, q_per_par, size=2 * per_dialect)
    pairs = []
    for i, ppos in enumerate(perm[: 2 * per_dialect]):
        dialect = "A" if i < per_dialect else "C"
        pairs.append({"ppos": int(ppos), "dialect": dialect,
                      "qpos": int(ppos) * q_per_par + int(qpick[i])})
    return {"pairs": pairs}


# ---------------------------------------------------------------- REAL-AAE

def read_real_aae(path: str | Path, cfg: dict) -> list[dict]:
    """Read the REAL-AAE CSV and check row count and the two text columns (checked at download)."""
    c = cfg["data"]["real_aae"]
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != c["expected_rows"]:
        raise ValueError(f"REAL-AAE: expected {c['expected_rows']} rows, got {len(rows)}")
    for col in c["text_columns"]:
        if col not in rows[0]:
            raise ValueError(f"REAL-AAE: column {col} missing")
        empty = sum(1 for r in rows if not (r.get(col) or "").strip())
        if empty:
            raise ValueError(f"REAL-AAE: {empty} rows with empty {col}")
    return rows


def real_aae_sample(n_rows: int, first: int, last: int) -> list[int]:
    """0-based CSV row indices of shuffle positions first..last (1-based, inclusive)."""
    perm = np.random.default_rng(SEED).permutation(n_rows)
    return [int(i) for i in perm[first - 1: last]]


# ---------------------------------------------------------------- UD-TwitterAAE fallback

def read_conllu_texts(text: str) -> list[str]:
    """Tweet texts from the `# text = ...` comment lines of a CoNLL-U file."""
    out = []
    for line in io.StringIO(text):
        m = re.match(r"#\s*text\s*=\s*(.*)$", line.rstrip("\n"))
        if m:
            out.append(m.group(1))
    return out


def ud_fallback_sample(texts: list[str], n_pilot: int = 50) -> tuple[list[int], list[int]]:
    """Shuffle one file's tweets with a fresh default_rng(86): first 50 pilot, rest study."""
    perm = np.random.default_rng(SEED).permutation(len(texts))
    return [int(i) for i in perm[:n_pilot]], [int(i) for i in perm[n_pilot:]]


# ---------------------------------------------------------------- download (login node, internet)

def _fetch(url: str, dest: Path) -> int:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.replace(dest)
    return dest.stat().st_size


def _snapshot(repo: str, revision: str) -> str:
    """Download one pinned model revision into the HF cache, skipping duplicate weight formats."""
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(repo, revision=revision)
    ignore = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot", "*.gguf", "*.pth", "original/*",
              "consolidated*", "onnx/*", "flax_model*", "tf_model*", "coreml/*", "rust_model*"]
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    return snapshot_download(repo_id=repo, revision=revision, ignore_patterns=ignore)


def download(cfg) -> dict:
    """Pilot preparation: SQuAD at its pinned revision, the REAL-AAE CSV (or, if it cannot be downloaded or
    its counts differ, the UD-TwitterAAE fallback), and the five pinned model revisions."""
    import zipfile

    from .config import now, read_json, sha256_file, write_json

    out_path = cfg.path("data_dir", "revisions.json")
    revs = read_json(out_path) if out_path.exists() else {}
    paras = load_squad(cfg)
    revs["squad"] = {**cfg["data"]["squad"], "paragraphs": len(paras),
                     "questions": sum(len(p["questions"]) for p in paras)}
    print(f"[download] SQuAD {cfg['data']['squad']['split']}: {revs['squad']['questions']} questions, "
          f"{len(paras)} paragraphs at {cfg['data']['squad']['revision']}")

    if "tweets" not in revs:
        csv_path = cfg.path("data_dir", "real_aae", "real_aae.csv")
        try:
            size = _fetch(cfg["data"]["real_aae"]["url"], csv_path)
            rows = read_real_aae(csv_path, cfg)
            revs["tweets"] = {"source": "REAL-AAE", "file": str(csv_path.relative_to(cfg.path("data_dir"))),
                              "bytes": size, "rows": len(rows), "sha256": sha256_file(csv_path), "utc": now()}
            print(f"[download] REAL-AAE: {len(rows)} rows, {size} bytes")
        except Exception as exc:  # noqa: BLE001 - any failure triggers the fixed fallback
            print(f"[download] REAL-AAE unavailable ({type(exc).__name__}: {exc}); using the UD-TwitterAAE fallback")
            u = cfg["data"]["ud_twitteraae"]
            zpath = cfg.path("data_dir", "ud_twitteraae", "TwitterAAE-UD-v1.zip")
            _fetch(u["url"], zpath)
            got = {}
            with zipfile.ZipFile(zpath) as z:
                for name in z.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base in u["files"]:
                        dest = cfg.path("data_dir", "ud_twitteraae", base)
                        dest.write_bytes(z.read(name))
                        got[base] = {"sha256": sha256_file(dest),
                                     "tweets": len(read_conllu_texts(dest.read_text(encoding="utf-8")))}
            missing = [f for f in u["files"] if f not in got]
            if missing or any(v["tweets"] != u["expected_tweets_per_file"] for v in got.values()):
                raise SystemExit(f"UD-TwitterAAE fallback incomplete: missing {missing}, counts {got}")
            revs["tweets"] = {"source": "UD-TwitterAAE", "reason": f"{type(exc).__name__}: {exc}",
                              "zip_sha256": sha256_file(zpath), "files": got, "utc": now()}
    else:
        print(f"[download] tweets already recorded: {revs['tweets']['source']}")

    revs.setdefault("models", {})
    for group in ("compress", "generate"):
        for key, m in cfg[group]["models"].items():
            path = _snapshot(m["repo"], m["revision"])
            revs["models"][key] = {"repo": m["repo"], "revision": m["revision"], "path": path}
            print(f"[download] model {key}: {m['repo']} at {m['revision']}")
    write_json(out_path, revs)
    return revs
