"""A small synthetic main study (160 SQuAD-like paragraphs, 40 tweet pairs) in a temporary folder, used
by the isolation and analysis tests. Not a test module itself."""
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idea86 import config as C  # noqa: E402
from idea86 import survival as sv  # noqa: E402

SAE = ["She is always studying at the library and he is not there today.",
       "They were not ready, so the team is going to wait for the results."]
AAE = ["She always be studying at the library and he ain't there today.",
       "They wasn't ready, so the team finna wait for the results."]


def fake_compress(text: str, c: str, rho: float) -> str:
    keep = [t.text for t in sv.tokens(text) if zlib.crc32(f"{c}{rho}{t.norm}".encode()) % 100 < 100 * rho + 10]
    return " ".join(keep)


def make(tmp) -> C.Config:
    tmp = Path(tmp)
    cfg = C.load(ROOT / "configs" / "main.yaml")
    cfg = C.Config({**cfg, "paths": {"data_dir": str(tmp / "data"), "runs_dir": str(tmp / "runs"), "prereg_dir": str(tmp / "prereg")},
                    "stats": {**cfg["stats"], "bootstrap_resamples": 50}})
    from idea86 import study as E

    paras = []
    for i in range(160):
        paras.append({"pnum": i, "title": f"T{i}", "context": f"Paragraph {i}. " + SAE[i % 2] + " " + SAE[(i + 1) % 2],
                      "questions": [{"qid": f"q{i}_{j}", "question": f"Where is she {j}?", "answers": ["the library"]} for j in range(2)]})
    tweets = [{"aae_text": AAE[r % 2] + f" #{r}", "sae_text": SAE[r % 2] + f" #{r}"} for r in range(60)]
    items = E.build_items(paras, tweets, list(range(10, 50)), cfg)
    st = E.Study(cfg)
    C.write_json(st.p("E1", "items.json"), items)
    rendered = {}
    for p in items["passages"]:
        ok = p["pos"] % 17 != 5
        dia = f"Paragraph {p['pos']}. " + AAE[p["pos"] % 2] + " " + AAE[(p["pos"] + 1) % 2]
        rendered[f"passage:{p['pos']}:A"] = {"ok": ok, "text": dia if ok else None, "method": "whole" if p["pos"] % 3 else "no_coref", "rules": []}
    C.write_json(st.p("E1", "render", "rendered.json"), rendered)
    from idea86.compress import record
    jobs = E.build_compress_jobs(items, rendered, cfg)
    for c, js in jobs.items():
        C.write_jsonl(st.p("E2", "compress", f"jobs_{c}.jsonl"), js)
        C.write_jsonl(st.p("E2", "compress", f"{c}.jsonl"),
                      [record(j, {"compressed_prompt": fake_compress(j["text"], c, j["rho"])}, 0.0) for j in js])
    rng = np.random.default_rng(1)
    from idea86 import mechanism as M5
    for s in cfg["mechanism"]["scorers"]:
        rows = [{"uid": f"{u['corpus']}:{u['pos']}:{u['rendering']}", "scores": rng.random(len(sv.tokens(u["text"]))).tolist()}
                for u in M5.score_units(st)]
        C.write_jsonl(st.p("E5", "scores", s, "shard_000.jsonl"), rows)
    mb = M5.meetingbank_counts([(SAE[0], "She studying library he not there"), (AAE[1], "They wasn't ready team wait results")])
    C.write_json(st.p("E5", "meetingbank_counts.json"), mb)
    labels = [{"item_key": f"passage:{i}:A", "CONTENT_PRESERVED": "YES" if i % 10 else "NO", "ANSWER_PRESERVED": "YES" if i % 10 else "NO",
               "FLUENCY": "NONE", "TIME_HABIT_SHIFT": "YES" if i % 7 == 0 else "NO"} for i in range(150)]
    C.write_jsonl(st.p("E1", "audit", "labels_e1_audit.jsonl"), labels)
    return cfg


def run_all(cfg) -> dict:
    from idea86 import mechanism as M5
    from idea86 import study as E

    st = E.Study(cfg)
    E.step_audit_export(st, "e1_audit")
    E.step_audit_report(st)
    E.step_random_control(st)
    E.step_analyze_e2(st)
    M5.step_analyze(st)
    return {"retention": C.read_json(st.p("E2", "results", "retention.json")),
            "mechanism": C.read_json(st.p("E5", "results", "mechanism.json")),
            "audit": C.read_json(st.p("E1", "results", "audit.json"))}
