"""Main-study pipeline on the synthetic study: items and audit draw, the 7-day re-label rule,
compression jobs and shards, E2 families (Holm, H1c), E5 tables, and the E1 audit columns."""
import datetime as dt
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import fake_study  # noqa: E402
from idea86 import audit as au  # noqa: E402
from idea86 import config as C  # noqa: E402
from idea86 import study as E  # noqa: E402


def test_items_and_audit_draw_fixed():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        a = C.read_json(Path(fake_study.make(t1)["paths"]["runs_dir"]) / "E1" / "items.json")
        b = C.read_json(Path(fake_study.make(t2)["paths"]["runs_dir"]) / "E1" / "items.json")
        assert a["checksums"] == b["checksums"] and a["audit"] == b["audit"]
        r = a["audit"]["relabel_positions"]
        assert len(r) == 20 and len(set(r)) == 20 and all(0 <= x < 150 for x in r)
        assert sorted(a["audit"]["order"]) == list(range(160))
        assert len(a["tweets"]) == 40 and a["tweets"][0]["row"] == 10


def test_full_pipeline_and_families():
    with tempfile.TemporaryDirectory() as t:
        cfg = fake_study.make(t)
        out = fake_study.run_all(cfg)
        ret = out["retention"]
        assert set(ret["F1"]) == {"L1", "L2", "SC"} and set(ret["replication_real_aae"]) == {"L1", "L2", "SC"}
        for v in ret["F1"].values():
            assert 0 <= v["holm"]["p_holm"] <= 1 and v["holm"]["p_holm"] >= v["p"] - 1e-12
        assert "H1c" in ret and isinstance(ret["H1c"]["supported"], bool)
        assert len([k for k in ret["cells"] if k.startswith("squad|")]) == 12
        n_usable = sum(1 for p in range(160) if p % 17 != 5)
        assert ret["cells"]["squad|L1|0.5"]["n"] <= n_usable
        assert ret["cells"]["tweets|L2|0.5"]["n"] <= 40
        m = out["mechanism"]
        assert set(m["results"]) >= {"squad", "tweets", "tweets_short", "tweets_long"}
        assert m["coverage"]["L1"]["texts"] == m["coverage"]["L1"]["expected"]
        assert m["results"]["squad"]["score_L1"]["n_clusters"] > 0
        a = out["audit"]
        assert a["n"] == 150 and a["content_yes"] == 135 and a["content_yes_strict_d5"] <= a["content_yes"]
        st = E.Study(cfg)
        assert (st.p("E2", "report", "E2_report.md")).exists() and (st.p("E5", "report", "E5_report.md")).exists()
        assert (st.p("E1", "audit", "export", "e1_audit", "e1_audit.xlsx")).exists()
        lc = ret["by_lexicon_class"]
        assert lc["ops"].get("Auxiliary", 0) > 0 and "Auxiliary" in lc["classes"]


def test_relabel_needs_seven_days():
    with tempfile.TemporaryDirectory() as t:
        cfg = fake_study.make(t)
        st = E.Study(cfg)
        C.write_json(st.p("E1", "audit", "e1_audit_import.json"), {"utc": C.now(), "rows": 150})
        try:
            E.step_audit_export(st, "e1_relabel")
        except SystemExit as e:
            assert "days" in str(e)
        else:
            raise AssertionError("re-label sheet exported before 7 days")
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        C.write_json(st.p("E1", "audit", "e1_audit_import.json"), {"utc": old, "rows": 150})
        E.step_audit_export(st, "e1_relabel")
        keys = C.read_json(st.p("E1", "audit", "keys", "e1_relabel_keys.json"))
        assert len(keys) == 20 and all(k.startswith("R") for k in keys)


def test_shards_cover_jobs_once():
    jobs = list(range(1003))
    got = [x for i in range(24) for x in E.shard(jobs, i, 24)]
    assert got == jobs


def test_compress_jobs_ids_unique_and_rates():
    with tempfile.TemporaryDirectory() as t:
        cfg = fake_study.make(t)
        st = E.Study(cfg)
        jobs = E.build_compress_jobs(st.items(), st.rendered(), cfg)
        for c, js in jobs.items():
            ids = [j["job_id"] for j in js]
            assert len(ids) == len(set(ids))
            assert {j["rho"] for j in js} == set(cfg["compress"]["rates"])


def test_e1_audit_columns_and_shift_note():
    cols = au.columns_for("e1_audit")
    assert au.SHIFT_COLUMN in cols and cols.index(au.SHIFT_COLUMN) == cols.index("note") - 1
    assert au.SHIFT_COLUMN not in au.columns_for("p0_audit")
    keys = {"E001": {}}
    row = {"pair_id": "E001", "dialect": "AAVE", "CONTENT_PRESERVED": "YES", "ANSWER_PRESERVED": "YES",
           "FLUENCY": "NONE", au.SHIFT_COLUMN: "YES", "note": ""}
    errs = au.validate([row], keys, au.allowed_for(cols))
    assert any("TIME_HABIT_SHIFT" in e for e in errs)
    assert au.validate([{**row, "note": "habitual be"}], keys, au.allowed_for(cols)) == []


def test_precommit_write_and_verify():
    from idea86 import prereg as P

    with tempfile.TemporaryDirectory() as t:
        cfg = fake_study.make(t)
        cfg = C.Config({**cfg, "plan_sources": {k: "0" * 64 for k in cfg["plan_sources"]}})
        rev = {"squad": {"repo": "r", "config": "c", "revision": "v", "split": "validation", "paragraphs": 160, "questions": 320},
               "real_aae": {"rows": 60, "sha256": "x", "study_rows": 40},
               "meetingbank": {"repo": "m", "revision": "v", "file": "f", "rows": 2, "sha256": "y"},
               "models": {"llama2": {"repo": "a", "revision": "b", "path": "/p"}}}
        C.write_json(cfg.path("data_dir", "revisions_main.json"), rev)
        try:
            P.write(cfg)
        except P.PreregError as e:
            assert "unit tests" in str(e)
        else:
            raise AssertionError("pre-commitment written without a unit-test log")
        C.write_json(P.unit_tests_file(cfg), {"passed": True, "code_hash": C.code_hash()})
        P.write(cfg)
        P.verify(cfg)
        try:
            P.write(cfg)
        except P.PreregError:
            pass
        else:
            raise AssertionError("pre-commitment overwritten without --force")
        changed = C.Config({**cfg, "compress": {**cfg["compress"], "rates": [0.5]}})
        try:
            P.verify(changed)
        except P.PreregError as e:
            assert "config_hash" in str(e)
        else:
            raise AssertionError("a changed configuration passed verification")
