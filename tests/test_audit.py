import tempfile
from pathlib import Path

from openpyxl import load_workbook

from idea86 import audit as au


def pairs():
    return [{"item_key": f"passage:{i}:A", "dialect": "A" if i % 2 else "C",
             "sae": "The museum is open on Sundays.", "dia": "The museum open on Sundays.",
             "question": "When is it open?", "golds": ["Sundays"]} for i in range(6)]


def fill(ws, n=6):
    for r in range(2, n + 2):
        ws.cell(r, 3).value = "YES"
        ws.cell(r, 4).value = "YES"
        ws.cell(r, 5).value = "NONE"


def test_pair_html_word_diff_only_gold_on_sae_side():
    h = au.pair_html("P001", "A", "She is always studying in Paris.", "She always be studying in Paris.",
                     "Where?", ["Paris"])
    assert 'class="del"' in h and 'class="ins"' in h
    sae_box, dia_box = h.split("SAE passage</h2>")[1].split("Dialect passage")
    assert sae_box.count('<span class="gold">') == 1 and '<span class="gold">' not in dia_box
    assert "habitual" not in h and "zero_copula" not in h      # guide D1: no rule names


def test_export_and_import_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        rows = au.export("p0_audit", pairs(), t / "out", t / "keys")
        assert [r["pair_id"] for r in rows] == [f"P{i:03d}" for i in range(1, 7)]
        assert (t / "out" / "pairs" / "P001.html").exists()
        x = t / "out" / "p0_audit.xlsx"
        wb = load_workbook(x)
        ws = wb.active
        assert [c.value for c in ws[1]] == au.COLUMNS
        assert len(ws.data_validations.dataValidation) == 3
        fill(ws)
        ws.cell(2, 3).value = "NO"
        ws.cell(2, 6).value = "1932 -> 1923, NUMBER"
        wb.save(x)
        merged, errs = au.import_labels(x, t / "keys" / "p0_audit_keys.json")
        assert errs == [] and len(merged) == 6
        assert "auto_answer_preserved" in merged[0] and merged[0]["item_key"].startswith("passage:")


def test_import_reports_every_problem():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        au.export("p0_audit", pairs(), t / "out", t / "keys")
        x = t / "out" / "p0_audit.xlsx"
        wb = load_workbook(x)
        ws = wb.active
        fill(ws)
        ws.cell(2, 3).value = "NO"                                  # NO without a note (D2)
        ws.cell(3, 4).value = "NO"
        ws.cell(3, 6).value = "gone"                                # answer NO, content YES (D3)
        ws.cell(4, 5).value = "BAD"                                 # unknown label
        ws.cell(5, 3).value = None                                  # blank
        wb.save(x)
        _, errs = au.import_labels(x, t / "keys" / "p0_audit_keys.json")
        text = "\n".join(errs)
        assert "D2" in text and "D3" in text and "BAD" in text and "blank" in text
