"""Rendering audit: spreadsheet and pair-view export, and label import with checks
(internal plan Section 9; annotation guide decisions D1 to D6).

Export writes, for one sitting:
    <outdir>/<sheet>.xlsx            pair_id, dialect, CONTENT_PRESERVED, ANSWER_PRESERVED, FLUENCY, note
    <outdir>/pairs/<pair_id>.html    SAE and dialect passages side by side, word diff highlighted,
                                     question and gold answers, gold answer marked on the SAE side only
    <keysdir>/<sheet>_keys.json      pair_id -> item, dialect, automatic answer flag (stays on the cluster)
The pair view shows no Multi-VALUE rule names (D1) and no automatic answer flag (D6).

Driven by `./run_main.sh audit-export` and `./run_main.sh audit-import`. Main-study (E1) sheets add
the TIME_HABIT_SHIFT column.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .config import read_json, write_json, write_jsonl
from .score import answer_survives
from .survival import diff_ops, tokens

COLUMNS = ["pair_id", "dialect", "CONTENT_PRESERVED", "ANSWER_PRESERVED", "FLUENCY", "note"]
ALLOWED = {"CONTENT_PRESERVED": ["YES", "NO"], "ANSWER_PRESERVED": ["YES", "NO"],
           "FLUENCY": ["NONE", "MINOR", "SEVERE"]}
# Main-study sheets (E1) add one column so that the audit can be reported under both readings of
# guide D5 (advisor's decision after P0): YES when the pair contains a tense, habit or voice shift,
# whatever CONTENT_PRESERVED says. The note must describe the shift.
SHIFT_COLUMN = "TIME_HABIT_SHIFT"
E1_COLUMNS = COLUMNS[:5] + [SHIFT_COLUMN] + COLUMNS[5:]


def columns_for(sheet: str) -> list[str]:
    return E1_COLUMNS if sheet.startswith("e1_") else COLUMNS


def allowed_for(columns: list[str]) -> dict:
    return {**ALLOWED, SHIFT_COLUMN: ["YES", "NO"]} if SHIFT_COLUMN in columns else dict(ALLOWED)
DIALECT_NAME = {"A": "AAVE", "C": "CollSgE", "I": "IndE"}
CLASS_NAME = {"AfricanAmericanVernacular": "AAVE", "ColloquialSingaporeDialect": "CollSgE", "IndianDialect": "IndE"}


def dialect_names(cfg) -> dict:
    """Display name per rendering code from the configured Multi-VALUE classes (C is IndE in round 2)."""
    return {code: CLASS_NAME.get(cls, cls) for code, cls in cfg["render"]["dialects"].items()}
PREFIX = {"p0_audit": "P", "p0_audit2": "Q", "e1_audit": "E", "e1_relabel": "R", "warmup_audit": "W"}

CSS = """
body{font-family:Georgia,serif;margin:24px;max-width:1300px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.box{border:1px solid #bbb;padding:12px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;margin:0 0 8px 0}
.rep{background:#fff2a8}.ins{background:#c9f2c4}.del{background:#ffd0d0;text-decoration:line-through}
.gold{outline:2px solid #1f5fbf;outline-offset:1px}
.q{margin-top:18px;border-top:1px solid #ddd;padding-top:12px}
.legend span{padding:1px 6px;margin-right:8px}
"""


# ---------------------------------------------------------------- pair view

def _gold_spans(sae: str, golds: list[str]) -> list[tuple[int, int]]:
    spans = []
    for g in golds:
        g = g.strip()
        if not g:
            continue
        i = sae.find(g)
        if i < 0:
            i = sae.lower().find(g.lower())
        if i >= 0:
            spans.append((i, i + len(g)))
    return spans


def _marked(text: str, toks, classes: dict[int, str], extra: list[tuple[int, int]] | None = None) -> str:
    """HTML for `text` with token classes (by token index) and extra outline spans."""
    extra = extra or []
    marks = [None] * len(text)
    for i, t in enumerate(toks):
        if i in classes:
            for c in range(t.start, t.end):
                marks[c] = classes[i]
    gold = [False] * len(text)
    for s, e in extra:
        for c in range(s, min(e, len(text))):
            gold[c] = True
    out, c = [], 0
    while c < len(text):
        state = (marks[c], gold[c])
        e = c
        while e < len(text) and (marks[e], gold[e]) == state:
            e += 1
        chunk = html.escape(text[c:e])
        if state[0]:
            chunk = f'<span class="{state[0]}">{chunk}</span>'
        if state[1]:
            chunk = f'<span class="gold">{chunk}</span>'
        out.append(chunk)
        c = e
    return "".join(out)


def pair_html(pair_id: str, dialect: str, sae: str, dia: str, question: str, golds: list[str]) -> str:
    st, dt = tokens(sae), tokens(dia)
    sae_cls, dia_cls = {}, {}
    for o in diff_ops(st, dt):
        if o.tag == "replace":
            sae_cls.update({i: "rep" for i in range(o.i1, o.i2)})
            dia_cls.update({j: "rep" for j in range(o.j1, o.j2)})
        elif o.tag == "insert":
            dia_cls.update({j: "ins" for j in range(o.j1, o.j2)})
        elif o.tag == "delete":
            sae_cls.update({i: "del" for i in range(o.i1, o.i2)})
    golds_html = "".join(f"<li>{html.escape(g)}</li>" for g in golds)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{pair_id}</title><style>{CSS}</style></head>
<body><h1>{pair_id} &middot; {DIALECT_NAME.get(dialect, dialect)}</h1>
<p class="legend"><span class="rep">replaced</span><span class="ins">inserted (dialect side)</span><span class="del">deleted (SAE side)</span><span class="gold">gold answer (SAE side only)</span></p>
<div class="grid"><div class="box"><h2>SAE passage</h2>{_marked(sae, st, sae_cls, _gold_spans(sae, golds))}</div>
<div class="box"><h2>Dialect passage ({DIALECT_NAME.get(dialect, dialect)})</h2>{_marked(dia, dt, dia_cls)}</div></div>
<div class="q"><b>Question (SAE):</b> {html.escape(question)}<br><b>Gold answers:</b><ul>{golds_html}</ul></div>
</body></html>"""


# ---------------------------------------------------------------- spreadsheet

def write_sheet(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = path.stem
    columns = columns or COLUMNS
    allowed = allowed_for(columns)
    ws.append(columns)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([r["pair_id"], DIALECT_NAME.get(r["dialect"], r["dialect"])] + [None] * (len(columns) - 2))
    n = len(rows) + 1
    letters = "ABCDEFGH"
    for i, key in enumerate(columns):
        if key in allowed:
            dv = DataValidation(type="list", formula1='"' + ",".join(allowed[key]) + '"', allow_blank=True,
                                showErrorMessage=True)
            dv.add(f"{letters[i]}2:{letters[i]}{n}")
            ws.add_data_validation(dv)
    widths = {"pair_id": 12, "dialect": 12, "CONTENT_PRESERVED": 22, "ANSWER_PRESERVED": 22, "FLUENCY": 14,
              SHIFT_COLUMN: 20, "note": 60}
    for i, key in enumerate(columns):
        ws.column_dimensions[letters[i]].width = widths[key]
    note_col = columns.index("note") + 1
    for row in ws.iter_rows(min_row=2, min_col=note_col, max_col=note_col):
        row[0].alignment = Alignment(wrap_text=True)
    ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def export(sheet: str, pairs: list[dict], outdir: Path, keysdir: Path, order_seed: int = 86,
           names: dict | None = None) -> list[dict]:
    """pairs: dicts with item_key, dialect, sae, dia, question, golds. Rows are put in a
    random order (default_rng(order_seed)) and given neutral IDs."""
    rng = np.random.default_rng(order_seed)
    order = rng.permutation(len(pairs))
    pre = PREFIX[sheet]
    rows, keys = [], {}
    for n, i in enumerate(order, start=1):
        p = pairs[int(i)]
        pid = f"{pre}{n:03d}"
        name = (names or DIALECT_NAME).get(p["dialect"], p["dialect"])   # shown to the annotator
        rows.append({"pair_id": pid, "dialect": name})
        (outdir / "pairs").mkdir(parents=True, exist_ok=True)
        (outdir / "pairs" / f"{pid}.html").write_text(
            pair_html(pid, name, p["sae"], p["dia"], p["question"], p["golds"]), encoding="utf-8")
        keys[pid] = {"item_key": p["item_key"], "dialect": p["dialect"],
                     "auto_answer_preserved": bool(answer_survives(p["golds"], p["dia"]))}
    write_sheet(outdir / f"{sheet}.xlsx", rows, columns_for(sheet))
    write_json(keysdir / f"{sheet}_keys.json", keys)
    return rows


# ---------------------------------------------------------------- import

def read_labels(xlsx: Path, columns: list[str] | None = None) -> list[dict]:
    from openpyxl import load_workbook

    columns = columns or COLUMNS
    ws = load_workbook(xlsx, data_only=True).active
    header = [c.value for c in ws[1]]
    if header[: len(columns)] != columns:
        raise SystemExit(f"{xlsx}: header {header} does not match {columns}")
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        out.append({k: (str(v).strip() if v is not None else "") for k, v in zip(columns, row)})
    return out


def validate(labels: list[dict], keys: dict, allowed: dict | None = None) -> list[str]:
    """All problems the import reports; an empty list means the sheet is accepted."""
    allowed = allowed or ALLOWED
    errs = []
    ids = [r["pair_id"] for r in labels]
    missing = sorted(set(keys) - set(ids))
    extra = sorted(set(ids) - set(keys))
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if missing:
        errs.append(f"rows missing: {missing}")
    if extra:
        errs.append(f"unknown pair_id: {extra}")
    if dup:
        errs.append(f"duplicate pair_id: {dup}")
    for r in labels:
        pid = r["pair_id"]
        for col, ok in allowed.items():
            v = r[col].upper()
            if v == "":
                errs.append(f"{pid}: {col} is blank")
            elif v not in ok:
                errs.append(f"{pid}: {col} = {r[col]!r} is not one of {ok}")
        if r.get(SHIFT_COLUMN, "").upper() == "YES" and not r["note"]:
            errs.append(f"{pid}: a note is required when {SHIFT_COLUMN} = YES (it names the shift)")
        if (r["CONTENT_PRESERVED"].upper() == "NO" or r["ANSWER_PRESERVED"].upper() == "NO"
                or r["FLUENCY"].upper() == "SEVERE") and not r["note"]:
            errs.append(f"{pid}: a note is required for every NO and every SEVERE (guide D2)")
        if r["ANSWER_PRESERVED"].upper() == "NO" and r["CONTENT_PRESERVED"].upper() == "YES":
            errs.append(f"{pid}: ANSWER_PRESERVED = NO requires CONTENT_PRESERVED = NO (guide D3)")
    return errs


def import_labels(xlsx: Path, keys_file: Path, columns: list[str] | None = None) -> tuple[list[dict], list[str]]:
    columns = columns or COLUMNS
    allowed = allowed_for(columns)
    keys = read_json(keys_file)
    labels = read_labels(xlsx, columns)
    errs = validate(labels, keys, allowed)
    merged = []
    if not errs:
        for r in labels:
            k = keys[r["pair_id"]]
            merged.append({**r, **{c: r[c].upper() for c in allowed}, **k})
    return merged, errs
