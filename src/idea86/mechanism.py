"""E5a: why the compressors differ (internal plan Section 19; exploratory, no hypothesis tests).

For every text that E2 compresses (SQuAD SAE and AAVE passages, REAL-AAE SAE and AAE tweets) each
compressor's own model scores every word token (survival.tokens):

  L1  Llama-2-7B (the LLMLingua scorer, pinned revision): surprisal -log p of each token given the
      preceding text of the same text (beginning-of-text token first), summed over the word's tokens.
  L2  LLMLingua-2's classifier (pinned revision): probability of the "keep" label for each token,
      averaged over the word's tokens (llmlingua's own word rule); texts longer than the model's
      window are scored in consecutive windows.
  SC  Selective Context's GPT-2 (pinned revision): the self-information the package computes for its
      phrase-level lexical units (sentences split and scored exactly as in SelectiveContext.__call__),
      averaged over the units a word overlaps.

From LLMLingua-2's training data (MeetingBank-LLMCompressed) every case-folded word gets its number of
occurrences in the original transcripts and the share of those kept by GPT-4 in the compressed
version (survival.word_survival, the same alignment as the main measure).

The analysis pairs each replaced dialect span with its standard twin, takes the dialect-minus-twin
difference of each score and relates it to the survival difference of the matching compressor at
the E2 keep rate. Results: paragraph (or tweet) means with 95% cluster-bootstrap intervals, and the
survival difference by quantile of the score difference.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import stats as S
from . import survival as sv
from .config import read_jsonl, write_json

SCORER_COMPRESSOR = {"L1": "L1", "L2": "L2", "SC": "SC"}


# ---------------------------------------------------------------- token -> word scores

def word_scores(text: str, spans: list[tuple[int, int]], values: list[float], agg: str) -> list[float | None]:
    """Score per word token of `text` from scores of model tokens with character spans `spans`.
    A model token belongs to the word it overlaps most (ties: the first); tokens that overlap no word
    (spaces, punctuation) are ignored. agg = "sum" or "mean"; a word without tokens gets None."""
    words = sv.tokens(text)
    starts = [w.start for w in words]
    acc: list[list[float]] = [[] for _ in words]
    import bisect
    for (s, e), v in zip(spans, values):
        if e <= s or v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        i = max(0, bisect.bisect_right(starts, s) - 1)
        best, best_ov = None, 0
        for j in range(i, len(words)):
            w = words[j]
            if w.start >= e:
                break
            ov = min(e, w.end) - max(s, w.start)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            acc[best].append(float(v))
    if agg == "sum":
        return [float(sum(a)) if a else None for a in acc]
    return [float(np.mean(a)) if a else None for a in acc]


# ---------------------------------------------------------------- scorers (GPU jobs)

class L1Scorer:
    def __init__(self, path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        self.dev = dev

    def __call__(self, text: str) -> list[float | None]:
        torch = self.torch
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [self.tok.bos_token_id] + enc["input_ids"]
        with torch.no_grad():
            logits = self.model(torch.tensor([ids], device=self.dev)).logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1)
            surpr = (-lp.gather(1, torch.tensor(ids[1:], device=self.dev).unsqueeze(1)).squeeze(1)).tolist()
        return word_scores(text, enc["offset_mapping"], surpr, "sum")


class L2Scorer:
    def __init__(self, path: str, window: int = 510):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(dev).eval()
        self.dev, self.window = dev, window

    def __call__(self, text: str) -> list[float | None]:
        torch = self.torch
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        probs: list[float] = []
        with torch.no_grad():
            for lo in range(0, len(ids), self.window):
                chunk = [self.tok.cls_token_id] + ids[lo:lo + self.window] + [self.tok.sep_token_id]
                logits = self.model(torch.tensor([chunk], device=self.dev)).logits[0].float()
                probs += torch.softmax(logits, dim=-1)[1:-1, 1].tolist()
        return word_scores(text, offs, probs, "mean")


class SCScorer:
    def __init__(self, path: str):
        from .compress import make_selective_context

        self.sc = make_selective_context(path)

    def units(self, text: str) -> tuple[str, list[str], list[float]]:
        sc = self.sc
        context = sc.beautify_context(text)
        sents = [s.strip() for s in re.split(sc.sent_tokenize_pattern, context) if s.strip()]
        _, phrase, _ = sc._lexical_unit(sents)
        return context, list(phrase.text), [float(x) for x in phrase.self_info]

    def __call__(self, text: str) -> list[float | None]:
        context, units, info = self.units(text)
        return unit_word_scores(text, context, units, info)


def unit_word_scores(text: str, context: str, units: list[str], info: list[float]) -> list[float | None]:
    """Map Selective Context's lexical units back to the word tokens of `text`. Each unit is found,
    in order, in the cleaned context the package scores; the context must have the same word tokens
    as the text (the package only normalises spacing), otherwise the text is recorded as failed."""
    if [w.norm for w in sv.tokens(context)] != [w.norm for w in sv.tokens(text)]:
        raise ValueError("Selective Context's cleaned text has different words from the original")
    spans, vals, pos = [], [], 0
    for u, v in zip(units, info):
        s_ = u.strip()
        if not s_:
            continue
        i = context.find(s_, pos)
        if i < 0:
            raise ValueError(f"unit {s_!r} not found in the cleaned text")
        spans.append((i, i + len(s_)))
        vals.append(v)
        pos = i + len(s_)
    # word positions in `context` equal word positions in `text` (checked above); a phrase unit covers
    # several words, so every word gets the mean self-information of the units it overlaps
    out = []
    for w in sv.tokens(context):
        v = [x for (s, e), x in zip(spans, vals) if s < w.end and w.start < e]
        out.append(float(np.mean(v)) if v else None)
    return out


def score_texts(scorer, units: list[dict]) -> list[dict]:
    out = []
    for u in units:
        uid = f"{u['corpus']}:{u['pos']}:{u['rendering']}"
        try:
            out.append({"uid": uid, "scores": scorer(u["text"])})
        except Exception as e:  # noqa: BLE001 - recorded and counted, never silently dropped
            out.append({"uid": uid, "scores": None, "error": f"{type(e).__name__}: {e}"})
    return out


# ---------------------------------------------------------------- MeetingBank (CPU job)

def meetingbank_counts(pairs) -> dict:
    """pairs: iterable of (original transcript, GPT-4 compressed transcript). Returns per case-folded
    word: [occurrences in the originals, occurrences kept], and totals."""
    counts: dict[str, list[int]] = {}
    n = 0
    for orig, comp in pairs:
        ot = sv.tokens(orig or "")
        kept = sv.survived(sv.word_survival(ot, sv.tokens(comp or "")))
        for t, k in zip(ot, kept):
            c = counts.setdefault(t.norm, [0, 0])
            c[0] += 1
            c[1] += int(k)
        n += 1
    return {"documents": n, "words": counts}


# ---------------------------------------------------------------- analysis

def _span_mean(xs, lo, hi):
    v = [x for x in xs[lo:hi] if x is not None]
    return float(np.mean(v)) if v else None


def op_table(pairs, scores: dict, comp: dict, mb: dict | None, rho: float, corpus: str) -> list[dict]:
    """One row per replace op: cluster (pair position), dialect-minus-twin score differences per
    scorer and MeetingBank feature, and survival differences per compressor."""
    gS, gD = ("S", "A") if corpus == "squad" else ("SAE", "AAE")
    rows = []
    for pos, sae, dia in pairs:
        st_, dt_ = sv.tokens(sae), sv.tokens(dia)
        ops = [o for o in sv.diff_ops(st_, dt_) if o.tag == "replace"]
        if not ops:
            continue
        surv = {}
        for c in SCORER_COMPRESSOR.values():
            cs = (comp.get(f"{corpus}:{pos}:{gS}:{c}:{rho}") or {}).get("compressed")
            cd = (comp.get(f"{corpus}:{pos}:{gD}:{c}:{rho}") or {}).get("compressed")
            if cs is not None and cd is not None:
                surv[c] = (sv.survived(sv.word_survival(st_, sv.tokens(cs))), sv.survived(sv.word_survival(dt_, sv.tokens(cd))))
        for o in ops:
            row = {"cluster": pos}
            for sc_name in SCORER_COMPRESSOR:
                a = (scores.get(sc_name) or {}).get(f"{corpus}:{pos}:{gD}")
                b = (scores.get(sc_name) or {}).get(f"{corpus}:{pos}:{gS}")
                if a is not None and b is not None:
                    d, s_ = _span_mean(a, o.j1, o.j2), _span_mean(b, o.i1, o.i2)
                    row[f"score_{sc_name}"] = None if d is None or s_ is None else d - s_
            if mb is not None:
                w = mb["words"]
                def feats(toks):
                    occ = [w.get(t.norm, [0, 0])[0] for t in toks]
                    keep = [w[t.norm][1] / w[t.norm][0] for t in toks if w.get(t.norm, [0, 0])[0] > 0]
                    return float(np.mean([math.log1p(x) for x in occ])), (float(np.mean(keep)) if keep else None)
                fd, kd = feats(dt_[o.j1:o.j2])
                fs_, ks = feats(st_[o.i1:o.i2])
                row["mb_log_freq"] = fd - fs_
                row["mb_keep_share"] = None if kd is None or ks is None else kd - ks
            for c, (ks_, kd_) in surv.items():
                row[f"surv_{c}"] = float(kd_[o.j1:o.j2].mean() - ks_[o.i1:o.i2].mean())
            rows.append(row)
    return rows


def summarize(rows: list[dict], feature: str, surv_key: str | None, B: int, seed: int, q: int) -> dict:
    """Cluster-mean of a dialect-minus-twin feature with a 95% bootstrap interval, and the mean
    survival difference by quantile of the feature (op level)."""
    by: dict[int, list[float]] = {}
    for r in rows:
        if r.get(feature) is not None:
            by.setdefault(r["cluster"], []).append(r[feature])
    if not by:
        return {"n_clusters": 0}
    v = np.array([np.mean(x) for _, x in sorted(by.items())])
    est, draws = S.bootstrap(S.mean_stat(v), len(v), B, seed)
    out = {"estimate": est, "ci95": S.percentile_ci(draws, 0.95), "n_clusters": len(v),
           "n_ops": sum(len(x) for x in by.values())}
    if surv_key:
        pts = [(r[feature], r[surv_key]) for r in rows if r.get(feature) is not None and r.get(surv_key) is not None]
        if len(pts) >= q:
            f = np.array([p[0] for p in pts]); s_ = np.array([p[1] for p in pts])
            cuts = np.quantile(f, np.linspace(0, 1, q + 1))
            idx = np.clip(np.searchsorted(cuts, f, side="right") - 1, 0, q - 1)
            out["survival_by_quantile"] = [{"quantile": k + 1, "feature_mean": float(f[idx == k].mean()),
                                            "survival_diff_mean": float(s_[idx == k].mean()), "ops": int((idx == k).sum())}
                                           for k in range(q) if (idx == k).any()]
            fr = np.argsort(np.argsort(f)); sr = np.argsort(np.argsort(s_))
            out["spearman_op_level"] = float(np.corrcoef(fr, sr)[0, 1]) if len(pts) > 2 else None
    return out


def analyze(rows_by_corpus: dict[str, list[dict]], B: int, seed: int, q: int) -> dict:
    res = {}
    for corpus, rows in rows_by_corpus.items():
        r = {}
        for sc_name, c in SCORER_COMPRESSOR.items():
            r[f"score_{sc_name}"] = summarize(rows, f"score_{sc_name}", f"surv_{c}", B, seed, q)
        r["mb_log_freq"] = summarize(rows, "mb_log_freq", "surv_L2", B, seed, q)
        r["mb_keep_share"] = summarize(rows, "mb_keep_share", "surv_L2", B, seed, q)
        for c in SCORER_COMPRESSOR.values():
            r[f"surv_{c}"] = summarize(rows, f"surv_{c}", None, B, seed, q)
        res[corpus] = r
    return res


# ---------------------------------------------------------------- steps (runs/E5)

def score_units(st) -> list[dict]:
    """Every distinct text E2 compresses (no rate): SQuAD SAE and usable AAVE passages, REAL-AAE pairs."""
    from .study import text_units
    return text_units(st.items(), st.rendered(), st.cfg)


def step_score(st, scorer: str, i: int, n: int) -> None:
    from .compress import _path
    from .config import write_jsonl
    from .study import shard

    units = shard(score_units(st), i, n)
    model = {"L1": lambda: L1Scorer(_path(st.cfg, "llama2")), "L2": lambda: L2Scorer(_path(st.cfg, "xlmr")),
             "SC": lambda: SCScorer(_path(st.cfg, "gpt2"))}[scorer]()
    rows = score_texts(model, units)
    write_jsonl(st.p("E5", "scores", scorer, f"shard_{i:03d}.jsonl"), rows)
    print(f"[mech-score] {scorer} shard {i}/{n}: {len(rows)} texts, {sum(r['scores'] is None for r in rows)} failed")


def load_scores(st, scorer: str) -> dict:
    out = {}
    for f in sorted(st.p("E5", "scores", scorer).glob("shard_*.jsonl")):
        for r in read_jsonl(f):
            out[r["uid"]] = r["scores"]
    return out


def step_meetingbank(st) -> None:
    import pyarrow.parquet as pq
    from .data import meetingbank_path

    t = pq.read_table(meetingbank_path(st.cfg), columns=["prompt", "compressed_prompt"]).to_pydict()
    res = meetingbank_counts(zip(t["prompt"], t["compressed_prompt"]))
    write_json(st.p("E5", "meetingbank_counts.json"), res)
    print(f"[mech-meetingbank] {res['documents']} transcripts, {len(res['words'])} distinct words")


def step_analyze(st) -> None:
    from .config import read_json
    from .study import load_compressed, pair_texts

    cfg, mc = st.cfg, st.cfg["mechanism"]
    items, rendered, comp = st.items(), st.rendered(), load_compressed(st)
    scores = {s: load_scores(st, s) for s in mc["scorers"]}
    expected = len(score_units(st))
    coverage = {s: {"texts": len(v), "expected": expected, "failed": sum(x is None for x in v.values())} for s, v in scores.items()}
    mbf = st.p("E5", "meetingbank_counts.json")
    mb = read_json(mbf) if mbf.exists() else None
    rows = {c: op_table(list(pair_texts(items, rendered, c)), scores, comp, mb, mc["rate"], c) for c in ("squad", "tweets")}
    nw = {t["pos"]: len(sv.tokens(t["aae"])) for t in items["tweets"]}
    med = float(np.median(list(nw.values()))) if nw else None
    if med is not None:
        rows["tweets_short"] = [r for r in rows["tweets"] if nw[r["cluster"]] <= med]
        rows["tweets_long"] = [r for r in rows["tweets"] if nw[r["cluster"]] > med]
    sc = cfg["stats"]
    res = {"rate": mc["rate"], "coverage": coverage, "meetingbank": mb is not None, "median_aae_words": med,
           "results": analyze(rows, sc["bootstrap_resamples"], sc["bootstrap_seed"], mc["quantiles"])}
    write_json(st.p("E5", "results", "mechanism.json"), res)
    write_report(st, res)
    print(f"[mech-analyze] report -> {st.p('E5', 'report', 'E5_report.md')}")


def write_report(st, res: dict) -> None:
    L = ["# E5a mechanism report (main study, exploratory)", "", f"Config SHA-256: `{st.cfg.hash}`. Keep rate {res['rate']}.", "",
         "Each value is the dialect-minus-standard difference for a replaced span, averaged per paragraph (or tweet), "
         "with a 95% cluster-bootstrap interval. Survival differences are for the matching compressor.", "",
         "## Coverage", "", "| Scorer | Texts scored | Expected | Failed |", "|---|---|---|---|"]
    for s, v in res["coverage"].items():
        L.append(f"| {s} | {v['texts']} | {v['expected']} | {v['failed']} |")
    for corpus, r in res["results"].items():
        L += ["", f"## {corpus}", "", "| Feature | Mean difference | 95% interval | Clusters | Spans | Spearman with survival |", "|---|---|---|---|---|---|"]
        for k, v in r.items():
            if v.get("n_clusters"):
                sp = v.get("spearman_op_level")
                L.append(f"| {k} | {v['estimate']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n_clusters']} | {v['n_ops']} | "
                         f"{'' if sp is None else f'{sp:.3f}'} |")
        for k, v in r.items():
            if v.get("survival_by_quantile"):
                L += ["", f"Survival difference by quantile of {k}:", "", "| Quantile | Feature mean | Survival difference | Spans |", "|---|---|---|---|"]
                for q in v["survival_by_quantile"]:
                    L.append(f"| {q['quantile']} | {q['feature_mean']:.4f} | {q['survival_diff_mean']:.4f} | {q['ops']} |")
    path = st.p("E5", "report", "E5_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
