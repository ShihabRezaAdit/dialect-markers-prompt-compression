# Pilot P0: Does Prompt Compression Remove Dialect Markers? (Idea 86)

Code for the pilot defined in `Research Proposals/Pilot_Plan` (binding) and summarized in
`Research Proposals/Internal_Execution_Plan`, Section 14. Every value the plans fix (seed 86,
samples, rates, revisions, thresholds, the prompt) lives in `configs/p0.yaml` and in the code
named below; nothing is chosen at run time.

## What the pilot runs

| Stage | What | Where |
|---|---|---|
| Items | 300 SQuAD v1.1 **train** paragraphs (600 questions), 100 audit pairs, 10 smoke paragraphs, 200 REAL-AAE rows (or 50 + 50 UD-TwitterAAE tweets) | login node |
| Rendering | Multi-VALUE value-nlp 0.3, AAVE and CollSgE, `set_seed(86 + k)`, rule logs: 1,800 renderings | CPU array, 64 tasks |
| Audit | 100 pairs (50 per dialect), your labels, gate G5 | you, about 3.5 h |
| Compression | Arm A: S, A, C passages with L1, L2, SC at rho 0.5, R matched to L2; Arm B: SAE passages with LL conditioned on each question rendering, R matched to LL; tweets with L1, L2, SC, R | 4 GPU jobs + CPU |
| Generation | Llama-3.1-8B-Instruct (M1) and Qwen2.5-14B-Instruct (M2), 600 questions x 21 prompts = 25,200 generations (1,200 served from the cache) | 2 GPU jobs |
| G1 re-runs | 20 paragraphs per dialect re-rendered; 20 texts per compressor re-compressed; 200 prompts per model re-generated | CPU + GPU |
| Analysis | gates G1 to G5, signals S1 and S2, the recommended outcome, preliminary results, projection, report | login node |

## The pilot checks in plain words

**Gates** ask "can we trust the pipeline?" A failed gate is fixed and re-checked; it never stops
the project on its own. **Signals** ask "is an effect large enough to lead the paper?"

| ID | Plain question | Met when |
|---|---|---|
| G1 | Is everything reproducible? | Re-rendering 40 of 40 identical; re-compressing 80 of 80 identical; re-generating >= 196 of 200 same answers per model |
| G2 | Do compressors hit their budget? | Mean achieved keep rate in [0.35, 0.65] per compressor and rendering; dialect and SAE means within 0.05 |
| G3 | Does the diff find the rule changes? | >= 90% of rule-log spans in the 100 audited paragraphs overlap a diff op |
| G4 | Is random deletion neutral? | abs(Delta_s) under R pooled over A and C <= 0.03 |
| G5 | Are the renderings faithful? | Your CONTENT_PRESERVED = YES in >= 43 of 50 pairs per dialect, and >= 0.60 of pilot questions answer-preserved per dialect |
| S1 | Retention effect? | abs(Delta_s) >= 0.04 and the 99.17% interval excludes 0 in at least one of {L1, L2, SC} x {A, C} |
| S2 | Answer-survival effect? | abs(Delta_a) >= 0.03 and the 97.5% interval excludes 0 for LL in A or C |

Recommendation once G1 to G5 pass: S1 and S2 -> Go. S1 only -> Go, retention focus. S2 only ->
Go, realistic-pipeline focus. Neither -> No-go. The rule only recommends; the decision is taken
with the advisor.

## Folder layout

```
configs/p0.yaml             all pilot settings (frozen by the pre-commitment)
src/idea86/                 the package: python -m idea86 <command>
  data.py render.py compress.py survival.py generate.py score.py metrics.py stats.py
  audit.py pilot.py precommit.py config.py cli.py
tests/                      unit tests and end-to-end dry runs (python tests/run_tests.py)
envs/                       requirements of the five environments; envs/lock/ gets exact versions
scripts/setup_envs.sh       creates the environments
slurm/                      render_array.sbatch (CPU array), gpu_job.sbatch (one GPU)
run_pilot.sh                step driver
cluster.env                 ARCC2 settings (reused from Idea 112)
prereg/                     pilot_oc.py (the pilot plan's Appendix A script); P0_precommit.md goes here
data/ runs/ logs/           created on the cluster; never committed
```

## Steps on the cluster

The exact commands, with what a good result looks like, are in `RUNBOOK_ARCC2.md`.

Task 1, prepare:
1. `./run_pilot.sh setup` (login node; then commit `envs/lock/`).
2. `./run_pilot.sh download` (SQuAD, REAL-AAE or the fallback, five models), `./run_pilot.sh warmup`.
3. `./run_pilot.sh items`, `./run_pilot.sh test`.
4. Smoke run: `smoke-items`, then `render`, `render-collect`, `compress`, `prompts`, `generate`,
   `g1`, `analyze`, all with `--smoke`. Optional audit warm-up: `audit-export --sheet warmup_audit --smoke`.
5. `./run_pilot.sh precommit`, then commit and push `prereg/P0_precommit.md` (the frozen
   annotation guide's SHA-256 is already in `configs/p0.yaml`).

Task 2, render and audit: `render`, `render-collect`, `audit-export --sheet p0_audit`; you label the 100 pairs.
Task 3, compress and measure: `compress`, as soon as rendering is collected (while you label).
Task 4, generate: `prompts`, `generate`, `g1`; then `audit-import` once your sheet is back.
Task 5, analyze and report: `analyze`; the report goes to the advisor (gate fixes, if any, come
first). Then the go/no-go meeting. Each task starts as soon as the one before it is done.

## What has and has not been tested

- **Tested here** (Linux, no GPU, no internet): 35 tests pass, covering tokens and
  diff ops, rule-log typing (inclusive end offsets), word survival with partial words, paired span
  survival, the stratified ratio, SQuAD scoring, ReDial answer extraction, Holm, the bootstrap,
  signals, Wilson and kappa, item sampling and disjointness, the random control, the audit sheet
  and pair views (no rule names, gold answer on the SAE side only), import checks (blank,
  unknown, missing note, D3), and end-to-end dry runs of every CPU step of P0 and of the smoke
  run on synthetic data with fake renderings, compressions and answers.
- **Not tested here:** anything that needs the cluster, a GPU or the internet: downloads,
  value-nlp rendering, llmlingua and selective-context compression, vLLM generation. The library
  calls follow each package's source (value-nlp 0.3, llmlingua 0.2.2 and selective-context 0.1.4
  were inspected) but could not be run. The smoke run is where these get tested; expect to fix
  small API differences there, before the pre-commitment.

## Open questions (please confirm before the pre-commitment)

Decided by the project lead:
1. **CPU partition.** Rendering runs on `public` (where Idea 112 ran its CPU job with QOS proj-606).
2. **Plan sources in the pre-commitment.** The two plans are not in git, so their SHA-256 is
   recorded instead of a commit hash (`plan_sources` in `configs/p0.yaml`).
3. **Annotation guide copy.** Frozen as `P0_annotation/Idea86_Annotation_Guide.pdf` in the
   project folder (includes D1: word-diff highlighting only); its SHA-256 is in `configs/p0.yaml`.

Implementation choices the plans leave open: 4 to 14 confirmed by the project lead (#8 with the
condition that partial words count as not kept in the primary result) and recorded in the
internal plan's decisions log:
4. **Random control R** deletes the unchosen word tokens from the string and keeps punctuation
   (`compress.random_delete`).
5. **LongLLMLingua output.** llmlingua 0.2.2 appends the question to the compressed prompt; it is
   stripped so the Arm B passage contains no question (`compress.LLMLinguaFamily.ll`).
6. **Selective Context and GPT-2.** The package loads `gpt2` by name with no revision argument;
   a subclass loads the same model from the pinned snapshot (`compress.make_selective_context`).
7. **Rule-log offsets.** value-nlp 0.3 rule spans end inclusively; an insert op is typed by a rule
   whose span ends right before it (`survival.type_op`).
8. **Partial words** are words not kept that share at least two characters, as a substring, with
   a compressed token in the same replace block (`survival.word_survival`). They count as **not
   kept** in the primary retention result (`partial_as_kept=False`, the default in
   `survival.survived`, `metrics.delta_s_text` and `metrics.ratio_counts_text`; S1 and G4 use it).
   "Partial counted as kept" is computed only as a sensitivity analysis
   (`pilot.measure`, `retention_partial_kept`) and reported in its own section of the report.
9. **Frequency deciles** for the ratio r are cut over all SAE word occurrences of the corpus
   (`survival.FreqStrata`).
10. **Smoke items** are the paragraphs at positions 300 to 309 of the same seed-86 shuffle (not
    pilot items, not study items); the audit warm-up uses 5 of them.
11. **Audit sample.** One seed-86 permutation of the 300 pilot paragraphs: the first 50 are AAVE
    pairs, the next 50 CollSgE pairs; the question of each pair is drawn with the same generator.
12. **G1 samples.** The same 20 paragraphs are re-rendered in both dialects; 20 Arm A texts are
    re-compressed per compressor (20 Arm B passages for LL); 200 prompts are re-generated per model.
13. **Bootstrap.** Percentile intervals; two-sided p = 2 min(P*[theta <= 0], P*[theta >= 0]).
14. **Environment pins.** value-nlp and selective-context pin old spaCy releases that need
    numpy 1.x, so `envs/env-render.txt` and `envs/env-sc.txt` add `numpy<2`. Stanza 1.8.2 (pinned
    by value-nlp) cannot load its models under torch 2.6 or later, so `envs/env-render.txt` also
    pins `torch<2.6`, `transformers>=4.40,<4.50` and `peft<0.15` (decisions log).

Decided during pilot preparation (smoke run; decisions log, internal plan):

15. **Sentence fallback.** A passage that Multi-VALUE cannot render whole is rendered sentence by
    sentence (NLTK Punkt), with the SAE text between sentences kept exactly (`render_passage`).
16. **No-coreference last resort.** A sentence that still fails the spaCy/Stanza alignment check
    in Multi-VALUE's coreference step is rendered with coreference off for that sentence only.
    Every text is tagged `whole`, `sentence_fallback`, `no_coref` or `failed`; `render-collect`
    and the report give the counts per dialect and the features that read coreference; the
    report adds sensitivity analyses without no-coref passages and without any fallback passages.
