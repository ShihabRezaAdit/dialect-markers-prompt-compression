# Does Prompt Compression Remove Dialect Markers? Main study (Idea 86)

Code for the main study defined in `Research Proposals/Internal_Execution_Plan` (binding) and
`Professor_Proposal`. Every value the plans fix lives in `configs/main.yaml`; nothing is chosen at
run time. The pilot P0 is complete and archived: its exact state is the git tag `pilot-final`, its
pilot-only files are in `pilot/`, and its outputs on the cluster are in `runs/pilot_archive/`. The
main study never reads any of them: `src/idea86/isolation.py` refuses every such file access, and
`tests/test_isolation.py` fails if any main-study step opens one.

## What runs now (must-have tier)

| Experiment | What | Where | Outputs |
|---|---|---|---|
| E1 | SQuAD v1.1 validation (2,067 paragraphs, 10,570 questions) rendered into AAVE with the meaning-preserving feature subset; 150-pair audit, 20 pairs re-labeled at least 7 days later | CPU array (64 tasks), you | `runs/E1/` |
| E2 | Retention: SQuAD AAVE and the 17,279 non-pilot REAL-AAE pairs, keep rates 0.75, 0.5, 0.33, compressors L1, L2, SC and R (matched to L2) | GPU arrays (L1 24, L2 8, SC 16 shards) | `runs/E2/` |
| E5a | Mechanism (exploratory): each compressor's own token scores and LLMLingua-2's training data (MeetingBank-LLMCompressed), related to survival | GPU arrays (L1 8, L2 2, SC 8 shards), CPU | `runs/E5/` |

Confirmatory family F1: Delta_s for L1, L2 and SC on SQuAD AAVE at keep rate 0.5 (Holm); REAL-AAE
is the replication family; H1c: the 90% interval of R lies inside +-0.03. The should-have tier
(E3 accuracy and flips, CoQA, E4, E6) is added after the must-have runs.

## Order

All steps go through `./run_main.sh <step>` on the ARCC2 login node (see the step list at the top
of the script): `setup` (once), `download`, `warmup`, `test`, `items`, `precommit` (commit and push
`prereg/prereg_main.md`), then `render`, `render-collect`, `audit-export --sheet e1_audit`, your
labels, `audit-import --sheet e1_audit`, `compress`, `compress-collect`, `analyze-e2`, `mech`,
`mech-analyze`, and after at least 7 days `audit-export --sheet e1_relabel`, `audit-import --sheet
e1_relabel`, `audit-report`. `./run_main.sh status` shows which outputs exist. Rendering, compression
and scoring verify the pre-commitment first and refuse to start if anything differs.

Source texts, rendered texts and annotation sheets are never committed; they move by scp.
