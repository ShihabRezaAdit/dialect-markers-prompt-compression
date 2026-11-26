# Pilot P0 on ARCC2: command runbook (Idea 86)

Laptop commands run in **Git Bash**. Cluster commands run after `ssh arcc2` in **PowerShell**.

| Where | Window | Used for |
|---|---|---|
| Laptop | **Git Bash** | `git commit` / `git push`; `scp` of audit sheets and results |
| Cluster | **PowerShell**, after `ssh arcc2` | `git pull`; every `./run_pilot.sh <step>`; `squeue`, `sacct`, `tail` |

What travels how: **code** goes laptop -> GitHub -> cluster (`git push`, then `git pull`).
**Source texts, rendered dialect data, audit sheets and results** go cluster <-> laptop by `scp`
only, never through GitHub.

Tasks run in order, each as soon as the one before it is done (no fixed days). Every cluster
session starts with:
```
ssh arcc2
P=/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression
cd $P
git pull
```

## Task 1: prepare

1. Environments (login node, about 1 to 2 hours):
   ```
   module load python/3.11.11
   nohup ./run_pilot.sh setup > setup.log 2>&1 &
   tail -f setup.log          # Ctrl+C stops watching; setup keeps running
   ```
   Good: the log ends with `Environments ready ... Locks written to envs/lock/`. Then commit the
   locks from the cluster: `git add envs/lock && git commit -m "Lock environment versions" && git push`
   (on the laptop: `git pull`).
2. Hugging Face token (shared cache with Idea 112): check it is still there:
   ```
   export HF_HOME=/N/lustre/project/proj-606/rezamb/hf_cache
   source /N/lustre/project/proj-606/rezamb/envs/env-gen/bin/activate
   hf auth whoami             # should print ShihabReza; if not: hf auth login (Read token)
   deactivate
   ```
3. Downloads and caches (login node, internet):
   ```
   nohup ./run_pilot.sh download > download.log 2>&1 &
   tail -f download.log
   ./run_pilot.sh warmup
   ```
   Good: `SQuAD train: 87599 questions`, `REAL-AAE: 17479 rows` (or a line saying the UD-TwitterAAE
   fallback was used: tell Claude), one `model ...` line for each of llama2, xlmr, gpt2, M1, M2;
   warmup prints one rendered sentence per dialect.
4. Items and tests:
   ```
   ./run_pilot.sh items       # 300 paragraphs, 600 questions, tweets: REAL-AAE
   ./run_pilot.sh test        # all tests pass
   ```
5. Smoke run (10 paragraphs through every stage). Wait for each submitted job (`squeue -u rezamb`)
   before the next line:
   ```
   ./run_pilot.sh smoke-items
   ./run_pilot.sh render --smoke            # 4 array tasks
   ./run_pilot.sh render-collect --smoke
   ./run_pilot.sh compress --smoke          # 4 GPU jobs
   ./run_pilot.sh prompts --smoke
   ./run_pilot.sh generate --smoke          # 2 GPU jobs
   ./run_pilot.sh g1 --smoke                # 1 array task + 6 GPU jobs
   ./run_pilot.sh analyze --smoke
   ./run_pilot.sh audit-export --sheet warmup_audit --smoke     # optional 5-pair warm-up (guide D4)
   ```
   If a step fails: `tail -n 40 logs/<file>.out`, paste it to Claude, pull the fix, repeat the step.
6. Pre-commitment (after a clean smoke run and your answers to the README's open questions):
   ```
   ./run_pilot.sh precommit
   git add prereg/P0_precommit.md && git commit -m "P0 pre-commitment" && git push
   ```
   From now on `configs/`, the package and the item lists are frozen: rendering, compression and
   generation of pilot items refuse to start if they differ from the pre-commitment.

## Task 2 and 3: rendering, audit, compression

```
./run_pilot.sh render                    # 64 array tasks
./run_pilot.sh render-collect            # "rendered 1800; failed N"
./run_pilot.sh audit-export --sheet p0_audit
./run_pilot.sh compress                  # 4 GPU jobs; runs while you label
```
Download the sheet and pair views (laptop, Git Bash):
```
mkdir -p "/g/Idea 86 - ACL 2027 Research/P0_annotation"
scp "arcc2:/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression/runs/P0/audit/export/p0_audit/p0_audit.xlsx" "/g/Idea 86 - ACL 2027 Research/P0_annotation/"
scp -r "arcc2:/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression/runs/P0/audit/export/p0_audit/pairs" "/g/Idea 86 - ACL 2027 Research/P0_annotation/"
```
The `keys/` folder stays on the cluster so the labels stay blind. Label following the annotation
guide, save, and upload:
```
scp "/g/Idea 86 - ACL 2027 Research/P0_annotation/p0_audit.xlsx" "arcc2:/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression/runs/P0/audit/"
```
On the cluster: `./run_pilot.sh audit-import` (lists every missing or unknown label; nothing is
imported until the sheet is clean).

## Task 4: generation and G1

```
./run_pilot.sh prompts                   # "prompts: 12600 per model"
./run_pilot.sh generate                  # 2 GPU jobs
./run_pilot.sh g1                        # re-runs for gate G1 (1 array task + 6 GPU jobs)
```

## Task 5: analysis and report

```
./run_pilot.sh analyze
```
Download (laptop, Git Bash):
```
mkdir -p "/g/Idea 86 - ACL 2027 Research/P0_results"
scp -r "arcc2:/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression/runs/P0/results" "arcc2:/N/lustre/project/proj-606/rezamb/dialect-markers-prompt-compression/runs/P0/report" "/g/Idea 86 - ACL 2027 Research/P0_results/"
```

## Monitoring

| Command (cluster) | What it does |
|---|---|
| `squeue -u rezamb` | your running and waiting jobs |
| `ls -t logs \| head` | newest job logs first |
| `tail -n 50 logs/<file>.out` | end of a job log (ends with `runtime_seconds`) |
| `sacct -j <jobid> --format=JobID,JobName%24,State,ExitCode,Elapsed` | how a job went (want `COMPLETED 0:0`) |
| `scancel <jobid>` | cancel a job |
| `./run_pilot.sh status` | which pipeline outputs exist |
