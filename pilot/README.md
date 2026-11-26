# Pilot P0 (frozen archive)

The pilot P0 is complete. Its exact, runnable state is the git tag `pilot-final`
(`git checkout pilot-final`). The files in this folder are the pilot-only parts of that state,
kept for the record: configuration (`configs/p0.yaml`), pre-commitment (`prereg/`), run script
and runbook, the pilot README (`README_P0.md`), follow-up scripts, the pilot driver, pre-commitment and command-line modules
(`src/`), and the pilot end-to-end test (`tests/`).

Nothing in this folder is imported by the package `idea86` or read by the main study. On the
cluster, the pilot outputs live in `runs/pilot_archive/`; the main-study code refuses to open
any file there or in this folder (`src/idea86/isolation.py`, tested by
`tests/test_isolation.py`).
