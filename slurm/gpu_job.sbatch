#!/usr/bin/env bash
# Generic one-GPU job (compression and generation). Submitted by run_pilot.sh, which passes
# partition, QOS, GPU request and time from cluster.env.
# Usage (via run_pilot.sh): sbatch ... slurm/gpu_job.sbatch <env short name> <idea86 args...>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster.env
export HF_HOME HF_HUB_OFFLINE NLTK_DATA="$NLP_DATA/nltk" STANZA_RESOURCES_DIR="$NLP_DATA/stanza"
SHORT="$1"; shift
MV="MODULES_${SHORT}"; MODS="${!MV:-${MODULES:-}}"
[ -z "$MODS" ] || module load $MODS
source "$ENV_ROOT/env-$SHORT/bin/activate"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) env=env-$SHORT args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
start=$(date +%s)
python -m idea86 "$@" --config configs/p0.yaml
echo "runtime_seconds=$(( $(date +%s) - start ))"
