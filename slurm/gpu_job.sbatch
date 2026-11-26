#!/usr/bin/env bash
# Generic one-GPU job (compression and mechanism scoring). Submitted by run_main.sh, which passes
# partition, QOS, GPU request, time and, for sharded steps, --array from cluster.env and the config.
# Inside an array, SLURM_ARRAY_TASK_ID is the shard index.
# Usage (via run_main.sh): sbatch ... slurm/gpu_job.sbatch <env short name> <idea86 args...>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster.env
export HF_HOME HF_HUB_OFFLINE NLTK_DATA="$NLP_DATA/nltk" STANZA_RESOURCES_DIR="$NLP_DATA/stanza"
SHORT="$1"; shift
MV="MODULES_${SHORT}"; MODS="${!MV:-${MODULES:-}}"
[ -z "$MODS" ] || module load $MODS
source "$ENV_ROOT/env-$SHORT/bin/activate"
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname) env=env-$SHORT args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
start=$(date +%s)
python -m idea86 "$@" --config "${IDEA86_CONFIG:-configs/main.yaml}"
echo "runtime_seconds=$(( $(date +%s) - start ))"
