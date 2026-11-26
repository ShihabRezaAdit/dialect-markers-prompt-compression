#!/usr/bin/env bash
# Rendering as a Slurm CPU array (internal plan Section 30): each task renders a contiguous block
# of the job list and writes its own shard. Submitted by run_main.sh with --array and the CPU
# partition from cluster.env.
# Usage (via run_main.sh): sbatch --array=0-<N-1> ... slurm/render_array.sbatch <N>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster.env
export HF_HOME HF_HUB_OFFLINE NLTK_DATA="$NLP_DATA/nltk" STANZA_RESOURCES_DIR="$NLP_DATA/stanza"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
NTASKS="$1"; shift
[ -z "${MODULES_render:-}" ] || module load $MODULES_render
source "$ENV_ROOT/env-render/bin/activate"
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0}/$NTASKS on $(hostname) args=$*"
start=$(date +%s)
python -m idea86 render --ntasks "$NTASKS" "$@" --config "${IDEA86_CONFIG:-configs/main.yaml}"
echo "runtime_seconds=$(( $(date +%s) - start ))"
