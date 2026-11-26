#!/usr/bin/env bash
# Pilot P0 driver (Idea 86). Usage: ./run_pilot.sh <step> [--smoke] [--g1] [--sheet NAME]
# CPU steps run in the current shell (login node); rendering goes to a Slurm CPU array and
# compression and generation to one-GPU Slurm jobs, using cluster.env.
# Steps, in order (README and RUNBOOK_ARCC2.md say who does what):
#   setup          create the five environments and lock their versions (login node, internet)
#   download       SQuAD, REAL-AAE (or the UD-TwitterAAE fallback) and the five pinned models
#   warmup         fill the spaCy/NLTK/Stanza caches and render one sentence per dialect
#   items          freeze the P0 item lists (SQuAD train sample, audit pairs, smoke items, tweets)
#   test           run the unit tests (required by the pre-commitment)
#   smoke-items    write the smoke-run items (10 paragraphs, 20 questions, 10 tweets)
#   render         write render jobs and submit the rendering array      [--smoke] [--g1]
#   render-collect merge the rendering shards                              [--smoke]
#   audit-export   export an audit sheet (--sheet warmup_audit with --smoke; --sheet p0_audit or p0_audit2)
#   compress       write compression jobs and submit L1, L2, LL, SC       [--smoke] [--g1]
#   prompts        random control R, then the generation prompts           [--smoke]
#   generate       submit generation for M1 and M2                         [--smoke] [--g1]
#   g1             write the G1 re-run files and submit the re-runs        [--smoke]
#   precommit      write prereg/P0_precommit.md (then commit and push it)
#   audit-import   import your filled sheet (stops on any unknown or missing label)
#   analyze        measurement, gates, signals, projection and the report  [--smoke]
#   status         which outputs exist                                     [--smoke]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f cluster.env ] || { echo "cluster.env not found"; exit 1; }
# shellcheck disable=SC1091
source cluster.env
for v in PARTITION GPU_GRES ENV_ROOT HF_HOME NLP_DATA; do
  [ -n "${!v:-}" ] || { echo "cluster.env: $v is required"; exit 1; }
done
export HF_HOME HF_HUB_OFFLINE NLTK_DATA="$NLP_DATA/nltk" STANZA_RESOURCES_DIR="$NLP_DATA/stanza"
export PYTHONUNBUFFERED=1   # progress lines reach the logs as they happen
mkdir -p "$NLTK_DATA" "$STANZA_RESOURCES_DIR" logs
CFG="configs/p0.yaml"

step="${1:-}"; shift || true
FLAGS=(); SMOKE=0; G1=0; SHEET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; FLAGS+=(--smoke) ;;
    --g1) G1=1 ;;
    --sheet) SHEET="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
TAG=$([ $SMOKE = 1 ] && echo smoke || echo p0)

act() {  # act <short env name>
  local v="MODULES_$1"; local m="${!v:-${MODULES:-}}"
  [ -z "$m" ] || module load $m
  source "$ENV_ROOT/env-$1/bin/activate"
}
ix() { python -m idea86 "$@" --config "$CFG"; }

qos() { local e=(); [ -n "${ACCOUNT:-}" ] && e+=(--account "$ACCOUNT"); [ -n "${QOS:-}" ] && e+=(--qos "$QOS"); echo "${e[@]}"; }

gpu() {  # gpu <job-name> <env> <idea86 args...>
  local name="$1" env="$2"; shift 2
  sbatch --parsable --job-name "$name" --partition "$PARTITION" --gres "$GPU_GRES" \
    --cpus-per-task "$CPUS_PER_TASK" --mem "$MEM" --time "$([ $SMOKE = 1 ] && echo "${TIME_GPU_SMOKE:-00:30:00}" || echo "$TIME_GPU")" $(qos) \
    --output "logs/%x_%j.out" slurm/gpu_job.sbatch "$env" "$@"
}

cpu_array() {  # cpu_array <job-name> <ntasks> <render args...>
  local name="$1" n="$2"; shift 2
  [ -n "${CPU_PARTITION:-}" ] || { echo "cluster.env: CPU_PARTITION is required for rendering"; exit 1; }
  sbatch --parsable --job-name "$name" --partition "$CPU_PARTITION" --array "0-$((n - 1))" \
    --cpus-per-task "$RENDER_CPUS" --mem "$RENDER_MEM" --time "$TIME_RENDER" $(qos) \
    --output "logs/%x_%A_%a.out" slurm/render_array.sbatch "$n" "$@"
}

case "$step" in
  setup)    bash scripts/setup_envs.sh ;;
  download) ( export HF_HUB_OFFLINE=0; act gen; ix download ) ;;
  warmup)   ( export HF_HUB_OFFLINE=0; act render; ix warmup-render )
            ( export HF_HUB_OFFLINE=0; act compress; ix warmup-compress ) ;;
  items)    ( act gen; ix items ) ;;
  test)     ( act stats; ix test ) ;;
  smoke-items) ( act stats; ix smoke-items ) ;;
  render)   if [ $G1 = 1 ]; then
              j=$(cpu_array "ix_render_g1_$TAG" 1 "${FLAGS[@]}" --g1)
            else
              ( act stats; ix render-jobs "${FLAGS[@]}" )
              n=$([ $SMOKE = 1 ] && echo 4 || grep -E '^\s+array_tasks:' "$CFG" | awk '{print $2}')
              j=$(cpu_array "ix_render_$TAG" "$n" "${FLAGS[@]}")
            fi
            echo "submitted rendering array $j; when all tasks finish: ./run_pilot.sh render-collect ${FLAGS[*]}" ;;
  render-collect) ( act stats; ix render-collect "${FLAGS[@]}" ) ;;
  audit-export) [ -n "$SHEET" ] || { echo "--sheet warmup_audit|p0_audit|p0_audit2 is required"; exit 1; }
            ( act stats; ix audit-export --sheet "$SHEET" "${FLAGS[@]}" ) ;;
  compress) [ $G1 = 1 ] || ( act stats; ix compress-jobs "${FLAGS[@]}" )
            G=(); [ $G1 = 1 ] && G=(--g1)
            ids=""
            for c in L1 L2 LL; do ids="$ids $(gpu "ix_${c}_$TAG${G:+_g1}" compress compress --compressor $c "${FLAGS[@]}" "${G[@]}")"; done
            ids="$ids $(gpu "ix_SC_$TAG${G:+_g1}" sc compress --compressor SC "${FLAGS[@]}" "${G[@]}")"
            echo "submitted compression jobs:$ids" ;;
  prompts)  ( act stats; ix random-control "${FLAGS[@]}"; ix prompts "${FLAGS[@]}" ) ;;
  generate) G=(); [ $G1 = 1 ] && G=(--g1)
            ids=""
            for m in M1 M2; do ids="$ids $(gpu "ix_gen_${m}_$TAG${G:+_g1}" gen generate --model $m "${FLAGS[@]}" "${G[@]}")"; done
            echo "submitted generation jobs:$ids" ;;
  g1)       ( act stats; ix g1-jobs "${FLAGS[@]}" )
            "$0" render "${FLAGS[@]}" --g1
            "$0" compress "${FLAGS[@]}" --g1
            "$0" generate "${FLAGS[@]}" --g1 ;;
  precommit) ( act stats; ix precommit ) ;;
  audit-import) ( act stats; ix audit-import --sheet "${SHEET:-p0_audit}" "${FLAGS[@]}" ) ;;
  analyze)  ( act stats; ix analyze "${FLAGS[@]}" ) ;;
  status)   ( act stats; ix status "${FLAGS[@]}" ) ;;
  *) sed -n '2,23p' "$0"; exit 1 ;;
esac
