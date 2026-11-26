#!/usr/bin/env bash
# Main-study driver (Idea 86). Usage: ./run_main.sh <step> [options]
# CPU steps run in the current shell (login node); rendering goes to a Slurm CPU array, compression
# and mechanism scoring to one-GPU Slurm arrays (one task per shard), using cluster.env.
# The pilot is archived separately (git tag pilot-final); nothing here reads it.
# Steps, in order (README.md says who does what):
#   setup            create the five environments and lock their versions (login node, internet)
#   download         SQuAD validation, REAL-AAE (SHA-256 checked), MeetingBank-LLMCompressed, the three compressor models
#   warmup           fill the spaCy/NLTK/Stanza caches and render one sentence
#   test             run the unit tests (required by the pre-commitment)
#   items            freeze the E1 item lists (SQuAD validation, REAL-AAE study pairs, audit draw)
#   precommit        write prereg/prereg_main.md (then commit and push it)
#   render           write render jobs and submit the rendering array
#   render-collect   merge the rendering shards
#   audit-export     --sheet e1_audit, or --sheet e1_relabel (at least 7 days after the e1_audit import)
#   audit-import     --sheet e1_audit | e1_relabel (stops on any unknown or missing label)
#   audit-report     audit shares, strict D5 counts, re-label agreement
#   compress         write compression jobs and submit L1, L2 and SC as GPU arrays
#   compress-shard   re-submit one shard: ./run_main.sh compress-shard <L1|L2|SC> <index>
#   compress-collect merge the compression shards
#   analyze-e2       random control R, then retention, families and the E2 report
#   mech             submit the mechanism scorers L1, L2, SC as GPU arrays
#   mech-shard       re-submit one shard: ./run_main.sh mech-shard <L1|L2|SC> <index>
#   mech-analyze     MeetingBank counts, then the E5 tables and report
#   status           which outputs exist
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
export PYTHONUNBUFFERED=1
export IDEA86_CONFIG="configs/main.yaml"
mkdir -p "$NLTK_DATA" "$STANZA_RESOURCES_DIR" logs
CFG="$IDEA86_CONFIG"

step="${1:-}"; shift || true
SHEET=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sheet) SHEET="$2"; shift ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

act() {  # act <short env name>
  local v="MODULES_$1"; local m="${!v:-${MODULES:-}}"
  [ -z "$m" ] || module load $m
  source "$ENV_ROOT/env-$1/bin/activate"
}
ix() { python -m idea86 "$@" --config "$CFG"; }
qos() { local e=(); [ -n "${ACCOUNT:-}" ] && e+=(--account "$ACCOUNT"); [ -n "${QOS:-}" ] && e+=(--qos "$QOS"); echo "${e[@]}"; }

shards() {  # shards <section> <name>: number of shards in the config (the "shards: {...}" line of that section)
  awk -v sec="$1:" -v key="$2" '
    /^[^ #]/ { insec = ($1 == sec) }
    insec && /^  shards:/ { if (match($0, key ": *[0-9]+")) { s = substr($0, RSTART, RLENGTH); sub(/.*: */, "", s); print s; exit } }
  ' "$CFG"
}

gpu_array() {  # gpu_array <job-name> <array spec> <env> <idea86 args...>
  local name="$1" arr="$2" env="$3"; shift 3
  sbatch --parsable --job-name "$name" --partition "$PARTITION" --gres "$GPU_GRES" --array "$arr" \
    --cpus-per-task "$CPUS_PER_TASK" --mem "$MEM" --time "$TIME_GPU" $(qos) \
    --output "logs/%x_%A_%a.out" slurm/gpu_job.sbatch "$env" "$@"
}

cpu_array() {  # cpu_array <job-name> <ntasks>
  local name="$1" n="$2"; shift 2
  [ -n "${CPU_PARTITION:-}" ] || { echo "cluster.env: CPU_PARTITION is required for rendering"; exit 1; }
  sbatch --parsable --job-name "$name" --partition "$CPU_PARTITION" --array "0-$((n - 1))" \
    --cpus-per-task "$RENDER_CPUS" --mem "$RENDER_MEM" --time "$TIME_RENDER" $(qos) \
    --output "logs/%x_%A_%a.out" slurm/render_array.sbatch "$n" "$@"
}

envof() { [ "$1" = SC ] && echo sc || echo compress; }

case "$step" in
  setup)    bash scripts/setup_envs.sh ;;
  download) ( export HF_HUB_OFFLINE=0; act gen; ix download ) ;;
  warmup)   ( export HF_HUB_OFFLINE=0; act render; ix warmup-render )
            ( export HF_HUB_OFFLINE=0; act compress; ix warmup-compress ) ;;
  test)     ( act stats; ix test ) ;;
  items)    ( act gen; ix items ) ;;
  precommit) ( act stats; ix precommit ) ;;
  render)   ( act stats; ix render-jobs )
            n=$(grep -E '^\s+array_tasks:' "$CFG" | awk '{print $2}')
            j=$(cpu_array ix_E1_render "$n")
            echo "submitted rendering array $j ($n tasks); when all finish: ./run_main.sh render-collect" ;;
  render-collect) ( act stats; ix render-collect ) ;;
  audit-export|audit-import)
            [ -n "$SHEET" ] || { echo "--sheet e1_audit or e1_relabel is required"; exit 1; }
            ( act stats; ix "$step" --sheet "$SHEET" ) ;;
  audit-report) ( act stats; ix audit-report ) ;;
  compress) ( act stats; ix compress-jobs )
            for c in L1 L2 SC; do
              n=$(shards compress $c)
              j=$(gpu_array "ix_E2_$c" "0-$((n - 1))" "$(envof $c)" compress --compressor $c --nshards "$n")
              echo "submitted $c: array $j ($n shards)"
            done
            echo "when all finish: ./run_main.sh compress-collect" ;;
  compress-shard) c="${ARGS[0]:?compressor}"; i="${ARGS[1]:?shard index}"; n=$(shards compress "$c")
            j=$(gpu_array "ix_E2_$c" "$i" "$(envof "$c")" compress --compressor "$c" --nshards "$n")
            echo "submitted $c shard $i: job $j" ;;
  compress-collect) ( act stats; ix compress-collect ) ;;
  analyze-e2) ( act stats; ix random-control; ix analyze-e2 ) ;;
  mech)     for s in L1 L2 SC; do
              n=$(shards mechanism $s)
              j=$(gpu_array "ix_E5_$s" "0-$((n - 1))" "$(envof $s)" mech-score --scorer $s --nshards "$n")
              echo "submitted scorer $s: array $j ($n shards)"
            done
            echo "when all finish: ./run_main.sh mech-analyze" ;;
  mech-shard) s="${ARGS[0]:?scorer}"; i="${ARGS[1]:?shard index}"; n=$(shards mechanism "$s")
            j=$(gpu_array "ix_E5_$s" "$i" "$(envof "$s")" mech-score --scorer "$s" --nshards "$n")
            echo "submitted scorer $s shard $i: job $j" ;;
  mech-analyze) ( act gen; ix mech-meetingbank ) && ( act stats; ix mech-analyze ) ;;
  status)   ( act stats; ix status ) ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
