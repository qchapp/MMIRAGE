#!/usr/bin/env bash
# Run every fast-run experiment end to end: preflight, then
# smoke -> calibrate -> overhead -> scaling -> recovery -> text -> vlm.
# Stages are isolated and log to experiments/run_all_logs/<stage>.log.
# Usage: run_all.sh [--only stage,stage] [--skip stage,stage]
# Requirements: see "Running everything unattended" in experiments/README.md.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$REPO_ROOT/experiments/run_all_logs"
mkdir -p "$LOG_DIR"

ALL_STAGES=(smoke calibrate overhead scaling recovery text vlm)
ONLY="" SKIP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

want_stage() {
  local stage="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$stage,"* ]] || return 1; fi
  [[ ",$SKIP," == *",$stage,"* ]] && return 1
  return 0
}

preflight() {
  local fail=0
  if ! python -c "import sglang" 2>/dev/null; then
    echo "preflight: 'python' cannot import sglang - activate the MMIRAGE venv" \
         "and install the GPU extra first (root README)." >&2
    fail=1
  fi
  local gpus
  gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
  if [[ "$gpus" -lt 4 ]]; then
    echo "preflight: need 4 visible GPUs, found ${gpus:-0}." >&2
    fail=1
  fi
  if [[ -z "${HF_TOKEN:-}" ]] && ! python -c "
from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
    echo "preflight: no Hugging Face token (set HF_TOKEN or 'hf auth login')." >&2
    fail=1
  fi
  for var in MMIRAGE_DATATROVE_PYTHON MMIRAGE_NEMO_CURATOR_PYTHON; do
    if want_stage overhead || want_stage text || want_stage vlm; then
      if [[ ! -x "${!var}" ]]; then
        echo "preflight: $var=${!var} is not executable - build the competitor" \
             "venvs (experiments/single_node_h100_scaling/environment/) or" \
             "point $var at the right interpreter." >&2
        fail=1
      fi
    fi
  done
  return "$fail"
}

stage_smoke() {
  python experiments/smoke/run_smoke.py --shared-root "$MMIRAGE_RECOVERY_ROOT"
}

stage_calibrate() {
  python experiments/smoke/calibrate.py --apply
  python experiments/raw_sglang_overhead/scripts/prepare_workload.py \
    --output-dir experiments/raw_sglang_overhead/workload
  python experiments/single_node_h100_scaling/scripts/prepare_workload.py \
    --output-dir experiments/single_node_h100_scaling/workload
  python experiments/shard_recovery/scripts/prepare_workload.py \
    --output-root "$MMIRAGE_RECOVERY_ROOT"
  python experiments/task_comparison/text_shortening/scripts/prepare_workload.py \
    --output-dir experiments/task_comparison/text_shortening/workload
  python experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py \
    --output-dir experiments/task_comparison/vlm_enrichment/workload
}

stage_overhead() {
  rm -rf experiments/raw_sglang_overhead/results
  python experiments/raw_sglang_overhead/scripts/run.py \
    --workload-dir experiments/raw_sglang_overhead/workload \
    --output-dir experiments/raw_sglang_overhead/results \
    --frameworks raw_sglang,mmirage_sglang,datatrove,nemo_curator \
    --repetitions 1
}

stage_scaling() {
  rm -rf experiments/single_node_h100_scaling/results
  local cfg
  for cfg in execution_1gpu execution_2gpu execution_4gpu; do
    python experiments/single_node_h100_scaling/scripts/run.py \
      --execution-config "experiments/single_node_h100_scaling/configs/$cfg.yaml"
  done
}

stage_recovery() {
  rm -rf "$MMIRAGE_RECOVERY_ROOT/runs"
  python experiments/shard_recovery/scripts/run_local.py run-condition \
    --condition baseline --rep 1 --shared-root "$MMIRAGE_RECOVERY_ROOT" \
    --max-active-shards 4 --gpu-ids 0,1,2,3 --overwrite
  mmirage merge-dir \
    --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/baseline/rep_01/output" \
    --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/baseline/rep_01/merged"
  local cond
  for cond in fail_1 fail_4 fail_8; do
    python experiments/shard_recovery/scripts/run_local.py run-condition \
      --condition "$cond" --rep 1 --shared-root "$MMIRAGE_RECOVERY_ROOT" \
      --max-active-shards 4 --gpu-ids 0,1,2,3 --overwrite
    python experiments/shard_recovery/scripts/run_local.py retry \
      --condition "$cond" --rep 1 --shared-root "$MMIRAGE_RECOVERY_ROOT" \
      --max-active-shards 4 --gpu-ids 0,1,2,3
    mmirage merge-dir \
      --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/$cond/rep_01/output" \
      --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/$cond/rep_01/merged"
  done
  python experiments/shard_recovery/scripts/extract_results.py \
    --shared-root "$MMIRAGE_RECOVERY_ROOT" \
    --conditions baseline,fail_1,fail_4,fail_8 \
    --reps 1 \
    --config "$REPO_ROOT/experiments/shard_recovery/configs/mmirage_recovery.yaml"
}

stage_text() {
  rm -rf experiments/task_comparison/text_shortening/results
  python experiments/single_node_h100_scaling/scripts/run.py \
    --execution-config experiments/task_comparison/text_shortening/configs/execution_4gpu.yaml
  "$MMIRAGE_DATATROVE_PYTHON" experiments/single_node_h100_scaling/scripts/run_datatrove_scaling.py \
    --workload-jsonl experiments/task_comparison/text_shortening/workload/workload.jsonl \
    --output-root experiments/task_comparison/text_shortening/results/native_competitors/datatrove \
    --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3 --model Qwen/Qwen3-4B
  "$MMIRAGE_NEMO_CURATOR_PYTHON" experiments/single_node_h100_scaling/scripts/run_nemo_curator_scaling.py \
    --workload-jsonl experiments/task_comparison/text_shortening/workload/workload.jsonl \
    --output-root experiments/task_comparison/text_shortening/results/native_competitors/nemo_curator \
    --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3 --model Qwen/Qwen3-4B
}

stage_vlm() {
  rm -rf experiments/task_comparison/vlm_enrichment/results
  python experiments/task_comparison/vlm_enrichment/scripts/run_mmirage_vlm.py \
    --execution-config experiments/task_comparison/vlm_enrichment/configs/execution_4gpu.yaml
  local fw worker
  for fw in sglang datatrove nemo_curator; do
    case "$fw" in
      sglang) worker="" ;;
      datatrove) worker="$MMIRAGE_DATATROVE_PYTHON" ;;
      nemo_curator) worker="$MMIRAGE_NEMO_CURATOR_PYTHON" ;;
    esac
    python experiments/task_comparison/vlm_enrichment/scripts/run_native_vlm_competitor.py \
      --framework "$fw" \
      --workload-jsonl experiments/task_comparison/vlm_enrichment/workload/rows.jsonl \
      --image-base-path experiments/task_comparison/vlm_enrichment/workload \
      --output-root "experiments/task_comparison/vlm_enrichment/results/native_competitors/$fw" \
      --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3 \
      ${worker:+--worker-python "$worker"}
  done
}

declare -A STATUS DURATION
overall=0

if ! preflight; then
  echo "run_all: preflight failed, nothing was run." >&2
  exit 1
fi

for stage in "${ALL_STAGES[@]}"; do
  want_stage "$stage" || { STATUS[$stage]="skipped"; continue; }
  log="$LOG_DIR/$stage.log"
  echo "run_all: [$stage] started $(date -u +%H:%M:%S) (log: $log)"
  start=$SECONDS
  if (set -euxo pipefail; "stage_$stage") >"$log" 2>&1; then
    STATUS[$stage]="ok"
  else
    STATUS[$stage]="FAILED"
    overall=1
    echo "run_all: [$stage] FAILED - last lines of $log:" >&2
    tail -5 "$log" >&2
  fi
  DURATION[$stage]=$((SECONDS - start))
  echo "run_all: [$stage] ${STATUS[$stage]} after ${DURATION[$stage]}s"
done

echo
echo "stage      status   seconds"
echo "---------  -------  -------"
for stage in "${ALL_STAGES[@]}"; do
  printf "%-9s  %-7s  %s\n" "$stage" "${STATUS[$stage]:-skipped}" "${DURATION[$stage]:-—}"
done
exit "$overall"
