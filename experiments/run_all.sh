#!/usr/bin/env bash
# Run every fast-run experiment end to end in one pass: preflight, then
# smoke -> calibrate -> scaling -> recovery -> text -> vlm.
#
# This is the single entry point for the whole A matrix on one 4-GPU pod:
# scaling runs all three GPU points (1/2/4), recovery runs the MMIRAGE and
# native (raw_sglang/datatrove/nemo_curator/distilabel/ray_data_llm) cells,
# and text_shortening + vlm_enrichment run the Comparison B tasks. There is
# no pod_a / pod_b separation here - everything runs together.
#
# Stages are isolated and log to experiments/run_all_logs/<stage>.log.
# Usage: run_all.sh [--only stage,stage] [--skip stage,stage] [--rerun-reused]
# Requirements: see "Running everything unattended" in experiments/README.md.
#
# The scaling/recovery/text/vlm stages delegate to
# experiments/a_matrix/scripts/run_setup.py, which schedules the per-GPU-point
# and per-framework units and pinpoints GPUs. The optional --pod pod_a / --pod
# pod_b flags of run_setup.py are only for splitting the same work across two
# nodes manually; run_all.sh always runs the full union on one pod.
#
# By default the MMIRAGE-only cells already covered by the 2026-08-15 fast-runs
# reproduction (gpu_scaling mmirage 1/2/4 GPU, text mmirage, vlm mmirage) are
# NOT rerun: run_setup.py is invoked with --reuse-fastruns, their previous
# results are preserved, and README "Reusing the 2026-08-15 fast-runs" lists
# how to restore them from the archive. Pass --rerun-reused to run those cells
# again from scratch (also wipes their previous results).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export MMIRAGE_DISTILABEL_PYTHON="${MMIRAGE_DISTILABEL_PYTHON:-$REPO_ROOT/.venv-distilabel/bin/python}"
export MMIRAGE_RAY_DATA_LLM_PYTHON="${MMIRAGE_RAY_DATA_LLM_PYTHON:-$REPO_ROOT/.venv-ray_data_llm/bin/python}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$REPO_ROOT/experiments/run_all_logs"
mkdir -p "$LOG_DIR"

ALL_STAGES=(smoke calibrate scaling recovery text vlm)
ONLY="" SKIP="" RERUN_REUSED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --rerun-reused) RERUN_REUSED=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --reuse-fastruns tells run_setup.py to skip cells satisfied by the fast-runs
# archive; --rerun-reused drops it so every cell runs from scratch.
REUSE_FLAG="--reuse-fastruns"
if [[ "$RERUN_REUSED" == 1 ]]; then
  REUSE_FLAG=""
  echo "run_all: --rerun-reused - every cell (including the fast-runs-reused MMIRAGE cells) will be rerun."
fi

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
  for var in MMIRAGE_DATATROVE_PYTHON MMIRAGE_NEMO_CURATOR_PYTHON MMIRAGE_DISTILABEL_PYTHON MMIRAGE_RAY_DATA_LLM_PYTHON; do
    if want_stage scaling || want_stage recovery || want_stage text || want_stage vlm; then
      if [[ ! -x "${!var}" ]]; then
        echo "preflight: $var=${!var} is not executable - build the competitor" \
             "venvs (experiments/single_node_h100_scaling/environment/ or" \
             "experiments/nemo_curator_comparison/environment/) or" \
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
  python experiments/a_matrix/scripts/prepare_workload.py \
    --output-dir experiments/a_matrix/workload \
    --shared-root "$MMIRAGE_RECOVERY_ROOT"
  python experiments/task_comparison/text_shortening/scripts/prepare_workload.py \
    --output-dir experiments/task_comparison/text_shortening/workload
  python experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py \
    --output-dir experiments/task_comparison/vlm_enrichment/workload
}

stage_scaling() {
  rm -rf experiments/a_matrix/results/gpu_scaling/{raw_sglang,datatrove,nemo_curator}
  [[ "$RERUN_REUSED" == 1 ]] && rm -rf experiments/a_matrix/results/gpu_scaling/mmirage
  python experiments/a_matrix/scripts/run_setup.py --setup gpu_scaling $REUSE_FLAG
}

stage_recovery() {
  rm -rf "$MMIRAGE_RECOVERY_ROOT"/runs "$MMIRAGE_RECOVERY_ROOT"/native_competitors "$MMIRAGE_RECOVERY_ROOT"/results
  python experiments/a_matrix/scripts/run_setup.py --setup recovery
  python experiments/a_matrix/scripts/run_setup.py --setup recovery --extract
}

stage_text() {
  rm -rf experiments/task_comparison/text_shortening/results/native_competitors
  [[ "$RERUN_REUSED" == 1 ]] && rm -rf experiments/task_comparison/text_shortening/results/runs
  python experiments/a_matrix/scripts/run_setup.py --setup text_shortening $REUSE_FLAG
}

stage_vlm() {
  rm -rf experiments/task_comparison/vlm_enrichment/results/native_competitors
  [[ "$RERUN_REUSED" == 1 ]] && rm -rf experiments/task_comparison/vlm_enrichment/results/runs
  python experiments/a_matrix/scripts/run_setup.py --setup vlm_enrichment $REUSE_FLAG
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
