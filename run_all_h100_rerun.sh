#!/usr/bin/env bash
# Self-contained H100 publication rerun: all corrected A-matrix experiments on
# one 4-GPU node. Recovery runs reps 1/2/3 and GPU scaling is serialized.
# Historical fast-run cells are intentionally not reused.
#
# Usage:
#   bash run_all_h100_rerun.sh
#   bash run_all_h100_rerun.sh --dry-run
#
# --dry-run is non-destructive: it performs preflight/hardware checks and prints
# the effective run_setup plans, but does not delete results, prepare workloads,
# run inference, or extract recovery results.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: bash run_all_h100_rerun.sh [--dry-run]" >&2; exit 2 ;;
esac

export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export MMIRAGE_DISTILABEL_PYTHON="${MMIRAGE_DISTILABEL_PYTHON:-$REPO_ROOT/.venv-distilabel/bin/python}"
export MMIRAGE_RAY_DATA_LLM_PYTHON="${MMIRAGE_RAY_DATA_LLM_PYTHON:-$REPO_ROOT/.venv-ray_data_llm/bin/python}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$MMIRAGE_RECOVERY_ROOT/hf"

# Resolve authentication without overwriting an already-provided token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "$HOME/keys/hf_key.txt" ]]; then
    export HF_TOKEN="$(<"$HOME/keys/hf_key.txt")"
  else
    echo "FATAL: HF_TOKEN is unset and $HOME/keys/hf_key.txt does not exist." >&2
    exit 1
  fi
fi

echo "=== Publication preflight ==="

if ! python -c 'import mmirage, sglang' >/dev/null 2>&1; then
  echo "FATAL: current Python cannot import both mmirage and sglang." >&2
  exit 1
fi

for var in \
  MMIRAGE_DATATROVE_PYTHON \
  MMIRAGE_NEMO_CURATOR_PYTHON \
  MMIRAGE_DISTILABEL_PYTHON \
  MMIRAGE_RAY_DATA_LLM_PYTHON; do
  value="${!var}"
  if [[ ! -x "$value" ]]; then
    echo "FATAL: $var=$value is not an executable interpreter." >&2
    exit 1
  fi
done

GPU_INFO="$(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1)" || {
  echo "FATAL: nvidia-smi failed." >&2
  exit 1
}
echo "$GPU_INFO"

GPU_COUNT="$(printf '%s\n' "$GPU_INFO" | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "$GPU_COUNT" -ne 4 ]]; then
  echo "FATAL: expected exactly 4 GPUs, found $GPU_COUNT." >&2
  exit 1
fi
BAD_GPUS="$(printf '%s\n' "$GPU_INFO" | grep -iv 'H100' || true)"
if [[ -n "$BAD_GPUS" ]]; then
  echo "FATAL: not all GPUs are H100:" >&2
  echo "$BAD_GPUS" >&2
  exit 1
fi
echo "Hardware check PASS: 4x H100"

RUN_SETUP="$REPO_ROOT/experiments/a_matrix/scripts/run_setup.py"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== Non-destructive publication dry-run ==="
  python "$RUN_SETUP" --setup gpu_scaling --serial --repetitions 3 --dry-run
  python "$RUN_SETUP" --setup recovery --repetitions 3 --dry-run
  echo "dry-run: recovery extraction would use repetitions 1,2,3 (not executed)."
  python "$RUN_SETUP" --setup text_shortening --repetitions 3 --dry-run
  python "$RUN_SETUP" --setup vlm_enrichment --repetitions 3 --dry-run
  echo "Dry-run complete; no results or workloads were modified."
  exit 0
fi

echo "=== Clearing old publication results ==="
OLD_SCALING="$REPO_ROOT/experiments/a_matrix/results/gpu_scaling"
OLD_TEXT="$REPO_ROOT/experiments/task_comparison/text_shortening/results"
OLD_VLM="$REPO_ROOT/experiments/task_comparison/vlm_enrichment/results"

for d in \
  "$OLD_SCALING/raw_sglang" \
  "$OLD_SCALING/datatrove" \
  "$OLD_SCALING/nemo_curator" \
  "$OLD_SCALING/mmirage"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done

for d in "$OLD_TEXT/native_competitors" "$OLD_TEXT/runs"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done

for d in "$OLD_VLM/native_competitors" "$OLD_VLM/runs"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done

for d in \
  "$MMIRAGE_RECOVERY_ROOT/runs" \
  "$MMIRAGE_RECOVERY_ROOT/native_competitors" \
  "$MMIRAGE_RECOVERY_ROOT/results"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done

rm -rf "$REPO_ROOT/experiments/a_matrix/results/recovery"

echo "=== Preparing fixed publication workloads (no calibration) ==="
python "$REPO_ROOT/experiments/a_matrix/scripts/prepare_workload.py" \
  --output-dir "$REPO_ROOT/experiments/a_matrix/workload" \
  --shared-root "$MMIRAGE_RECOVERY_ROOT"

python "$REPO_ROOT/experiments/task_comparison/text_shortening/scripts/prepare_workload.py" \
  --output-dir "$REPO_ROOT/experiments/task_comparison/text_shortening/workload"

python "$REPO_ROOT/experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py" \
  --output-dir "$REPO_ROOT/experiments/task_comparison/vlm_enrichment/workload"

python - <<'PY'
import json
from pathlib import Path
metadata = json.loads(Path("experiments/a_matrix/workload/metadata.json").read_text())
print("A-MATRIX workload_sha256:", metadata["workload_sha256"])
print("A-MATRIX dataset_revision_resolved:", metadata.get("dataset_revision_resolved"))
print("Copy this exact workload/ directory to the A100 node; do not regenerate it there.")
PY

echo "=== 1/5 GPU scaling (serialized) ==="
python "$RUN_SETUP" --setup gpu_scaling --serial --repetitions 3 --overwrite

echo "=== 2/5 Recovery (3 repetitions per condition) ==="
python "$RUN_SETUP" --setup recovery --repetitions 3 --overwrite

echo "=== 3/5 Recovery extraction (reps 1,2,3) ==="
python "$RUN_SETUP" --setup recovery --extract --repetitions 3

echo "=== 4/5 Text shortening ==="
python "$RUN_SETUP" --setup text_shortening --repetitions 3 --overwrite

echo "=== 5/5 VLM enrichment ==="
python "$RUN_SETUP" --setup vlm_enrichment --repetitions 3 --overwrite

echo "=== H100 publication suite complete ==="
