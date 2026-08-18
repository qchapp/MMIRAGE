#!/usr/bin/env bash
# Self-contained H100 publication rerun: all A-matrix experiments on one 4-GPU
# node with corrected batch=64 configuration. Recovery runs reps 1/2/3.
# GPU scaling runs serialized. Do NOT use --reuse-fastruns (old fast-run cells
# are incompatible with the corrected publication batch config).
#
# Requirements: 4x H100 GPUs, venvs at .venv-datatrove, .venv-nemo_curator,
# .venv-distilabel, .venv-ray_data_llm. HF token via ~/keys/hf_key.txt.
#
# Usage:
#   bash run_all_h100_rerun.sh              # run everything
#   bash run_all_h100_rerun.sh --dry-run    # print plan, verify hardware, exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export MMIRAGE_DISTILABEL_PYTHON="${MMIRAGE_DISTILABEL_PYTHON:-$REPO_ROOT/.venv-distilabel/bin/python}"
export MMIRAGE_RAY_DATA_LLM_PYTHON="${MMIRAGE_RAY_DATA_LLM_PYTHON:-$REPO_ROOT/.venv-ray_data_llm/bin/python}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$MMIRAGE_RECOVERY_ROOT/hf"

# HF token by path — never read/displayed
export HF_TOKEN="$(cat ~/keys/hf_key.txt)"

# ---------------------------------------------------------------------------
# Verify exactly 4 H100 GPUs
# ---------------------------------------------------------------------------
echo "=== Hardware verification ==="
GPU_INFO="$(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1)" || {
    echo "FATAL: nvidia-smi failed" >&2
    exit 1
}
echo "$GPU_INFO"
GPU_COUNT="$(echo "$GPU_INFO" | wc -l)"
if [[ "$GPU_COUNT" -ne 4 ]]; then
    echo "FATAL: expected exactly 4 GPUs, found $GPU_COUNT" >&2
    exit 1
fi
BAD_GPUS="$(echo "$GPU_INFO" | grep -iv 'H100' || true)"
if [[ -n "$BAD_GPUS" ]]; then
    echo "FATAL: not all GPUs are H100:" >&2
    echo "$BAD_GPUS" >&2
    exit 1
fi
echo "Hardware check PASS: 4x H100"

# ---------------------------------------------------------------------------
# Clear old publication result directories (fresh measurements only)
# ---------------------------------------------------------------------------
echo "=== Clearing old results ==="
OLD_SCALING="$REPO_ROOT/experiments/a_matrix/results/gpu_scaling"
OLD_RECOVERY="$REPO_ROOT/experiments/a_matrix/results/recovery"
OLD_TEXT="$REPO_ROOT/experiments/task_comparison/text_shortening/results"
OLD_VLM="$REPO_ROOT/experiments/task_comparison/vlm_enrichment/results"

for d in "$OLD_SCALING/raw_sglang" "$OLD_SCALING/datatrove" "$OLD_SCALING/nemo_curator" "$OLD_SCALING/mmirage"; do
    [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done
[[ -d "$OLD_RECOVERY" ]] && rm -rf "$OLD_RECOVERY" && echo "  cleared $OLD_RECOVERY"
for d in "$OLD_TEXT/native_competitors" "$OLD_TEXT/runs"; do
    [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done
for d in "$OLD_VLM/native_competitors" "$OLD_VLM/runs"; do
    [[ -d "$d" ]] && rm -rf "$d" && echo "  cleared $d"
done

# ---------------------------------------------------------------------------
# Prepare workloads (no recalibration — use committed sizes)
# ---------------------------------------------------------------------------
echo "=== Preparing workloads ==="
python "$REPO_ROOT/experiments/a_matrix/scripts/prepare_workload.py" \
    --output-dir "$REPO_ROOT/experiments/a_matrix/workload" \
    --shared-root "$MMIRAGE_RECOVERY_ROOT"

python "$REPO_ROOT/experiments/task_comparison/text_shortening/scripts/prepare_workload.py" \
    --output-dir "$REPO_ROOT/experiments/task_comparison/text_shortening/workload"

python "$REPO_ROOT/experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py" \
    --output-dir "$REPO_ROOT/experiments/task_comparison/vlm_enrichment/workload"

# ---------------------------------------------------------------------------
# Run A-matrix experiments
# ---------------------------------------------------------------------------
RUN_SETUP="$REPO_ROOT/experiments/a_matrix/scripts/run_setup.py"

echo ""
echo "=== 1/5 GPU scaling (serialized) ==="
python "$RUN_SETUP" --setup gpu_scaling --serial --repetitions 3 --overwrite $DRY_RUN

echo ""
echo "=== 2/5 Recovery ==="
python "$RUN_SETUP" --setup recovery --repetitions 3 --overwrite $DRY_RUN

echo ""
echo "=== 3/5 Recovery extraction ==="
python "$RUN_SETUP" --setup recovery --extract --repetitions 3 $DRY_RUN

echo ""
echo "=== 4/5 Text shortening ==="
python "$RUN_SETUP" --setup text_shortening --repetitions 3 --overwrite $DRY_RUN

echo ""
echo "=== 5/5 VLM enrichment ==="
python "$RUN_SETUP" --setup vlm_enrichment --repetitions 3 --overwrite $DRY_RUN

echo ""
echo "=== All stages complete ==="
