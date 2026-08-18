#!/usr/bin/env bash
# Self-contained H100 rerun: all experiments on a 4-GPU pod with results
# written to experiments/a_matrix/results_h100_rerun/ (separate from original).
# Recovery runs 3 reps per condition. GPU scaling runs serialized.
#
# Requirements: 4x H100 GPUs, venvs at .venv-datatrove, .venv-nemo_curator,
# .venv-distilabel, .venv-ray_data_llm. HF token via ~/keys/hf_key.txt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Environment
export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export MMIRAGE_DISTILABEL_PYTHON="${MMIRAGE_DISTILABEL_PYTHON:-$REPO_ROOT/.venv-distilabel/bin/python}"
export MMIRAGE_RAY_DATA_LLM_PYTHON="${MMIRAGE_RAY_DATA_LLM_PYTHON:-$REPO_ROOT/.venv-ray_data_llm/bin/python}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$MMIRAGE_RECOVERY_ROOT/hf"

# HF token by path - never read/displayed
export HF_TOKEN="$(cat ~/keys/hf_key.txt)"

# Results directory (separate from original a100_4gpu results)
RESULTS_ROOT="$REPO_ROOT/experiments/a_matrix/results_h100_rerun"
LOG_DIR="$REPO_ROOT/experiments/run_all_logs/a_matrix/rerun_h100"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

# Old publication results to clear (as specified in constraint F)
OLD_SCALING="$REPO_ROOT/experiments/a_matrix/results/gpu_scaling"
OLD_TEXT="$REPO_ROOT/experiments/task_comparison/text_shortening/results"
OLD_VLM="$REPO_ROOT/experiments/task_comparison/vlm_enrichment/results"

# Old recovery results to clear
OLD_RECOVERY="$REPO_ROOT/experiments/a_matrix/results/recovery"

# Clear old results (as required: fresh results only)
if [[ -d "$OLD_SCALING" ]]; then
  rm -rf "$OLD_SCALING"/{raw_sglang,datatrove,nemo_curator,mmirage}
fi
if [[ -d "$OLD_TEXT" ]]; then
  rm -rf "$OLD_TEXT"/{native_competitors,runs}
fi
if [[ -d "$OLD_VLM" ]]; then
  rm -rf "$OLD_VLM"/{native_competitors,runs}
fi
if [[ -d "$OLD_RECOVERY" ]]; then
  rm -rf "$OLD_RECOVERY"
fi

# Stage functions...
ALL_STAGES=(gpu_scaling recovery text_shortening vlm_enrichment)

# ... rest of script
