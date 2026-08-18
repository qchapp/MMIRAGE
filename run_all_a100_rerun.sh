#!/usr/bin/env bash
# Four-GPU A100 publication transfer point.
#
# This script MUST consume the exact A-MATRIX workload prepared for the H100
# publication run. Copy experiments/a_matrix/workload/ from the H100 node to
# this node before running; do not regenerate it here.
#
# Usage:
#   bash run_all_a100_rerun.sh
#   bash run_all_a100_rerun.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: bash run_all_a100_rerun.sh [--dry-run]" >&2; exit 2 ;;
esac

export MMIRAGE_DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
export MMIRAGE_NEMO_CURATOR_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false

echo "=== A100 publication preflight ==="

if ! python -c 'import mmirage, sglang' >/dev/null 2>&1; then
  echo "FATAL: current Python cannot import both mmirage and sglang." >&2
  exit 1
fi

for var in MMIRAGE_DATATROVE_PYTHON MMIRAGE_NEMO_CURATOR_PYTHON; do
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
BAD_GPUS="$(printf '%s\n' "$GPU_INFO" | grep -iv 'A100' || true)"
if [[ -n "$BAD_GPUS" ]]; then
  echo "FATAL: not all GPUs are A100:" >&2
  echo "$BAD_GPUS" >&2
  exit 1
fi
echo "Hardware check PASS: 4x A100"

WORKLOAD_DIR="$REPO_ROOT/experiments/a_matrix/workload"
WORKLOAD="$WORKLOAD_DIR/workload.jsonl"
METADATA="$WORKLOAD_DIR/metadata.json"

if [[ ! -f "$WORKLOAD" || ! -f "$METADATA" ]]; then
  echo "FATAL: missing A-MATRIX workload.jsonl or metadata.json." >&2
  echo "Copy the exact experiments/a_matrix/workload/ directory from the H100 node." >&2
  exit 1
fi

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("experiments/a_matrix/workload")
workload = root / "workload.jsonl"
metadata = json.loads((root / "metadata.json").read_text())
expected = metadata.get("workload_sha256")
if not expected:
    raise SystemExit("FATAL: metadata.json has no workload_sha256")
digest = hashlib.sha256(workload.read_bytes()).hexdigest()
if digest != expected:
    raise SystemExit(f"FATAL: workload SHA-256 mismatch: file={digest}, metadata={expected}")
print("Workload hash PASS:", digest)
print("dataset_revision_resolved:", metadata.get("dataset_revision_resolved"))
PY

RUN_SETUP="$REPO_ROOT/experiments/a_matrix/scripts/run_setup.py"

if [[ "$DRY_RUN" -eq 1 ]]; then
  python "$RUN_SETUP" --setup a100_4gpu --repetitions 3 --dry-run
  echo "Dry-run complete; no A100 results were modified."
  exit 0
fi

A100_RESULTS="$REPO_ROOT/experiments/a_matrix/results/a100_4gpu"
for d in \
  "$A100_RESULTS/raw_sglang" \
  "$A100_RESULTS/datatrove" \
  "$A100_RESULTS/nemo_curator" \
  "$A100_RESULTS/mmirage"; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "cleared $d"
done

python "$RUN_SETUP" --setup a100_4gpu --repetitions 3 --overwrite

echo "=== A100 publication transfer point complete ==="
