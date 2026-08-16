#!/usr/bin/env bash
# Run the 4x A100 GPU-scaling point on the A100 pod (Comparison A, S2).
#
# The A100 pod only runs the 4-GPU scaling point: the workload and sizes are
# produced on the H100 pod (experiments/a_matrix/workload + calibration) and
# are expected to be present here (same checkout, or the workload copied over).
# Prepares nothing; it reuses the committed workload_size.yaml / recovery size
# files. See experiments/a_matrix/README.md.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM=false

fail=0
if ! python -c "import sglang" 2>/dev/null; then
  echo "preflight: 'python' cannot import sglang - activate the MMIRAGE venv." >&2
  fail=1
fi
gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "$gpus" -lt 4 ]]; then
  echo "preflight: need 4 visible GPUs, found ${gpus:-0}." >&2
  fail=1
fi
for var in MMIRAGE_DATATROVE_PYTHON MMIRAGE_NEMO_CURATOR_PYTHON; do
  if [[ ! -x "${!var}" ]]; then
    echo "preflight: $var=${!var} is not executable - build the competitor venvs." >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

exec python experiments/a_matrix/scripts/run_setup.py --setup a100_4gpu
