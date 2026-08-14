#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
${PYTHON:-python3} "$SCRIPT_DIR/run.py" \
  --execution-config "$EXPERIMENT_DIR/configs/execution_4gpu.yaml" \
  "$@"
