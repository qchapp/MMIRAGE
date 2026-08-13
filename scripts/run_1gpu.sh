#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
${PYTHON:-python3} "$SCRIPT_DIR/run_scaling.py" \
  --execution-config "$PROJECT_ROOT/experiments/single_node_h100_scaling/execution_1gpu.yaml" \
  "$@"
