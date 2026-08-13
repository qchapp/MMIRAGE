#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${CHARTQA_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${CHARTQA_SGLANG_PORT:-30000}"
TP_SIZE="${CHARTQA_TP_SIZE:-1}"
DTYPE="${CHARTQA_DTYPE:-bfloat16}"
MAX_RUNNING_REQUESTS="${CHARTQA_MAX_RUNNING_REQUESTS:-1024}"

exec "${PYTHON:-python3}" -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  "$@"
