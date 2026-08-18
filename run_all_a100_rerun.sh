#!/usr/bin/env bash
# Corrected unattended four-GPU A100 publication transfer point.
# Copy from H100 first: experiments/a_matrix/workload/ and experiments/raw_sglang_overhead/workload/
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
DRY_RUN=0
case "${1:-}" in "") ;; --dry-run) DRY_RUN=1 ;; *) echo "usage: bash run_all_a100_rerun.sh [--dry-run]" >&2; exit 2 ;; esac
export SETUPTOOLS_USE_DISTUTILS=local TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"; NEMO_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
if [[ -z "${HF_TOKEN:-}" ]]; then if [[ -f "$HOME/keys/hf_key.txt" ]]; then export HF_TOKEN="$(<"$HOME/keys/hf_key.txt")"; else echo "FATAL: HF_TOKEN unavailable." >&2; exit 1; fi; fi
python -c 'import mmirage, sglang, transformers, huggingface_hub' >/dev/null 2>&1 || { echo "FATAL: main Python imports failed." >&2; exit 1; }
[[ -x "$DATATROVE_PYTHON" ]] && "$DATATROVE_PYTHON" -c 'import datatrove, vllm' >/dev/null 2>&1 || { echo "FATAL: DataTrove environment not ready." >&2; exit 1; }
[[ -x "$NEMO_PYTHON" ]] && "$NEMO_PYTHON" -c 'import nemo_curator, vllm' >/dev/null 2>&1 || { echo "FATAL: NeMo environment not ready." >&2; exit 1; }
GPU_INFO="$(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1)" || { echo "FATAL: nvidia-smi failed." >&2; exit 1; }; echo "$GPU_INFO"
GPU_COUNT="$(printf '%s\n' "$GPU_INFO" | sed '/^[[:space:]]*$/d' | wc -l)"; [[ "$GPU_COUNT" -eq 4 ]] || { echo "FATAL: expected exactly 4 GPUs, found $GPU_COUNT." >&2; exit 1; }
BAD_GPUS="$(printf '%s\n' "$GPU_INFO" | grep -iv 'A100' || true)"; [[ -z "$BAD_GPUS" ]] || { echo "FATAL: not all GPUs are A100:" >&2; echo "$BAD_GPUS" >&2; exit 1; }
A_WORKLOAD="$REPO_ROOT/experiments/a_matrix/workload"; OVERHEAD_WORKLOAD="$REPO_ROOT/experiments/raw_sglang_overhead/workload"
for f in "$A_WORKLOAD/workload.jsonl" "$A_WORKLOAD/metadata.json" "$A_WORKLOAD/model_revisions.json" "$A_WORKLOAD/publication_manifest.json" "$OVERHEAD_WORKLOAD/prompts.jsonl" "$OVERHEAD_WORKLOAD/warmup_prompts.jsonl" "$OVERHEAD_WORKLOAD/metadata.json"; do [[ -f "$f" ]] || { echo "FATAL: missing copied H100 artifact: $f" >&2; exit 1; }; done
python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
a=Path('experiments/a_matrix/workload'); o=Path('experiments/raw_sglang_overhead/workload'); m=json.loads((a/'publication_manifest.json').read_text()); commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if commit!=m['git_commit']: raise SystemExit(f"FATAL: code commit differs from H100: A100={commit}, H100={m['git_commit']}")
for label,actual,expected in [('A-MATRIX',sha(a/'workload.jsonl'),m['a_matrix_workload_sha256']),('overhead prompts',sha(o/'prompts.jsonl'),m['overhead_prompts_sha256']),('overhead warmup',sha(o/'warmup_prompts.jsonl'),m['overhead_warmup_sha256'])]:
 if actual!=expected: raise SystemExit(f'FATAL: {label} SHA mismatch: {actual} != {expected}')
 print(f'{label} hash PASS: {actual}')
print('git commit PASS:',commit); print('expected text model revision:',m['models']['Qwen/Qwen3-4B'])
PY
WRAPPER_DIR="$(mktemp -d)"; trap 'rm -rf "$WRAPPER_DIR"' EXIT
make_python_wrapper() { local actual="$1" dest="$2"; cat >"$dest" <<EOF
#!/usr/bin/env bash
export PATH="$(dirname "$actual"):\${PATH:-}"
exec "$actual" "\$@"
EOF
chmod +x "$dest"; }
make_python_wrapper "$DATATROVE_PYTHON" "$WRAPPER_DIR/datatrove-python"; make_python_wrapper "$NEMO_PYTHON" "$WRAPPER_DIR/nemo-python"
export MMIRAGE_DATATROVE_PYTHON="$WRAPPER_DIR/datatrove-python" MMIRAGE_NEMO_CURATOR_PYTHON="$WRAPPER_DIR/nemo-python"
PUB_SCALING="$REPO_ROOT/experiments/a_matrix/scripts/run_publication_scaling.py"; PREFETCH="$REPO_ROOT/experiments/a_matrix/scripts/prefetch_publication_models.py"; OVERHEAD_RUN="$REPO_ROOT/experiments/raw_sglang_overhead/scripts/run.py"
if [[ "$DRY_RUN" -eq 1 ]]; then python "$PUB_SCALING" --setup a100_4gpu --repetitions 3 --overwrite --dry-run; echo "overhead-plan: raw_sglang + mmirage_sglang, 3 reps, GPU0, concurrency64, max_tokens1024"; exit 0; fi
python "$PREFETCH" --models Qwen/Qwen3-4B --expected-json "$A_WORKLOAD/model_revisions.json" --output-json "$A_WORKLOAD/a100_model_prefetch.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$REPO_ROOT/experiments/a_matrix/results/a100_4gpu" "$REPO_ROOT/experiments/raw_sglang_overhead/results/a100_publication"
python "$PUB_SCALING" --setup a100_4gpu --repetitions 3 --overwrite
TEXT_MODEL_REV="$(python -c 'import json; print(json.load(open("experiments/a_matrix/workload/model_revisions.json"))["Qwen/Qwen3-4B"])')"
python "$OVERHEAD_RUN" --workload-dir "$OVERHEAD_WORKLOAD" --output-dir "$REPO_ROOT/experiments/raw_sglang_overhead/results/a100_publication" --frameworks raw_sglang,mmirage_sglang --repetitions 3 --gpu-index 0 --concurrency 64 --max-tokens 1024 --temperature 0.0 --model-revision "$TEXT_MODEL_REV"
echo "=== A100 publication suite complete ==="
