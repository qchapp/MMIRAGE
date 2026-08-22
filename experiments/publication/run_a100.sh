#!/usr/bin/env bash
# Four-GPU A100 publication transfer evaluation.
# Reuses the exact scaling and SGLang-overhead workloads produced on H100.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"
DRY_RUN=0
case "${1:-}" in "") ;; --dry-run) DRY_RUN=1 ;; *) echo "usage: bash experiments/publication/run_a100.sh [--dry-run]" >&2; exit 2 ;; esac
TRACKED_DIRTY="$(git status --porcelain --untracked-files=no)"
[[ -z "$TRACKED_DIRTY" ]] || { echo "FATAL: tracked working tree has uncommitted changes; publication provenance would be ambiguous." >&2; printf '%s\n' "$TRACKED_DIRTY" >&2; exit 1; }
export SETUPTOOLS_USE_DISTUTILS=local TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"; NEMO_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face access token before running this evaluation.}"
python -c 'import mmirage, sglang, transformers, huggingface_hub, yaml' >/dev/null 2>&1 || { echo "FATAL: main Python imports failed." >&2; exit 1; }
[[ -x "$DATATROVE_PYTHON" ]] && "$DATATROVE_PYTHON" -c 'import datatrove, vllm' >/dev/null 2>&1 || { echo "FATAL: DataTrove environment not ready." >&2; exit 1; }
[[ -x "$NEMO_PYTHON" ]] && "$NEMO_PYTHON" -c 'import nemo_curator, vllm' >/dev/null 2>&1 || { echo "FATAL: NeMo environment not ready." >&2; exit 1; }
GPU_INFO="$(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1)" || { echo "FATAL: nvidia-smi failed." >&2; exit 1; }; echo "$GPU_INFO"
GPU_COUNT="$(printf '%s\n' "$GPU_INFO" | sed '/^[[:space:]]*$/d' | wc -l)"; [[ "$GPU_COUNT" -eq 4 ]] || { echo "FATAL: expected exactly 4 GPUs, found $GPU_COUNT." >&2; exit 1; }
BAD_GPUS="$(printf '%s\n' "$GPU_INFO" | grep -iv 'A100' || true)"; [[ -z "$BAD_GPUS" ]] || { echo "FATAL: not all GPUs are A100:" >&2; echo "$BAD_GPUS" >&2; exit 1; }

SCALING_WORKLOAD="$REPO_ROOT/experiments/scaling/workload"; OVERHEAD_WORKLOAD="$REPO_ROOT/experiments/sglang_overhead/workload"
for f in "$SCALING_WORKLOAD/workload.jsonl" "$SCALING_WORKLOAD/metadata.json" "$SCALING_WORKLOAD/model_revisions.json" "$SCALING_WORKLOAD/publication_manifest.json" "$OVERHEAD_WORKLOAD/prompts.jsonl" "$OVERHEAD_WORKLOAD/warmup_prompts.jsonl" "$OVERHEAD_WORKLOAD/metadata.json"; do [[ -f "$f" ]] || { echo "FATAL: missing H100-produced artifact: $f" >&2; exit 1; }; done
python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
s=Path('experiments/scaling/workload'); o=Path('experiments/sglang_overhead/workload'); m=json.loads((s/'publication_manifest.json').read_text()); commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if commit!=m['git_commit']: raise SystemExit(f"FATAL: code commit differs from H100: A100={commit}, H100={m['git_commit']}")
for label,actual,expected in [('scaling',sha(s/'workload.jsonl'),m['scaling_workload_sha256']),('overhead prompts',sha(o/'prompts.jsonl'),m['overhead_prompts_sha256']),('overhead warmup',sha(o/'warmup_prompts.jsonl'),m['overhead_warmup_sha256'])]:
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
ORCH="$REPO_ROOT/experiments/publication/orchestrate.py"; OVERHEAD_RUN="$REPO_ROOT/experiments/sglang_overhead/scripts/run.py"
if [[ "$DRY_RUN" -eq 1 ]]; then python "$ORCH" --stage scaling --hardware a100 --repetitions 3 --overwrite --dry-run; echo "overhead-plan: raw_sglang + mmirage_sglang, 3 reps, GPU0, concurrency64, max_tokens1024"; exit 0; fi
python "$REPO_ROOT/experiments/publication/prefetch_models.py" --models Qwen/Qwen3-4B --expected-json "$SCALING_WORKLOAD/model_revisions.json" --output-json "$SCALING_WORKLOAD/a100_model_prefetch.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf "$REPO_ROOT/experiments/scaling/results/a100" "$REPO_ROOT/experiments/sglang_overhead/results/a100"
python "$ORCH" --stage scaling --hardware a100 --repetitions 3 --overwrite
TEXT_MODEL_REV="$(python -c 'import json; print(json.load(open("experiments/scaling/workload/model_revisions.json"))["Qwen/Qwen3-4B"])')"
python "$OVERHEAD_RUN" --workload-dir "$OVERHEAD_WORKLOAD" --output-dir "$REPO_ROOT/experiments/sglang_overhead/results/a100" --frameworks raw_sglang,mmirage_sglang --repetitions 3 --gpu-index 0 --concurrency 64 --max-tokens 1024 --temperature 0.0 --model-revision "$TEXT_MODEL_REV"
echo "=== A100 publication suite complete ==="
