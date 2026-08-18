#!/usr/bin/env bash
# Canonical unattended H100 publication suite.
# Usage: bash experiments/publication/run_h100.sh [--dry-run]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
DRY_RUN=0
case "${1:-}" in "") ;; --dry-run) DRY_RUN=1 ;; *) echo "usage: bash experiments/publication/run_h100.sh [--dry-run]" >&2; exit 2 ;; esac
TRACKED_DIRTY="$(git status --porcelain --untracked-files=no)"
[[ -z "$TRACKED_DIRTY" ]] || { echo "FATAL: tracked working tree has uncommitted changes; publication provenance would be ambiguous." >&2; printf '%s\n' "$TRACKED_DIRTY" >&2; exit 1; }

export MMIRAGE_RECOVERY_ROOT="${MMIRAGE_RECOVERY_ROOT:-/workspace/mmirage-recovery}"
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$MMIRAGE_RECOVERY_ROOT/hf}"
DATATROVE_PYTHON="${MMIRAGE_DATATROVE_PYTHON:-$REPO_ROOT/.venv-datatrove/bin/python}"
NEMO_PYTHON="${MMIRAGE_NEMO_CURATOR_PYTHON:-$REPO_ROOT/.venv-nemo_curator/bin/python}"
DISTILABEL_PYTHON="${MMIRAGE_DISTILABEL_PYTHON:-$REPO_ROOT/.venv-distilabel/bin/python}"
RAY_PYTHON="${MMIRAGE_RAY_DATA_LLM_PYTHON:-$REPO_ROOT/.venv-ray_data_llm/bin/python}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "$HOME/keys/hf_key.txt" ]]; then export HF_TOKEN="$(<"$HOME/keys/hf_key.txt")"; else echo "FATAL: HF_TOKEN is unset and $HOME/keys/hf_key.txt does not exist." >&2; exit 1; fi
fi

echo "=== H100 publication preflight ==="
python -c 'import mmirage, sglang, datasets, transformers, huggingface_hub, yaml' >/dev/null 2>&1 || { echo "FATAL: main Python imports failed." >&2; exit 1; }
command -v mmirage >/dev/null 2>&1 || { echo "FATAL: mmirage CLI is not on PATH." >&2; exit 1; }
check_env() { local label="$1" python_bin="$2" import_code="$3"; [[ -x "$python_bin" ]] || { echo "FATAL: $label interpreter not executable: $python_bin" >&2; exit 1; }; "$python_bin" -c "$import_code" >/dev/null 2>&1 || { echo "FATAL: $label imports failed: $import_code" >&2; exit 1; }; }
check_env "DataTrove" "$DATATROVE_PYTHON" 'import datatrove, vllm'
check_env "NeMo Curator" "$NEMO_PYTHON" 'import nemo_curator, vllm'
check_env "Distilabel" "$DISTILABEL_PYTHON" 'import distilabel, vllm'
check_env "Ray Data LLM" "$RAY_PYTHON" 'import ray, vllm'
GPU_INFO="$(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>&1)" || { echo "FATAL: nvidia-smi failed." >&2; exit 1; }
echo "$GPU_INFO"
GPU_COUNT="$(printf '%s\n' "$GPU_INFO" | sed '/^[[:space:]]*$/d' | wc -l)"
[[ "$GPU_COUNT" -eq 4 ]] || { echo "FATAL: expected exactly 4 GPUs, found $GPU_COUNT." >&2; exit 1; }
BAD_GPUS="$(printf '%s\n' "$GPU_INFO" | grep -iv 'H100' || true)"
[[ -z "$BAD_GPUS" ]] || { echo "FATAL: not all GPUs are H100:" >&2; echo "$BAD_GPUS" >&2; exit 1; }
echo "Hardware check PASS: 4x H100"

WRAPPER_DIR="$(mktemp -d)"; trap 'rm -rf "$WRAPPER_DIR"' EXIT
make_python_wrapper() { local actual="$1" dest="$2"; cat >"$dest" <<EOF
#!/usr/bin/env bash
export PATH="$(dirname "$actual"):\${PATH:-}"
exec "$actual" "\$@"
EOF
chmod +x "$dest"; }
make_python_wrapper "$DATATROVE_PYTHON" "$WRAPPER_DIR/datatrove-python"
make_python_wrapper "$NEMO_PYTHON" "$WRAPPER_DIR/nemo-python"
make_python_wrapper "$DISTILABEL_PYTHON" "$WRAPPER_DIR/distilabel-python"
make_python_wrapper "$RAY_PYTHON" "$WRAPPER_DIR/ray-python"
export MMIRAGE_DATATROVE_PYTHON="$WRAPPER_DIR/datatrove-python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$WRAPPER_DIR/nemo-python"
export MMIRAGE_DISTILABEL_PYTHON="$WRAPPER_DIR/distilabel-python"
export MMIRAGE_RAY_DATA_LLM_PYTHON="$WRAPPER_DIR/ray-python"

ORCH="$REPO_ROOT/experiments/publication/orchestrate.py"
OVERHEAD_RUN="$REPO_ROOT/experiments/sglang_overhead/scripts/run.py"
if [[ "$DRY_RUN" -eq 1 ]]; then
  python "$ORCH" --stage scaling --hardware h100 --repetitions 3 --overwrite --dry-run
  python "$ORCH" --stage recovery --repetitions 3 --overwrite --dry-run
  python "$ORCH" --stage text --repetitions 3 --overwrite --dry-run
  python "$ORCH" --stage vlm --repetitions 3 --overwrite --dry-run
  echo "overhead-plan: raw_sglang + mmirage_sglang, 3 reps, GPU0, concurrency64, max_tokens1024"
  echo "Dry-run complete; no workloads/results/models were modified."
  exit 0
fi

echo "=== Preparing deterministic publication workloads ==="
python "$REPO_ROOT/experiments/scaling/scripts/prepare_workload.py" --output-dir "$REPO_ROOT/experiments/scaling/workload" --shared-root "$MMIRAGE_RECOVERY_ROOT"
python "$REPO_ROOT/experiments/text_shortening/scripts/prepare_workload.py" --output-dir "$REPO_ROOT/experiments/text_shortening/workload"
python "$REPO_ROOT/experiments/vlm_enrichment/scripts/prepare_workload.py" --output-dir "$REPO_ROOT/experiments/vlm_enrichment/workload"
python "$REPO_ROOT/experiments/sglang_overhead/scripts/prepare_workload.py" --output-dir "$REPO_ROOT/experiments/sglang_overhead/workload" --num-rows 1000

echo "=== Prefetching exact model snapshots outside timed regions ==="
MODEL_REVISIONS="$REPO_ROOT/experiments/scaling/workload/model_revisions.json"
python "$REPO_ROOT/experiments/publication/prefetch_models.py" --models Qwen/Qwen3-4B Qwen/Qwen3-VL-4B-Instruct --output-json "$MODEL_REVISIONS"

python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
repo=Path('.').resolve(); s=repo/'experiments/scaling/workload'; t=repo/'experiments/text_shortening/workload'; v=repo/'experiments/vlm_enrichment/workload'; o=repo/'experiments/sglang_overhead/workload'
s_meta=json.loads((s/'metadata.json').read_text()); t_meta=json.loads((t/'metadata.json').read_text()); v_meta=json.loads((v/'metadata.json').read_text()); o_meta=json.loads((o/'metadata.json').read_text()); models=json.loads((s/'model_revisions.json').read_text()); text_model=models['Qwen/Qwen3-4B']
for label,meta in [('scaling',s_meta),('text_shortening',t_meta),('sglang_overhead',o_meta)]:
    recorded=meta.get('model_revision_resolved')
    if recorded and recorded!=text_model: raise SystemExit(f'FATAL: {label} model revision {recorded} != prefetched {text_model}')
manifest={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'models':models,'scaling_workload_sha256':sha(s/'workload.jsonl'),'text_workload_sha256':sha(t/'workload.jsonl'),'vlm_rows_sha256':sha(v/'rows.jsonl'),'overhead_prompts_sha256':sha(o/'prompts.jsonl'),'overhead_warmup_sha256':sha(o/'warmup_prompts.jsonl'),'scaling_dataset_revision':s_meta.get('dataset_revision_resolved'),'text_dataset_revision':t_meta.get('dataset_revision_resolved'),'vlm_dataset_revision':v_meta.get('dataset_revision_resolved'),'overhead_dataset_revision':o_meta.get('dataset_revision_resolved'),'h100_hardware':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True).strip().splitlines()}
(s/'publication_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); print(json.dumps(manifest,indent=2,sort_keys=True))
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Clearing prior publication outputs only after preparation passed ==="
rm -rf "$REPO_ROOT/experiments/scaling/results/h100" \
       "$REPO_ROOT/experiments/text_shortening/results" \
       "$REPO_ROOT/experiments/vlm_enrichment/results" \
       "$REPO_ROOT/experiments/recovery/results" \
       "$REPO_ROOT/experiments/sglang_overhead/results/h100"
for d in "$MMIRAGE_RECOVERY_ROOT/runs" "$MMIRAGE_RECOVERY_ROOT/native_competitors" "$MMIRAGE_RECOVERY_ROOT/results"; do [[ -d "$d" ]] && rm -rf "$d"; done

python "$ORCH" --stage scaling --hardware h100 --repetitions 3 --overwrite
python "$ORCH" --stage recovery --repetitions 3 --overwrite
python "$ORCH" --stage recovery_extract --repetitions 3
python - <<'PY'
import os, shutil
from pathlib import Path
repo=Path('.').resolve(); shared=Path(os.environ['MMIRAGE_RECOVERY_ROOT']); dest=repo/'experiments/recovery/results'; dest.mkdir(parents=True,exist_ok=True)
if (shared/'results').exists():
    for src in (shared/'results').iterdir():
        if src.is_file(): shutil.copy2(src,dest/src.name)
for source_name in ('runs','native_competitors'):
    source=shared/source_name
    if source.exists():
        for src in source.rglob('*.json'):
            target=dest/'evidence'/src.relative_to(shared); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,target)
print(f'Persisted recovery JSON evidence under {dest}')
PY
python "$ORCH" --stage text --repetitions 3 --overwrite
python "$ORCH" --stage vlm --repetitions 3 --overwrite
TEXT_MODEL_REV="$(python -c 'import json; print(json.load(open("experiments/scaling/workload/model_revisions.json"))["Qwen/Qwen3-4B"])')"
python "$OVERHEAD_RUN" --workload-dir "$REPO_ROOT/experiments/sglang_overhead/workload" --output-dir "$REPO_ROOT/experiments/sglang_overhead/results/h100" --frameworks raw_sglang,mmirage_sglang --repetitions 3 --gpu-index 0 --concurrency 64 --max-tokens 1024 --temperature 0.0 --model-revision "$TEXT_MODEL_REV"
echo "=== H100 publication suite complete ==="
echo "For a separate A100 node, copy/share experiments/scaling/workload/ and experiments/sglang_overhead/workload/."
