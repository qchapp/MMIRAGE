# Pod agent runbook for MMIRAGE fast runs

You are an agent running inside an EPFL Run:ai GPU pod for the
`fabnemEPFL/MMIRAGE` repository, branch `experiment/fast-runs`. Your job is to
execute the MMIRAGE fast-run experiments described below, verify them, and save the
the results as a single ZIP file alongside a prompt for an agent to read, for analyzing
the results and integrating them into a publication showcasing MMIRAGE.
Everything you generate under `experiments/*/workload/`
and `experiments/*/results/` is git-ignored; never commit those.

## Hard rules

1. Do NOT touch, delete, inspect, or interfere with any Kubernetes jobs or pods
   other than your own container. No `kubectl`, no `runai`, no `runctl` work on
   other objects. You only run processes in this pod.
2. Use 1 GPU unless an experiment explicitly says otherwise. Verify with
   `nvidia-smi` that the GPU(s) you will use are free before starting a run.
3. A human is watching the terminal. Stop and ask before doing anything
   destructive (deleting directories outside the experiment dirs, changing git
   history, force-pushing).
4. Never run `pip install` into the system python. All dependencies go in uv
   virtual environments. The pod's base `python3` may be broken (torch/aten
   errors); always use a uv venv python.
5. If a step fails, diagnose from the logs and retry once. If it fails again,
   record the failure verbatim in `RUNLOG.md` and move on; do not loop.
6. Never change the code in the `src/` folder.

## Environment

- The active job (`meditron-fab-h100`, 4x H100) is yours to use; do not create
  jobs.
- Set `export HF_TOKEN=<your token>` so MedTrinity and gated datasets download.
- `Qwen/Qwen3-4B`, `Qwen/Qwen3-VL-4B-Instruct`, `HuggingFaceH4/ultrachat_200k`,
  `cnn_dailymail`, and `UCSC-VLAA/MedTrinity-25M` may need pre-downloading into
  the shared `HF_HOME`; the downloads are cached on the pod after the first run.

Create one venv with the MMIRAGE runtime:

```
export HF_HOME=/workspace/mmirage-hf
export MMIRAGE_RECOVERY_ROOT=/workspace/mmirage-recovery
source /lightscratch/users/nemo/.local/bin/uv 2>/dev/null || export PATH=/lightscratch/users/nemo/.local/bin:$PATH #may change depending on the actual environment, adapt accordingly
cd /workspace/MMIRAGE
uv venv .venv
source .venv/bin/activate
# MMIRAGE dependencies (see pyproject/setup); pins sglang + vLLM + mmirage itself
uv pip install -e . 
```

Pinned sglang versions need `setuptools<76` and `SETUPTOOLS_USE_DISTUTILS=local`:

```
uv pip install "setuptools<76"
export SETUPTOOLS_USE_DISTUTILS=local
```

`sglang` does not bound `flash-attn-4` or `nvidia-cutlass-dsl`, so a fresh resolve
can build a combination that fails at H100 attention-backend init with
`AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute
'RoundingModeKind'`. If the MMIRAGE install or a server fails that way, reinstall
with prereleases allowed and pin the recorded versions:

```
uv pip install --prerelease=allow -e .
uv pip install "flash-attn-4==4.0.0b15" "nvidia-cutlass-dsl==4.5.2" "apache-tvm-ffi==0.1.11"
```

Create the competitor venvs exactly as the per-experiment
`environment/*_uv_requirements.txt` files describe (datatrove, nemo_curator,
distilabel, ray_data_llm, raw_sglang). Each runner's README names the venv it
expects.

## Verify the repo

```
git status --short          # expect only expected modifications on this branch
git log --oneline -5
python -m py_compile experiments/_shared/*.py \
  experiments/smoke/*.py \
  experiments/raw_sglang_overhead/scripts/*.py \
  experiments/single_node_h100_scaling/scripts/*.py \
  experiments/shard_recovery/scripts/*.py \
  experiments/task_comparison/*/scripts/*.py
# YAML sanity (run in each experiment configs dir):
python -c "import yaml,glob;[yaml.safe_load(open(f)) for f in glob.glob('experiments/**/configs/*.yaml', recursive=True)]"
```

## Calibration workflow (the whole point of `experiments/smoke`)

Workload sizes are committed in each experiment's `configs/workload_size.yaml`.
They are good fallbacks but are NOT calibrated for this pod. Calibrate first:

```
# 1. Prepare + time one tiny cell per experiment (all fit on 1 GPU except the
#    4-GPU cells, which run when free).
python experiments/smoke/run_smoke.py            # writes experiments/smoke/timing.json
# 2. Size every workload to fit its budget (writes configs/workload_size.yaml).
python experiments/smoke/calibrate.py --apply    # writes experiments/smoke/calibration.json
# 3. Re-prepare each workload at the calibrated size, then run the experiments.
```

`calibrate.py --apply` prints `expected_wall_seconds` per experiment. If it
reports `over_budget: true`, use the printed `recommended` size by editing the
experiment's `configs/workload_size.yaml` down to `recommended`, or note the
deviation in `RUNLOG.md` and accept the longer wall.

After calibration, re-run each experiment's `prepare_workload.py` (no flags; it
reads the calibrated size from its config) so the workload matches the new size.

## Experiments (run in this order)

For every experiment below, record in `RUNLOG.md`: date, pod, GPU ids, calibrated
size, per-command wall times, and any failures. Each experiment's README is a
literal runbook; the commands below are the summary.

### 1. raw_sglang_overhead (1 GPU, 4 paths, 1 rep each)

```
python experiments/raw_sglang_overhead/scripts/prepare_workload.py \
  --output-dir experiments/raw_sglang_overhead/workload
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir experiments/raw_sglang_overhead/results
```

Outputs `results/summary.json` (throughput_retention, relative_orchestration_overhead),
`results/raw_results.csv`, `results/table.tex`.

### 2. single_node_h100_scaling (1/2/4 GPUs)

```
python experiments/single_node_h100_scaling/scripts/prepare_workload.py \
  --output-dir experiments/single_node_h100_scaling/workload
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/single_node_h100_scaling/configs/execution_1gpu.yaml
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/single_node_h100_scaling/configs/execution_2gpu.yaml
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/single_node_h100_scaling/configs/execution_4gpu.yaml
```

Native competitors (DataTrove, NeMo Curator, Distilabel, Ray Data LLM, raw
SGLang) each have `scripts/run_<framework>_scaling.py` wrappers; the
`configs/native_competitors.yaml` `command_template` shows exact invocation.

### 3. shard_recovery (4 GPUs, MMIRAGE only)

```
python experiments/shard_recovery/scripts/prepare_workload.py \
  --output-root /workspace/mmirage-recovery
python experiments/shard_recovery/scripts/run_local.py run-condition --condition baseline --rep 1 --max-active-shards 4 --gpu-ids 0,1,2,3
# then for each of fail_1, fail_4, fail_8 and each rep:
python experiments/shard_recovery/scripts/run_local.py run-condition --condition <cond> --rep <r> --max-active-shards 4 --gpu-ids 0,1,2,3
python experiments/shard_recovery/scripts/run_local.py retry --condition <cond> --rep <r> --max-active-shards 4 --gpu-ids 0,1,2,3
```

`run_k8s.py` exists for the Kubernetes path; `run_local.py` is the in-pod path.

### 4. task_comparison (4 GPUs)

```
python experiments/task_comparison/text_shortening/scripts/prepare_workload.py \
  --output-dir experiments/task_comparison/text_shortening/workload
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/task_comparison/text_shortening/configs/execution_4gpu.yaml
# native text competitors reuse the scaling wrappers (see text_shortening README)

python experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py \
  --output-dir experiments/task_comparison/vlm_enrichment/workload     # needs HF_TOKEN
python experiments/task_comparison/vlm_enrichment/scripts/run_mmirage_vlm.py \
  --execution-config experiments/task_comparison/vlm_enrichment/configs/execution_4gpu.yaml
# native VLM competitors:
python experiments/task_comparison/vlm_enrichment/scripts/run_native_vlm_competitor.py \
  --framework sglang \
  --workload-jsonl experiments/task_comparison/vlm_enrichment/workload/rows.jsonl \
  --image-base-path experiments/task_comparison/vlm_enrichment/workload \
  --output-root experiments/task_comparison/vlm_enrichment/results/native_competitors/sglang \
  --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3
# repeat for --framework datatrove and --framework nemo_curator
```

## Static verification of results

For every completed run, confirm before recording it:

- `results/summary.json` parses and its `summary` rows (scaling/text/vlm) or `metrics` (raw) are populated.
- Per-repetition `validation.json` has `"valid": true` (native competitors).
- `processed_rows` equals the prepared workload row count.
- No shard failed (`failed_shards` empty) and exit codes were 0.

## Commit and push

Do not commit `experiments/*/workload/`, `experiments/*/results/`, or any model
outputs (git-ignored). Commit the deltas that ARE intended for the branch:

- any fixes you had to make to scripts or configs (with a message explaining why),
- `experiments/smoke/timing.json` and `experiments/smoke/calibration.json`
  (evidence of the calibration), and a `RUNLOG.md` summarizing wall times.

```
git add experiments/smoke/timing.json experiments/smoke/calibration.json RUNLOG.md
git add experiments/**/configs/  # only if calibrate --apply changed them
git status --short               # confirm no workload/results staged
git commit -m "smoke: calibrated fast-run workload sizes on <pod>"
git push origin experiment/fast-runs
```

Report back a table of: experiment, calibrated size, GPU count, per-command wall
time, throughput metrics from `summary.json`, and validation status.
