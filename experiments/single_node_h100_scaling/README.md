# Single-Node H100 Strong-Scaling Experiment

This experiment measures MMIRAGE throughput as the number of independent one-GPU shard workers increases on one node. It is single-node data-parallel scaling only. Do not use it as evidence for multi-node scaling.

Run all commands from the repository root inside the MMIRAGE GPU environment.

## What It Measures

The total JSONL workload is fixed. Only the number of independent MMIRAGE shard workers changes:

| Point | Logical shards | Visible GPUs |
|---|---:|---|
| `1gpu` | 1 | `0` |
| `2gpu` | 2 | `0,1` |
| `4gpu` | 4 | `0,1,2,3` |

Each worker is pinned with `CUDA_VISIBLE_DEVICES=<gpu_id>`, uses `tp_size: 1`, and runs the same semantic recipe. The model is not tensor-parallelized.

Primary metrics:

- `aggregate_output_tok_s`: total output-token throughput across all shard workers.
- `output_tok_s_per_gpu`: aggregate throughput divided by GPU count.
- `rows_s`: completed workload rows per second.
- `speedup_vs_1gpu`: mean aggregate throughput divided by the 1-GPU mean.
- `parallel_efficiency`: `speedup_vs_1gpu / gpu_count`.

This experiment does not measure multi-node scaling, tensor parallelism, Kubernetes scheduling, image startup, dataset preparation, or model download time. The only intended independent variable is the number of one-GPU MMIRAGE shard workers on one H100 node.

## Files

| Path | Purpose |
|---|---|
| `configs/semantic_recipe.yaml` | Fixed MMIRAGE transformation used for every GPU count. |
| `configs/execution_1gpu.yaml` | Execution settings for the 1-GPU point. |
| `configs/execution_2gpu.yaml` | Execution settings for the 2-GPU point. |
| `configs/execution_4gpu.yaml` | Execution settings for the 4-GPU point. |
| `configs/native_competitors.yaml` | Native-mode completion settings for DataTrove, NeMo Curator, Distilabel, Ray Data LLM, and raw SGLang baselines on the same task. |
| `scripts/prepare_workload.py` | Builds the deterministic UltraChat JSONL workload. |
| `scripts/run.py` | Launches shard workers and aggregates summaries. |
| `scripts/run_1gpu.sh` | Wrapper for the 1-GPU point. |
| `scripts/run_2gpu.sh` | Wrapper for the 2-GPU point. |
| `scripts/run_4gpu.sh` | Wrapper for the 4-GPU point. |
| `scripts/run_native_text_competitor.py` | Native-mode orchestrator: shards the workload, launches one worker per GPU, merges, validates, and aggregates. |
| `scripts/native_shard_worker.py` | Native-mode per-GPU worker subprocess that runs one framework backend and writes ANONLIB-compatible status files. |
| `scripts/run_datatrove_scaling.py` | Wrapper that runs the DataTrove native baseline (see section 6). |
| `scripts/run_nemo_curator_scaling.py` | Wrapper that runs the NeMo Curator native baseline (see section 6). |
| `scripts/run_distilabel_scaling.py` | Wrapper that runs the Distilabel native baseline (see section 6). |
| `scripts/run_ray_data_llm_scaling.py` | Wrapper that runs the Ray Data LLM native baseline (see section 6). |
| `scripts/run_raw_sglang_scaling.py` | Wrapper that runs the raw SGLang native baseline (see section 6). |
| `scripts/plan_native_competitors.py` | Emits dry-run manifests for native competitor runs without launching GPU work. |
| `scripts/plot.py` | Regenerates plots from `summary.csv`. |
| `environment/` | Per-framework requirement pins for the native competitor environments. |
| `experiments/_shared/native_frameworks.py` | Shared `run_<framework>(...)` implementations and output-contract helpers for all native baselines. |

## Prerequisites

- One terminal inside a pod or job with 4 visible H100 GPUs for the full run. The 1-GPU and 2-GPU commands use subsets of those visible devices.
- Python environment with MMIRAGE GPU dependencies, SGLang, CUDA-compatible PyTorch, `datasets`, and `matplotlib`.
  `matplotlib` is not part of any MMIRAGE extra and must be installed separately. Without it a run
  still writes every summary, but no plot files are produced.
- Hugging Face access to `HuggingFaceH4/ultrachat_200k` and `Qwen/Qwen3-4B` if they are not cached.
- Enough local or shared storage for the workload, model cache, and result directories.

Check the environment:

```bash
python -m pip install -e '.[gpu]'
python -m pip install matplotlib
python -c "import mmirage, sglang, torch; print('mmirage/sglang/torch ok', torch.cuda.device_count())"
nvidia-smi
```

Expected for the full experiment: at least 4 visible H100 GPUs.

## 1. Prepare The Workload

Use 30,000 unique UltraChat prompts by default. This writes the workload and metadata under `experiments/single_node_h100_scaling/workload/`.

```bash
python experiments/single_node_h100_scaling/scripts/prepare_workload.py \
  --output-dir experiments/single_node_h100_scaling/workload \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --num-rows 30000 \
  --model-path Qwen/Qwen3-4B
```

Expected files:

```text
experiments/single_node_h100_scaling/workload/workload.jsonl
experiments/single_node_h100_scaling/workload/warmup.jsonl
experiments/single_node_h100_scaling/workload/metadata.json
```

If the run is too short for stable measurements, regenerate with `--num-rows 50000` before running the benchmark.

## 2. Smoke Check Without Running Inference

This checks config composition and the commands that would be launched. It does not start SGLang or run MMIRAGE shard processing.

```bash
bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh --dry-run
```

Expected result: JSON listing four shard commands and configs under `experiments/single_node_h100_scaling/results/runs/gpu_4/dry_run/`.

To inspect the planned native competitor commands without launching any framework or inference backend:

```bash
python experiments/single_node_h100_scaling/scripts/plan_native_competitors.py \
  --framework all \
  --gpu-count all \
  --visible-gpus 0,1,2,3
```

The native competitor settings cover DataTrove, NeMo Curator, Distilabel, Ray Data LLM, and raw SGLang. They use the same UltraChat workload, prompt, model family, GPU points, shard split, and output schema. See section 6 for how to install their environments and run them.

## 3. Run The Three Scaling Points

Run the points from the same 4-H100 pod or from equivalent pods that write to the same result directory. The default result directory is `experiments/single_node_h100_scaling/results/`.

```bash
bash experiments/single_node_h100_scaling/scripts/run_1gpu.sh
bash experiments/single_node_h100_scaling/scripts/run_2gpu.sh
bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh
```

Each wrapper runs three repetitions by default. Use a fresh `experiments/single_node_h100_scaling/results/` directory for a clean rerun, or pass `--overwrite` for the specific GPU point you want to replace.

If you rerun a point and want to replace its existing repetition directories, pass `--overwrite`:

```bash
bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh --overwrite
```

To change repetitions without editing configs, pass `--repetitions`:

```bash
bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh --repetitions 5
```

## 4. Regenerate Summaries Or Plots

The runner updates CSV, JSON, LaTeX, and plot outputs after each repetition. To regenerate summaries from existing `rep_summary.json` files without launching workers, pass any one execution config so the runner can locate the experiment defaults:

```bash
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/single_node_h100_scaling/configs/execution_4gpu.yaml \
  --aggregate-only
```

To regenerate plots from `summary.csv`:

```bash
python experiments/single_node_h100_scaling/scripts/plot.py \
  --summary-csv experiments/single_node_h100_scaling/results/summary.csv \
  --output-dir experiments/single_node_h100_scaling/results
```

## 5. Inspect Outputs

Generated files under `experiments/single_node_h100_scaling/results/`:

| Output | Contents |
|---|---|
| `raw_results.csv` | One row per GPU-count repetition. |
| `summary.csv` | Mean and sample standard deviation per GPU count. |
| `summary.json` | Full structured summary and environment metadata. |
| `latex_table.txt` | Paper-ready LaTeX table fragment. |
| `aggregate_throughput_vs_gpu.png` | Aggregate output-token throughput plot. |
| `parallel_efficiency_vs_gpu.png` | Parallel efficiency plot. |
| `runs/gpu_<N>/rep_<R>/` | Per-repetition shard configs, logs, state, outputs, and summaries. |

Expected tree after all three scaling points complete:

```text
experiments/single_node_h100_scaling/results/
  experiment_metadata.json
  raw_results.csv
  summary.csv
  summary.json
  latex_table.txt
  aggregate_throughput_vs_gpu.png
  parallel_efficiency_vs_gpu.png
  runs/
    gpu_1/
      rep_1/
        run_manifest.json
        rep_summary.json
        configs/
        logs/
        output/
        state/
      rep_2/
      rep_3/
    gpu_2/
      rep_1/
      rep_2/
      rep_3/
    gpu_4/
      rep_1/
      rep_2/
      rep_3/
```

Important metrics:

- `aggregate_output_tok_s = total_output_tokens / end_to_end_wall_seconds`
- `output_tok_s_per_gpu = aggregate_output_tok_s / gpu_count`
- `speedup_vs_1gpu = mean aggregate_output_tok_s at N GPUs / 1-GPU mean`
- `parallel_efficiency = speedup_vs_1gpu / gpu_count`

## 6. Native Competitor Baselines

The native baselines answer the same strong-scaling question using each framework's own inference stack instead of AnonLib. They reuse the same workload, prompt template, model, GPU points, shard split, and output contract. Each framework runs in its own Python environment; nothing here imports or executes AnonLib.

### 6.1 What runs where

| Framework | Inference backend | Wrapper script |
|---|---|---|
| DataTrove | `InferenceRunner` against a self-managed `vllm serve` | `scripts/run_datatrove_scaling.py` |
| NeMo Curator | `nemo_curator.pipeline` with `OpenAIClient` against a self-managed `vllm serve` | `scripts/run_nemo_curator_scaling.py` |
| Distilabel | in-process vLLM-backed `TextGeneration` step | `scripts/run_distilabel_scaling.py` |
| Ray Data LLM | `ray.data.llm` `vLLMEngineProcessorConfig` | `scripts/run_ray_data_llm_scaling.py` |
| raw SGLang | one in-process `sglang.Engine` per shard worker | `scripts/run_raw_sglang_scaling.py` |

All five share the same plumbing:

- `scripts/native_shard_worker.py` runs the framework backend for one shard and writes ANONLIB-compatible `running.json`/`status.json` plus the contract output.
- `scripts/run_native_text_competitor.py` splits the workload into one contiguous shard per visible GPU, launches one worker subprocess per GPU pinned with `CUDA_VISIBLE_DEVICES`, merges shard outputs in input order, validates the contract, and reuses `run.py` for aggregation.
- `experiments/_shared/native_frameworks.py` holds the `run_<framework>(...)` implementations, the vLLM server helpers, and the contract validator.

### 6.2 Install the framework environments

Each framework pins a different inference stack, so create one uv virtualenv per framework (Python 3.12) and install its requirement file:

```bash
uv venv --python 3.12 .venv-datatrove
uv pip install --python .venv-datatrove/bin/python -r experiments/single_node_h100_scaling/environment/datatrove_uv_requirements.txt
uv pip install --python .venv-datatrove/bin/python "setuptools<76"
```

Repeat for `distilabel`, `nemo_curator`, `ray_data_llm`, and `raw_sglang` using their matching `environment/<name>_uv_requirements.txt`. Versions used and verified for this experiment:

| Environment | Key packages |
|---|---|
| `.venv-anonlib` | sglang 0.5.10 |
| `.venv-datatrove` | datatrove 0.9.0, vllm 0.23.0, setuptools 75.9.1 |
| `.venv-distilabel` | distilabel 1.5.3, vllm 0.27.1 |
| `.venv-ray` | ray 2.57.0, vllm 0.27.1 |
| `.venv-nemo` | nemo-curator 1.3.0, ray 2.57.0 |

Environment notes:

- `.venv-datatrove` needs `setuptools<76`: setuptools 76 removed the vendored `distutils` that DataTrove's dependency stack still imports on Python 3.12; the `setuptools<76` pin above keeps the distutils shim available.
- If your shell exports `SETUPTOOLS_USE_DISTUTILS=stdlib` (some shared environments do), setuptools will refuse to install a working distutils shim. `experiments/_shared/native_frameworks.py` normalizes this back to `local` when it loads, and the vLLM servers it spawns inherit the same setting, so no manual workaround is needed.
- vLLM 0.23 removed the CLI flags that DataTrove 0.9.0's bundled `VLLMServer` passes, which is why every vLLM-backed path spawns its own `vllm serve` (current flags) instead of using a framework-managed server. The server also requires a real `vllm` executable on `PATH` (or `<venv>/bin/vllm`); `python -m vllm` does not work in vLLM 0.23.
- The workload and model are read from the Hugging Face cache. Pre-cache `Qwen/Qwen3-4B` in each environment (or point `HF_HOME` at a shared cache) before running so the first shard does not download.

### 6.3 Run a point

Run each point with that framework's venv python so the shard workers inherit the framework environment (the orchestrator defaults `--worker-python` to `sys.executable`):

```bash
.venv-datatrove/bin/python experiments/single_node_h100_scaling/scripts/run_datatrove_scaling.py \
  --workload-jsonl experiments/single_node_h100_scaling/workload/workload.jsonl \
  --output-root experiments/single_node_h100_scaling/results/native_competitors/datatrove \
  --gpu-count 1 \
  --visible-gpus 0 \
  --repetitions 3 \
  --model Qwen/Qwen3-4B
```

Swap `run_datatrove_scaling.py` for `run_nemo_curator_scaling.py`, `run_distilabel_scaling.py`, `run_ray_data_llm_scaling.py`, or `run_raw_sglang_scaling.py` (and the venv) for the other frameworks. Alternatively keep the orchestrator in `.venv-anonlib` and pass `--worker-python .venv-datatrove/bin/python`.

Useful flags:

- `--dry-run` prints the run manifest and exits without launching GPU work.
- `--overwrite` replaces existing repetition directories for that framework/GPU point.
- `--repetitions N` changes the repetition count.
- `--aggregate-only` re-aggregates an existing output root without running workers.

To print planned manifests for every framework and GPU point without launching anything, use the plan script (also section 2):

```bash
python experiments/single_node_h100_scaling/scripts/plan_native_competitors.py \
  --framework all \
  --gpu-count all \
  --visible-gpus 0,1,2,3
```

### 6.4 Output contract and validation

Each repetition writes per shard under `runs/gpu_<N>/rep_<R>/state/shard_<i>/`: `input.jsonl`, `output.jsonl`, `running.json`, `status.json`, and `worker.log` (plus `worker.vllm.log` where a vLLM server is spawned). The orchestrator merges shard outputs in input order into `runs/gpu_<N>/rep_<R>/output/native_competitor_output.jsonl`, writes `validation.json`, and produces the same ANONLIB repetition summary and `summary.csv`/`summary.json` outputs as the AnonLib run.

Output rows are one JSON line per workload row with the contract fields `stable_id`, `source_index`, `prompt_sha256`, `prompt_text`, `answer`. Validation checks that the row count, id set, row order, and prompt hashes match the input (`processed_rows_equals_input_rows`, `stable_id_set_matches_input`, `no_duplicate_stable_ids`, `prompt_sha256_preserved`, `schema_valid_for_every_row`, plus `row_order_matches_input`). Decoding is greedy (`temperature=0.0`, `max_new_tokens=256`), so generation is deterministic per model and framework. Each repetition prints `validation=PASS|FAIL`.

### 6.5 Verification status

Smoke runs on one H100 exercised the full path (worker subprocess, GPU inference, merge, validation, aggregation) for the DataTrove and raw SGLang scaling points, and for the DataTrove ChartQA pipeline (see `experiments/nemo_curator_comparison/README.md`). The 2-GPU and 4-GPU points are the same code path with more shard workers and require 2 and 4 visible GPUs respectively. The NeMo Curator, Distilabel, and Ray Data LLM backends follow the identical orchestrator path but have not been GPU smoke-run.

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| 4-GPU wrapper fails immediately | Fewer than four visible CUDA devices. | Run `nvidia-smi` and start the job in a pod with four visible H100s. |
| A point refuses to rerun | Existing `runs/gpu_<N>/rep_<R>/` directories are present. | Use a fresh `results/` directory or pass `--overwrite` for the point being replaced. |
| A shard fails and `failed_launch.json` appears | Worker process exited nonzero. | Inspect `runs/gpu_<N>/rep_<R>/logs/` and the shard `state/` directories. |
| Results are noisy or efficiency is unexpectedly low | Shared-node interference, wrong GPU type, or different workload size. | Record `experiment_metadata.json`, verify H100 devices, and keep workload metadata with the result. |
| Plot files are missing | `matplotlib` is unavailable or plotting failed after summary generation. | Install `matplotlib` and rerun `scripts/plot.py` from `summary.csv`. |

## Reproducibility Metadata

Keep these with any reported result:

- Git commit of the branch used to run the wrappers.
- `experiment_metadata.json`, including visible GPU IDs, environment metadata, batch size, repetitions, and workload metadata.
- `workload/metadata.json`, including dataset/model revision information.
- Exact execution configs under `configs/execution_*.yaml` and semantic recipe under `configs/semantic_recipe.yaml`.
- Per-repetition `run_manifest.json`, `rep_summary.json`, logs, state, and output directories.
- Final `raw_results.csv`, `summary.csv`, `summary.json`, `latex_table.txt`, and generated plots.

## Paper Artifact Mapping

Use these generated files for paper artifacts:

| Paper artifact | Source file |
|---|---|
| Per-repetition scaling data | `experiments/single_node_h100_scaling/results/raw_results.csv` |
| Aggregate scaling table | `experiments/single_node_h100_scaling/results/summary.csv` and `summary.json` |
| Paper table fragment | `experiments/single_node_h100_scaling/results/latex_table.txt` |
| Throughput-vs-GPU figure | `experiments/single_node_h100_scaling/results/aggregate_throughput_vs_gpu.png` |
| Parallel-efficiency figure | `experiments/single_node_h100_scaling/results/parallel_efficiency_vs_gpu.png` |

## Interpretation Boundary

This experiment supports a controlled single-node multi-GPU scaling claim. It does not resolve or depend on historical MedTrinity allocation discrepancies, and it does not measure multi-node cluster scaling.
