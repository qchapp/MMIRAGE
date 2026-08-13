# Single-Node H100 Strong-Scaling Experiment

This experiment measures AnonLib throughput as the number of independent one-GPU shard workers increases on one node. It is single-node data-parallel scaling only. Do not use it as evidence for multi-node scaling.

Run all commands from the repository root inside the AnonLib GPU environment.

## What It Measures

The total JSONL workload is fixed. Only the number of independent AnonLib shard workers changes:

| Point | Logical shards | Visible GPUs |
|---|---:|---|
| `1gpu` | 1 | `0` |
| `2gpu` | 2 | `0,1` |
| `4gpu` | 4 | `0,1,2,3` |

Each worker is pinned with `CUDA_VISIBLE_DEVICES=<gpu_id>`, uses `tp_size: 1`, and runs the same semantic recipe. The model is not tensor-parallelized.

## Files

| Path | Purpose |
|---|---|
| `configs/semantic_recipe.yaml` | Fixed AnonLib transformation used for every GPU count. |
| `configs/execution_1gpu.yaml` | Execution settings for the 1-GPU point. |
| `configs/execution_2gpu.yaml` | Execution settings for the 2-GPU point. |
| `configs/execution_4gpu.yaml` | Execution settings for the 4-GPU point. |
| `scripts/prepare_workload.py` | Builds the deterministic UltraChat JSONL workload. |
| `scripts/run.py` | Launches shard workers and aggregates summaries. |
| `scripts/run_1gpu.sh` | Wrapper for the 1-GPU point. |
| `scripts/run_2gpu.sh` | Wrapper for the 2-GPU point. |
| `scripts/run_4gpu.sh` | Wrapper for the 4-GPU point. |
| `scripts/plot.py` | Regenerates plots from `summary.csv`. |

## Prerequisites

- One terminal inside a pod or job with 4 visible H100 GPUs for the full run. The 1-GPU and 2-GPU commands use subsets of those visible devices.
- Python environment with AnonLib GPU dependencies, SGLang, CUDA-compatible PyTorch, `datasets`, and `matplotlib`.
- Hugging Face access to `HuggingFaceH4/ultrachat_200k` and `Qwen/Qwen3-4B` if they are not cached.
- Enough local or shared storage for the workload, model cache, and result directories.

Check the environment:

```bash
python -m pip install -e '.[gpu]'
python -c "import anonlib, sglang, torch; print('anonlib/sglang/torch ok', torch.cuda.device_count())"
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

This checks config composition and the commands that would be launched. It does not start SGLang or run AnonLib shard processing.

```bash
bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh --dry-run
```

Expected result: JSON listing four shard commands and configs under `experiments/single_node_h100_scaling/results/runs/gpu_4/dry_run/`.

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

Important metrics:

- `aggregate_output_tok_s = total_output_tokens / end_to_end_wall_seconds`
- `output_tok_s_per_gpu = aggregate_output_tok_s / gpu_count`
- `speedup_vs_1gpu = mean aggregate_output_tok_s at N GPUs / 1-GPU mean`
- `parallel_efficiency = speedup_vs_1gpu / gpu_count`

## Interpretation Boundary

This experiment supports a controlled single-node multi-GPU scaling claim. It does not resolve or depend on historical MedTrinity allocation discrepancies, and it does not measure multi-node cluster scaling.
