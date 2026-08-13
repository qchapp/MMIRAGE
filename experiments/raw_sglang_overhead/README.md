# Raw SGLang Overhead Experiment

This experiment measures how much throughput AnonLib retains relative to a minimal raw SGLang client when both use the same one-GPU SGLang HTTP server, model, workload, generation settings, and concurrency.

Run all commands from the repository root.

## What It Measures

The runner compares two paths:

1. `raw_sglang`: sends prompts directly to `/v1/completions`.
2. `anonlib_sglang`: runs the complete AnonLib pipeline while forwarding AnonLib SGLang calls to the same HTTP endpoint.

The reported metrics are based on every repetition. The scripts do not select the best run.

Primary metrics:

- `output_tokens_per_second_per_gpu`: measured output-token throughput for each path.
- `rows_per_second`: completed prompt rows per second.
- `throughput_retention`: mean AnonLib throughput divided by mean raw SGLang throughput.
- `relative_orchestration_overhead`: `1 - throughput_retention`.
- `generation_wall_seconds`: measured generation time, excluding model startup.

This experiment does not measure image build time, dependency installation, dataset preparation, model download time, or cold cluster scheduling. Each measured path gets a fresh SGLang server per repetition, so model-load time is recorded separately and is not part of the throughput-retention numerator.

## Files

| Path | Purpose |
|---|---|
| `workload/` | Committed 1,000 measured prompts, 16 warm-up prompts, and provenance metadata. |
| `configs/anonlib_sglang.yaml` | AnonLib config used by the AnonLib path. |
| `scripts/run.py` | Starts SGLang, runs both paths, and writes summaries. |
| `scripts/prepare_workload.py` | Optional workload regeneration script. |
| `scripts/raw_sglang_client.py` | Raw baseline client. |
| `scripts/run_anonlib_with_sglang_endpoint.py` | Benchmark-only adapter that forwards AnonLib SGLang calls to the HTTP endpoint. |

## Prerequisites

- One available CUDA GPU with enough memory for `Qwen/Qwen3-4B`.
- A free local TCP port, default `30000`. If that port is busy, use the same alternate `--port` value for the command you run.
- Python environment with AnonLib GPU dependencies and SGLang `0.5.10`.
- Hugging Face access to `Qwen/Qwen3-4B` if the model is not already cached.

Install and check the environment:

```bash
python -m pip install -e '.[gpu]'
python -c "import sglang; print(sglang.__version__)"
nvidia-smi
```

Expected SGLang version: `0.5.10`.

## 1. Smoke Check Without Starting SGLang

This validates paths, workload metadata, server command construction, and output writing. It does not start a model server and does not run inference.

```bash
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir /tmp/anonlib_overhead_dry_run \
  --repetitions 1 \
  --dry-run
```

Expected result: the command prints JSON with `"dry_run": true` and writes `/tmp/anonlib_overhead_dry_run/experiment_metadata.json`. Delete or change `/tmp/anonlib_overhead_dry_run` before repeating the smoke check if you want a clean output directory.

## 2. Choose The Workload

Use the committed workload unless you need to regenerate it. The committed workload is the default for reproduced paper numbers.

To regenerate the same-size workload in `/tmp`:

```bash
python experiments/raw_sglang_overhead/scripts/prepare_workload.py \
  --output-dir /tmp/anonlib_overhead_workload \
  --num-rows 1000 \
  --warmup-rows 16 \
  --model-path Qwen/Qwen3-4B
```

If you regenerate, replace `experiments/raw_sglang_overhead/workload` with `/tmp/anonlib_overhead_workload` in the full run command.

## 3. Run The Full Experiment

Use a fresh `--output-dir` for each full experiment run. On a single-GPU machine, omit `--gpu-index`. On a multi-GPU machine, keep `--gpu-index 0` or replace `0` with the GPU to use.

```bash
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir /tmp/anonlib_overhead_results \
  --repetitions 3 \
  --gpu-count 1 \
  --gpu-index 0 \
  --concurrency 64 \
  --port 30000 \
  --model-path Qwen/Qwen3-4B
```

Each repetition starts one fresh SGLang server for `raw_sglang` and one fresh SGLang server for `anonlib_sglang`. With `--repetitions 3`, the model is loaded six times.

For the recorded H100 setup, force the `flashinfer` attention backend:

```bash
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir /tmp/anonlib_overhead_results_h100 \
  --repetitions 3 \
  --gpu-count 1 \
  --gpu-index 0 \
  --concurrency 64 \
  --port 30000 \
  --model-path Qwen/Qwen3-4B \
  --server-extra-args-json '["--tp-size","1","--trust-remote-code","--disable-custom-all-reduce","--max-running-requests","1000","--attention-backend","flashinfer"]'
```

`--server-extra-args-json` replaces the default SGLang server arguments. Include every server flag you want to pass.

## 4. Inspect Outputs

The output directory contains:

| Output | Contents |
|---|---|
| `raw_results.csv` | One row per repetition and path. |
| `summary.json` | Full metrics, throughput retention, overhead, and environment metadata. |
| `summary.csv` | Aggregate metrics in CSV form. |
| `table.tex` | Paper-ready LaTeX table. |
| `plot_throughput.py` | Plot script for `throughput_boxplot.png`. |
| `rep_*/raw_sglang/` | Raw-client outputs and SGLang logs. |
| `rep_*/anonlib_sglang/` | AnonLib outputs, state, stats, and SGLang logs. |

Expected tree after the full command with three repetitions:

```text
/tmp/anonlib_overhead_results/
  experiment_metadata.json
  raw_results.csv
  summary.csv
  summary.json
  table.tex
  plot_throughput.py
  rep_1/
    raw_sglang/
      outputs.jsonl
      summary.json
      sglang_server.log
      warmup/
    anonlib_sglang/
      output/
      state/
      wrapper_summary.json
      sglang_server.log
      warmup/
  rep_2/
  rep_3/
```

Key definitions:

- `throughput retention = mean(AnonLib tok/s) / mean(raw SGLang tok/s)`
- `relative orchestration overhead = 1 - throughput retention`

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| SGLang never becomes ready | Model download is slow, model path is unavailable, or GPU memory is insufficient. | Check `rep_*/<path>/sglang_server.log`, verify Hugging Face access, and rerun on a GPU with enough memory. |
| Port bind error | Another process is using `--port 30000`. | Rerun with a free `--port` value. |
| `output_dir` contains mixed old and new files | A previous run used the same directory. | Use a fresh `--output-dir` for full runs. |
| Raw and AnonLib success counts differ | One path failed rows or timed out. | Inspect `raw_results.csv`, `rep_*/raw_sglang/summary.json`, and `rep_*/anonlib_sglang/state/shard_0/status.json`. |
| Throughput is far below the recorded reference | Wrong GPU, missing `flashinfer` backend on H100, or noisy shared node. | Record `experiment_metadata.json`, check `nvidia-smi`, and use the H100 command if reproducing H100 numbers. |

## Reproducibility Metadata

Keep these with any reported result:

- Git commit from `experiment_metadata.json` field `anonlib_commit`.
- SGLang version from `experiment_metadata.json` field `sglang_version`.
- GPU names, CUDA version, and driver from `experiment_metadata.json` field `gpu_cuda_driver`.
- Exact server command from `experiment_metadata.json` field `server_arguments`.
- Workload metadata and model/dataset revisions from `experiment_metadata.json`.
- Output directory checksum or archived copy containing `raw_results.csv`, `summary.json`, and all `rep_*/` logs.

## Paper Artifact Mapping

Use these generated files for paper artifacts:

| Paper artifact | Source file |
|---|---|
| Raw per-repetition data | `raw_results.csv` |
| Aggregate numbers and overhead | `summary.json` and `summary.csv` |
| Paper table fragment | `table.tex` |
| Optional throughput plot | Run `python plot_throughput.py` inside the output directory to create `throughput_boxplot.png`. |

Fill in the final paper figure or table number here after manuscript numbering is fixed.

## Recorded Reference Results

Both reference runs completed with `1000/1000` rows succeeding in every repetition.

| GPU | Path | Output tok/s/GPU | Rows/s | Gen. wall (s) | Success |
|---|---|---|---|---|---|
| A100 | Raw SGLang 0.5.10 | 5187.44 +/- 2.08 | 5.09 +/- 0.00 | 196.65 +/- 0.10 | 1000/1000 |
| A100 | AnonLib over SGLang | 5026.21 +/- 4.98 | 4.92 +/- 0.01 | 203.00 +/- 0.14 | 1000/1000 |
| H100 | Raw SGLang 0.5.10 | 8781.55 +/- 3.16 | 8.60 +/- 0.00 | 116.28 +/- 0.04 | 1000/1000 |
| H100 | AnonLib over SGLang | 8376.01 +/- 3.36 | 8.20 +/- 0.01 | 121.92 +/- 0.05 | 1000/1000 |

Throughput retention was `0.96892` on A100 and `0.95382` on H100. This is an end-to-end overhead estimate, not a CPU-only orchestration microbenchmark.
