# Raw SGLang overhead experiment

This experiment estimates the end-to-end overhead introduced by AnonLib's data
loading, declarative mapping, rendering, state tracking, and statistics. Both
measured paths send the same prompts to the same one-GPU SGLang HTTP endpoint:

1. `raw_sglang`: a minimal `/v1/completions` client;
2. `anonlib_sglang`: the complete AnonLib pipeline.

The experiment reports every repetition and summarizes them with mean and
standard deviation; it never selects the best run.

## Files

| Path | Purpose |
|---|---|
| `workload/` | Bundled 1,000 measured prompts, 16 warm-up prompts, and provenance metadata |
| `configs/anonlib_sglang.yaml` | AnonLib pipeline configuration used by the runner |
| `scripts/prepare_workload.py` | Optionally regenerates a workload from `simplescaling/s1K-1.1` |
| `scripts/run.py` | Starts SGLang, runs both paths, and aggregates results |
| `scripts/raw_sglang_client.py` | Implements the raw comparison path |
| `scripts/run_anonlib_with_sglang_endpoint.py` | Adapts the AnonLib path to the shared HTTP endpoint for this experiment only |

All commands below must be run from the repository root.

## 1. Install and verify dependencies

```bash
python -m pip install -e '.[gpu]'
python -c "import sglang; print(sglang.__version__)"
nvidia-smi
```

The expected SGLang version is `0.5.10`. The runner uses `sglang serve` when the
console script is available and otherwise falls back to
`python -m sglang.launch_server`.

The measured run requires one available GPU, a free local TCP port, and enough
space for `Qwen/Qwen3-4B`. On a multi-GPU node, use `--gpu-index` to pin the
server and both clients to the same physical GPU.

## 2. Validate the experiment without starting a GPU server

The bundled workload can be used directly:

```bash
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir /tmp/anonlib_overhead_dry_run \
  --repetitions 1 \
  --dry-run
```

This checks the workload, resolves the server command, and writes experiment
metadata. It does not start SGLang or run inference.

## 3. Optionally regenerate the workload

Regeneration requires access to Hugging Face. Writing to `/tmp` leaves the
bundled workload unchanged:

```bash
python experiments/raw_sglang_overhead/scripts/prepare_workload.py \
  --output-dir /tmp/anonlib_overhead_workload \
  --num-rows 1000 \
  --warmup-rows 16 \
  --model-path Qwen/Qwen3-4B
```

Use `/tmp/anonlib_overhead_workload` as `--workload-dir` in the next command to
run the regenerated version.

## 4. Run the full one-GPU comparison

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

Omit `--gpu-index` on a single-GPU machine. Each repetition runs the raw path
and the AnonLib path with a freshly started server, so three repetitions load
the model six times.

For an H100 with the dependency versions used for the recorded results, force
the `flashinfer` attention backend:

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

`--server-extra-args-json` replaces the default argument list, so all desired
server arguments must be included. If the installed SGLang CLI differs, use
`--server-command-json` to provide the complete launch command.

## 5. Inspect the outputs

| Output | Contents |
|---|---|
| `raw_results.csv` | One row per repetition and path |
| `summary.json` | Aggregate metrics, throughput retention, overhead, and environment metadata |
| `summary.csv` | Aggregate metrics in tabular form |
| `table.tex` | Paper-ready LaTeX table |
| `plot_throughput.py` | Script that writes `throughput_boxplot.png` when run |
| `rep_*/raw_sglang/` | Raw-client outputs and server logs |
| `rep_*/anonlib_sglang/` | AnonLib outputs, state, statistics, and server logs |

Definitions:

- `throughput retention = mean(AnonLib tok/s) / mean(raw SGLang tok/s)`
- `relative orchestration overhead = 1 - throughput retention`

This is an end-to-end empirical overhead estimate, not a CPU-only orchestration
microbenchmark. Generated result directories are intentionally kept local and
are ignored by Git.

## Recorded reference results

Both recorded runs completed with `1000/1000` rows succeeding in every
repetition (three repetitions each). The table below records the reference
measurements; generated per-run result directories are not committed.

| GPU | Path | Output tok/s/GPU | Rows/s | Gen. wall (s) | Success |
|---|---|---|---|---|---|
| A100 | Raw SGLang 0.5.10 | 5187.44 +/- 2.08 | 5.09 +/- 0.00 | 196.65 +/- 0.10 | 1000/1000 |
| A100 | AnonLib over SGLang | 5026.21 +/- 4.98 | 4.92 +/- 0.01 | 203.00 +/- 0.14 | 1000/1000 |
| H100 | Raw SGLang 0.5.10 | 8781.55 +/- 3.16 | 8.60 +/- 0.00 | 116.28 +/- 0.04 | 1000/1000 |
| H100 | AnonLib over SGLang | 8376.01 +/- 3.36 | 8.20 +/- 0.01 | 121.92 +/- 0.05 | 1000/1000 |

Throughput retention: **0.96892** (A100, 3.1% overhead) and **0.95382** (H100,
4.6% overhead). H100 is ~1.7x faster in absolute token throughput, but the
overhead behaves like a small fixed additive cost (~6 s per 1000 rows on both
GPUs), so the relative cost is slightly larger on the faster H100.

Both runs used the flashinfer attention backend (A100 auto-selected; H100
forced via `--server-extra-args-json`), same model (`Qwen/Qwen3-4B`, revision
`1cfa9a72...`), same dataset (`simplescaling/s1K-1.1`, revision
`96c411f1...`), concurrency 64, `max_tokens=1024`, temperature 0.0.
