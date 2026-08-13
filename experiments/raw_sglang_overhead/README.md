# Raw SGLang Overhead Benchmark

This experiment estimates how much raw SGLang throughput AnonLib retains while adding dataset loading, declarative field mapping, output rendering, state tracking, and statistics.

It directly addresses the current paper's stated RQ3 gap: the existing DataTrove-style result shows that AnonLib can drive a high-throughput SGLang path, but it does not isolate AnonLib overhead under a like-for-like raw SGLang baseline.

This is an end-to-end empirical overhead estimate, not a pure CPU orchestration microbenchmark.

Definitions:

- `AnonLib throughput retention = mean(AnonLib tok/s) / mean(raw SGLang tok/s)`
- `relative orchestration overhead = 1 - throughput_retention`

The scripts report mean +/- standard deviation across all repetitions. They never select the best run.

Important implementation constraint:

- Stock AnonLib text generation constructs an in-process `sgl.Engine` rather than connecting to an OpenAI-compatible HTTP endpoint.
- To keep AnonLib source unchanged while satisfying same-endpoint parity, `scripts/benchmarks/run_anonlib_with_sglang_endpoint.py` patches `sglang.Engine` only inside the benchmark subprocess and forwards AnonLib generation calls to the same external SGLang `/v1/completions` endpoint used by the raw client.
- This wrapper is benchmark scaffolding, not a AnonLib feature change.

PDF discrepancy to preserve:

- The current paper resolves the MedTrinity run as `16 nodes x 1 GPU`.
- The semester report appendix says the 16-node Slurm run used `16 nodes with 4 GH200 GPUs per node`.
- This overhead experiment avoids that allocation ambiguity by targeting one GPU in one Run:ai pod, without Slurm. On nodes with multiple GPUs, pass `--gpu-index <N>` so the server and both paths run on a single pinned GPU (`CUDA_VISIBLE_DEVICES`).

Dependencies / setup:

- The SGLang stack is declared in `pyproject.toml` under the `gpu` optional
  extra (`sglang==0.5.10`, `sgl_kernel`, `xgrammar`, `compressed_tensors`).
  Install it with the project in editable mode:
  `uv pip install -e '.[gpu]'` (or `pip install -e '.[gpu]'`).
- The benchmark runner needs the `sglang` console script on `PATH` (installed
  with the extra); otherwise it falls back to `python -m sglang.launch_server`.
- Verify before running: `python -c "import sglang; print(sglang.__version__)"`
  should print `0.5.10`.

Tiny validation without GPU benchmark execution:

```bash
python scripts/benchmarks/prepare_s1k_overhead_workload.py --output-dir /tmp/anonlib_overhead_tiny --num-rows 2 --warmup-rows 1
python scripts/benchmarks/run_sglang_overhead_benchmark.py --workload-dir /tmp/anonlib_overhead_tiny --output-dir /tmp/anonlib_overhead_tiny_results --repetitions 1 --dry-run
```

Full one-GPU experiment on A100 (`results_a100/`):

```bash
python scripts/benchmarks/prepare_s1k_overhead_workload.py --output-dir experiments/raw_sglang_overhead/workload --num-rows 1000 --warmup-rows 16 --model-path Qwen/Qwen3-4B
python scripts/benchmarks/run_sglang_overhead_benchmark.py --workload-dir experiments/raw_sglang_overhead/workload --output-dir experiments/raw_sglang_overhead/results_a100 --repetitions 3 --gpu-count 1 --concurrency 64 --port 30000 --model-path Qwen/Qwen3-4B --gpu-index 0
```

Full one-GPU experiment on H100 (`results_h100/`) - **requires** forcing the
flashinfer attention backend, otherwise the default sm_90 backend (FA3) crashes
against the installed nvidia-cutlass/cutlass-mlir wheel:

```bash
python scripts/benchmarks/run_sglang_overhead_benchmark.py --workload-dir experiments/raw_sglang_overhead/workload --output-dir experiments/raw_sglang_overhead/results_h100 --repetitions 3 --gpu-count 1 --concurrency 64 --port 30000 --model-path Qwen/Qwen3-4B --gpu-index 0 --server-extra-args-json '["--tp-size","1","--trust-remote-code","--disable-custom-all-reduce","--max-running-requests","1000","--attention-backend","flashinfer"]'
```

`--server-extra-args-json` replaces the default extra args, so the full default
set must be passed explicitly. On A100 (sm_80) SGLang auto-selects `flashinfer`,
so both runs use the same attention backend.

The runner uses `sglang serve` when the console script exists on `PATH`; otherwise it falls back to `python -m sglang.launch_server` from the active Python environment.

If the installed SGLang CLI uses different flag names, keep both paths matched by passing the exact command once:

```bash
python scripts/benchmarks/run_sglang_overhead_benchmark.py --workload-dir experiments/raw_sglang_overhead/workload --output-dir experiments/raw_sglang_overhead/results_h100 --server-command-json '["python", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-4B", "--host", "127.0.0.1", "--port", "30000", "--tp-size", "1", "--trust-remote-code", "--disable-custom-all-reduce", "--max-running-requests", "1000"]'
```

Expected outputs:

- `raw_results.csv`
- `summary.json`
- `summary.csv`
- `table.tex`
- `plot_throughput.py`

## Results

Both runs completed with `1000/1000` rows succeeding in every repetition (3 reps
each). Results are kept under `results_a100/` and `results_h100/`; a merged,
paper-ready evidence package is in `paper_evidence/` (with a zip archive
`anonlib_vs_sglang_evidence.zip`).

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
