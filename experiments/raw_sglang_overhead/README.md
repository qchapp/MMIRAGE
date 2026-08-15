# Raw SGLang Overhead (hardware comparison)

Measures how much throughput MMIRAGE retains relative to minimal framework
baselines, on one GPU, at the same model, workload, generation settings, and
concurrency. One fresh serving stack per path per repetition.

Paths (`--frameworks`): `raw_sglang` (direct `/v1/completions`),
`mmirage_sglang` (MMIRAGE forwarding its SGLang calls to the endpoint),
`datatrove`, `nemo_curator` (self-managed vLLM via `native_shard_worker.py`).

Primary metrics: `output_tokens_per_second_per_gpu`, `rows_per_second`,
`throughput_retention` (= MMIRAGE/raw tok/s), `relative_orchestration_overhead`
(= 1 - retention). Each cell records model startup separately
(`model_loading_seconds`), so generation wall excludes loading.

Fast-run deviation: one repetition per path (the comparison needs no
repetition); the committed workload size is calibrated by `experiments/smoke`.

## Run

From the repository root:

```
python experiments/raw_sglang_overhead/scripts/prepare_workload.py \
  --output-dir experiments/raw_sglang_overhead/workload \
  --num-rows 1000
python experiments/raw_sglang_overhead/scripts/run.py \
  --workload-dir experiments/raw_sglang_overhead/workload \
  --output-dir experiments/raw_sglang_overhead/results
```

Add `--gpu-index <n>` on a multi-GPU node, or `--server-extra-args-json
'["--tp-size","1","--trust-remote-code","--disable-custom-all-reduce",
"--max-running-requests","1000","--attention-backend","flashinfer"]'` to force
the H100-recorded `flashinfer` backend. `--dry-run` prints the server command
without starting SGLang.

The default `--frameworks raw_sglang,mmirage_sglang` runs entirely in the
MMIRAGE venv. The `datatrove` / `nemo_curator` paths need their framework (plus
a `vllm` CLI) importable, which lives in a separate venv per
`../single_node_h100_scaling/environment/<name>_uv_requirements.txt`; point the
runner at those interpreters with `--datatrove-python` / `--nemo-curator-python`
or the `MMIRAGE_DATATROVE_PYTHON` / `MMIRAGE_NEMO_CURATOR_PYTHON` environment
variables, then add `--frameworks raw_sglang,mmirage_sglang,datatrove,nemo_curator`.

## Outputs

`summary.json` (throughput retention + overhead + environment metadata),
`summary.csv`, `raw_results.csv` (per rep/path), `table.tex`,
`plot_throughput.py`, `rep_*/<path>/` (logs and outputs). The committed
`workload/` is the 1000-row paper workload; `prepare_workload.py` regenerates it.

## Recorded reference (paper Table 4, 1000 rows / 1024 tok)

| GPU | Path | Output tok/s/GPU | Rows/s | Gen. wall (s) | Success |
|---|---|---|---|---|---|
| A100 | Raw SGLang 0.5.10 | 5187.44 +/- 2.08 | 5.09 | 196.65 | 1000/1000 |
| A100 | MMIRAGE over SGLang | 5026.21 +/- 4.98 | 4.92 | 203.00 | 1000/1000 |
| H100 | Raw SGLang 0.5.10 | 8781.55 +/- 3.16 | 8.60 | 116.28 | 1000/1000 |
| H100 | MMIRAGE over SGLang | 8376.01 +/- 3.36 | 8.20 | 121.92 | 1000/1000 |

Throughput retention: 0.96892 (A100), 0.95382 (H100). End-to-end overhead
estimate, not a CPU-only orchestration microbenchmark.

## Common failures

- SGLang never ready: check `rep_*/<path>/sglang_server.log`, verify HF access, GPU memory.
- Port bind error: pass a free `--port`.
- Mixed old/new outputs: use a fresh `--output-dir` per run.
- Throughput far below reference: wrong GPU, missing `flashinfer` on H100, or noisy node.
