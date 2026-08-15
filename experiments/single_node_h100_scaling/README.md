# Single-Node H100 Strong Scaling

Single-node, data-parallel MMIRAGE scaling: fixed JSONL workload, 1 / 2 / 4
one-GPU MMIRAGE shard workers on one node. Not multi-node, not tensor-parallel.
Workload size lives in `configs/workload_size.yaml` (`num_rows`, written by
`experiments/smoke/calibrate.py`; prepare with no flags to use it).

Metrics: `aggregate_output_tok_s`, `output_tok_s_per_gpu`, `rows_s`,
`speedup_vs_1gpu`, `parallel_efficiency` (= speedup / gpu_count).

## Run

From the repository root, inside the MMIRAGE GPU environment:

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

`bash scripts/run_<N>gpu.sh` are equivalent wrappers. `--dry-run` prints the
launched commands; `--overwrite` replaces existing repetition dirs;
`--repetitions N` overrides the config; `--aggregate-only` re-aggregates
existing `rep_summary.json` files. `scripts/plot.py --summary-csv results/summary.csv
--output-dir results` regenerates plots.

## Outputs

`results/summary.json`, `results/summary.csv`, `results/raw_results.csv`,
`results/latex_table.txt`, two PNG plots, and `results/runs/gpu_<N>/rep_<R>/`
per-repetition configs/logs/state/output.

## Native competitors (same task, same contract)

Reuse the same workload, prompt, model, GPU points, shard split, and output
contract (`stable_id`, `source_index`, `prompt_sha256`, `prompt_text`,
`answer`). Frameworks: DataTrove, NeMo Curator, Distilabel, Ray Data LLM, raw
SGLang — each in its own uv venv from
`environment/<name>_uv_requirements.txt` (Python 3.12) plus
`uv pip install "setuptools<76"`.

```
<venv>/bin/python scripts/run_<framework>_scaling.py \
  --workload-jsonl experiments/single_node_h100_scaling/workload/workload.jsonl \
  --output-root experiments/single_node_h100_scaling/results/native_competitors/<framework> \
  --gpu-count 1 --visible-gpus 0 --repetitions 3 --model Qwen/Qwen3-4B
```

Environment notes:

- `setuptools<76` keeps the distutils shim DataTrove's stack needs;
  `experiments/_shared/native_frameworks.py` forces
  `SETUPTOOLS_USE_DISTUTILS=local` when spawning servers.
- vLLM 0.23 requires a real `vllm` executable on `PATH` (`python -m vllm` fails).
- Pre-cache `Qwen/Qwen3-4B` per venv (or point `HF_HOME` at a shared cache).
- Use `--worker-python <venv>/bin/python` if the orchestrator runs in another venv.

Each repetition writes per-shard `state/shard_<i>/{input,output,running,status}.json`,
merges in input order, writes `validation.json` (`validation=PASS|FAIL` printed),
and reuses the MMIRAGE aggregator. Smoke-verified on H100 for DataTrove and raw
SGLang; NeMo Curator, Distilabel, and Ray Data LLM follow the identical path.

## Reproducibility / paper mapping

Keep `experiment_metadata.json`, `workload/metadata.json`, execution configs,
per-repetition logs, and the final CSVs/JSON/LaTeX. Paper table fragment:
`results/latex_table.txt`; figures: `results/aggregate_throughput_vs_gpu.png`,
`results/parallel_efficiency_vs_gpu.png`. See also the raw reference numbers in
`experiments/raw_sglang_overhead/README.md`.
