# Single-Node H100 Strong-Scaling Experiment

This experiment measures AnonLib dataset-processing throughput as independent GPU resources increase within one Kubernetes pod on one node. It is a single-node multi-GPU experiment, not a multi-node or distributed-cluster scaling experiment.

The experiment complements the current paper by filling the stated gap around controlled scaling efficiency. The current paper already reports a 16-node MedTrinity observation, shard recovery, and matched single-GPU raw-SGLang overhead. This experiment instead holds the total workload fixed and varies only the number of independent single-GPU AnonLib shard workers on one H100 node.

PDF-derived caveat: the current paper states the MedTrinity distributed run used `16 nodes x 1 GPU`, while the semester report appendix states `16 nodes with 4 GH200 GPUs per node`. This experiment does not rely on either allocation; report the discrepancy if discussing historical MedTrinity evidence.

## Design

The semantic transformation is fixed in `semantic_recipe.yaml` for all GPU counts. The wrappers only change execution/resource settings: `gpu_count`, visible GPU IDs, and therefore the number of deterministic AnonLib logical shards.

Each shard process is pinned to one GPU with `CUDA_VISIBLE_DEVICES=<gpu_id>` and uses `tp_size: 1`. This intentionally avoids tensor-parallelizing Qwen3-4B, because the model fits on one H100 and tensor parallelism would answer a different question.

AnonLib uses Hugging Face Datasets sharding through `Dataset.shard(num_shards=<gpu_count>, index=<shard_id>)`. Across shard IDs, each GPU-count point collectively covers the same complete prepared JSONL workload.

## Prepare Dataset

Default target size is 30,000 unique UltraChat prompts. Increase to 50,000 if the run is too short on your pod.

```bash
python3 scripts/prepare_dataset.py \
  --output-dir experiments/single_node_h100_scaling/workload \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --num-rows 30000 \
  --model-path Qwen/Qwen3-4B
```

The script writes `workload/workload.jsonl`, `workload/metadata.json`, and SHA-256 checksums. Rows contain `stable_id`, `source_index`, `prompt_text`, and prompt hashes. Duplicate prompts are skipped rather than duplicated.

## Run In A 4-H100 Pod Terminal

Run each point three times:

```bash
bash scripts/run_1gpu.sh
bash scripts/run_2gpu.sh
bash scripts/run_4gpu.sh
```

To inspect commands without launching SGLang/AnonLib work:

```bash
bash scripts/run_4gpu.sh --dry-run
```

All outputs are written under `experiments/single_node_h100_scaling/` by default:

```text
raw_results.csv
summary.csv
summary.json
latex_table.txt
aggregate_throughput_vs_gpu.png
parallel_efficiency_vs_gpu.png
runs/gpu_<N>/rep_<R>/...
```

## Metrics

Per repetition, `raw_results.csv` records:

```text
gpu_count, repetition, processed_rows, total_input_tokens, total_output_tokens,
end_to_end_wall_seconds, model_loading_seconds, aggregate_output_tok_s,
output_tok_s_per_gpu, rows_s, mean_gpu_utilization
```

`aggregate_output_tok_s`, `output_tok_s_per_gpu`, and `rows_s` are end-to-end metrics using the driver wall time from launching all shard processes until all have exited. `steady_state_output_tok_s` and `steady_state_rows_s` are also recorded for comparison and exclude the maximum observed shard model-loading time.

`summary.csv` and `summary.json` report mean and sample standard deviation over the three repetitions. Speedup is computed from mean end-to-end aggregate output-token throughput relative to the 1-GPU mean. Parallel efficiency is `speedup / gpu_count`.

## Plot

After runs finish, plots are generated automatically. You can regenerate them with:

```bash
python3 scripts/scaling_plot.py \
  --summary-csv experiments/single_node_h100_scaling/summary.csv \
  --output-dir experiments/single_node_h100_scaling
```
