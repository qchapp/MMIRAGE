# text_shortening (cnn_dailymail → summary)

Same prompt, model (`Qwen/Qwen3-4B`), 4-GPU point, and output contract for MMIRAGE, DataTrove, and NeMo Curator. Rows use the shared native contract (`stable_id`, `source_index`, `prompt_sha256`, `prompt_text`), so the scaling runners and `native_shard_worker` consume them unchanged.

Workload size lives in `configs/workload_size.yaml` (`num_rows`, written by the calibrator).

## 1. Prepare workload

```
python experiments/task_comparison/text_shortening/scripts/prepare_workload.py \
  --output-dir experiments/task_comparison/text_shortening/workload \
  --num-rows 40000
```

Writes `workload/workload.jsonl`, `workload/warmup.jsonl`, `workload/metadata.json`.

## 2. MMIRAGE (4 GPUs)

```
python experiments/single_node_h100_scaling/scripts/run.py \
  --execution-config experiments/task_comparison/text_shortening/configs/execution_4gpu.yaml
```

## 3. Native competitors

```
python experiments/single_node_h100_scaling/scripts/run_datatrove_scaling.py \
  --workload-jsonl experiments/task_comparison/text_shortening/workload/workload.jsonl \
  --output-root experiments/task_comparison/text_shortening/results/native_competitors/datatrove \
  --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3 --model Qwen/Qwen3-4B

python experiments/single_node_h100_scaling/scripts/run_nemo_curator_scaling.py \
  --workload-jsonl experiments/task_comparison/text_shortening/workload/workload.jsonl \
  --output-root experiments/task_comparison/text_shortening/results/native_competitors/nemo_curator \
  --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3 --model Qwen/Qwen3-4B
```

## 4. Artifacts

- `results/summary.json`, `results/summary.csv`, `results/raw_results.csv`, `results/latex_table.txt`
- per-repetition `validation.json` (output contract) under each framework's `runs/gpu_4/rep_*`
