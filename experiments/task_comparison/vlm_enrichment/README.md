# vlm_enrichment (MedTrinity-25M demo → markdown caption)

Same MedTrinity demo subset, prompt, model (`Qwen/Qwen3-VL-4B-Instruct`), and 4-GPU point for MMIRAGE, SGLang, DataTrove, and NeMo Curator. MMIRAGE runs through `mmirage.shard_process`; the natives run the shared VLM runners in `experiments/_shared/vlm_runners.py`.

Workload size lives in `configs/workload_size.yaml` (`num_rows`, written by the calibrator). `UCSC-VLAA/MedTrinity-25M` is a gated dataset (auto-approved): request access once with your account on its Hub page, then set `HF_TOKEN` so the demo config can download (~8 GB).

## 1. Prepare workload

```
python experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py \
  --output-dir experiments/task_comparison/vlm_enrichment/workload \
  --num-rows 400
```

Writes `workload/rows.jsonl`, `workload/images/*.png`, `workload/metadata.json`.

## 2. MMIRAGE (4 GPUs)

```
python experiments/task_comparison/vlm_enrichment/scripts/run_mmirage_vlm.py \
  --execution-config experiments/task_comparison/vlm_enrichment/configs/execution_4gpu.yaml
```

## 3. Native competitors

```
python experiments/task_comparison/vlm_enrichment/scripts/run_native_vlm_competitor.py \
  --framework sglang \
  --workload-jsonl experiments/task_comparison/vlm_enrichment/workload/rows.jsonl \
  --image-base-path experiments/task_comparison/vlm_enrichment/workload \
  --output-root experiments/task_comparison/vlm_enrichment/results/native_competitors/sglang \
  --gpu-count 4 --visible-gpus 0,1,2,3 --repetitions 3

# repeat with --framework datatrove and --framework nemo_curator
```

## 4. Artifacts

- `results/summary.json`, `results/summary.csv`, `results/raw_results.csv`, `results/latex_table.txt`
- per-repetition `validation.json` ({id, formatted_description} contract) under each framework's `runs/gpu_4/rep_*`
