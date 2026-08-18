# Scaling

This experiment measures complete-path throughput for deterministic UltraChat rewriting with MMIRAGE, Direct SGLang, DataTrove, and NeMo Curator.

## Workload

Source dataset: [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), split `train_sft`.

`scripts/prepare_workload.py` shuffles the dataset with seed `20260813`, extracts the first user request from each conversation, normalizes trailing whitespace, removes duplicate prompt hashes, and keeps the first **5,430** unique prompts. The resolved Hugging Face dataset revision and model revision are stored in `workload/metadata.json`.

The transformation is:

```text
You are helping construct a public text dataset.

Rewrite the following user request into a helpful assistant response of 4 to 6 sentences.
Preserve the user's intent.

User request:
{prompt_text}
```

## Execution

Every framework uses `Qwen/Qwen3-4B`, temperature `0`, and a maximum of `256` new tokens. MMIRAGE uses batch size `64`; native runners use concurrency `64`. Every reported cell has three repetitions.

The H100 scaling sweep uses the **same physical GPU IDs for every framework**:

- 1-GPU point: GPU `0` only.
- 2-GPU point: GPUs `0` and `1`.
- 4-GPU point: GPUs `0`, `1`, `2`, and `3`.

The A100 transfer experiment contains only the 4-GPU point and likewise uses GPUs `0`, `1`, `2`, and `3`. It reuses the byte-identical scaling workload and exact model revision prepared on H100.

The MMIRAGE single-node runner and framework-specific native runners are under `scripts/`. Publication orchestration is handled by `../publication/orchestrate.py` and the hardware drivers in `../publication/`.

## Outputs

Results are written below `results/h100/` and `results/a100/`, separated by framework. Rows/s and end-to-end wall time are the primary complete-runner metrics. Direct SGLang in this experiment is a full runner/path baseline; the controlled abstraction-overhead measurement is in [`../sglang_overhead/`](../sglang_overhead/).
