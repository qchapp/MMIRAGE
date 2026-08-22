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

The H100 sweep assigns the same physical devices to every framework so framework comparisons at a given GPU count use identical device placement:

- 1-GPU point: only physical GPU `0` is visible to the runner.
- 2-GPU point: only physical GPUs `0` and `1` are visible.
- 4-GPU point: physical GPUs `0`, `1`, `2`, and `3` are visible.

The A100 transfer experiment contains only the 4-GPU point and uses physical GPUs `0`, `1`, `2`, and `3`. It reuses the byte-identical scaling workload and exact model revision prepared for the H100 run.

## Run this experiment only

Create the competitor environments as described in [`../publication/ENVIRONMENTS.md`](../publication/ENVIRONMENTS.md), then point the framework variables to their Python executables:

```bash
export HF_TOKEN=...
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
```

Prepare the deterministic workload and cache the exact current model revision before timing:

```bash
python experiments/scaling/scripts/prepare_workload.py \
  --output-dir experiments/scaling/workload

python experiments/publication/prefetch_models.py \
  --models Qwen/Qwen3-4B \
  --output-json experiments/scaling/workload/model_revisions.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run only the H100 scaling sweep:

```bash
python experiments/publication/orchestrate.py \
  --stage scaling \
  --hardware h100 \
  --repetitions 3 \
  --overwrite
```

To inspect the exact commands first, add `--dry-run`.

For the A100 transfer point, copy or mount the **unchanged** H100 `experiments/scaling/workload/` directory on the A100 node. Verify and cache the same model revision, then run only the A100 point:

```bash
python experiments/publication/prefetch_models.py \
  --models Qwen/Qwen3-4B \
  --expected-json experiments/scaling/workload/model_revisions.json \
  --output-json /tmp/a100_model_revisions.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python experiments/publication/orchestrate.py \
  --stage scaling \
  --hardware a100 \
  --repetitions 3 \
  --overwrite
```

## Outputs

Results are written below `results/h100/` and `results/a100/`, separated by framework. Rows/s and end-to-end wall time are the primary complete-runner metrics. Direct SGLang in this experiment is a full runner/path baseline; the controlled abstraction-overhead measurement is in [`../sglang_overhead/`](../sglang_overhead/).
