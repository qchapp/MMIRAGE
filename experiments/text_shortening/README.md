# Text shortening

This experiment measures task generalization by summarizing news articles with MMIRAGE, DataTrove, and NeMo Curator on four H100 GPUs.

## Workload

Source dataset: [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail), configuration `3.0.0`, split `train`.

`scripts/prepare_workload.py` shuffles with seed `20260813`, truncates each article to at most **4,096 source characters**, normalizes whitespace, removes duplicate article hashes, and keeps the first **9,471** unique articles. The resolved dataset and model revisions are recorded in `workload/metadata.json`.

The transformation prompt is applied exactly once by every framework:

```text
Summarize the following news article in 2 to 3 sentences.
Keep the summary faithful to the facts in the article.

Article:
{prompt_text}
```

## Execution

All three frameworks use `Qwen/Qwen3-4B`, temperature `0`, and a maximum of `128` new tokens. MMIRAGE uses batch size `64`; the native runners use concurrency `64`. The experiment uses physical GPUs `0`, `1`, `2`, and `3`, with three measured repetitions per framework.

## Run this experiment only

Create the competitor environments described in [`../publication/ENVIRONMENTS.md`](../publication/ENVIRONMENTS.md), then set the Python interpreter variables:

```bash
export HF_TOKEN=...
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
```

Prepare the deterministic workload and cache the model revision before timing:

```bash
python experiments/text_shortening/scripts/prepare_workload.py \
  --output-dir experiments/text_shortening/workload

python experiments/publication/prefetch_models.py \
  --models Qwen/Qwen3-4B \
  --output-json experiments/text_shortening/workload/model_revisions.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run only the text-shortening experiment:

```bash
python experiments/publication/orchestrate.py \
  --stage text \
  --repetitions 3 \
  --overwrite
```

Add `--dry-run` to inspect the exact framework commands without running inference.

## Outputs

Results are written under `results/`. Rows/s and end-to-end wall time are the reported performance metrics.
