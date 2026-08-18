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

All three frameworks use `Qwen/Qwen3-4B`, temperature `0`, and a maximum of `128` new tokens. MMIRAGE uses batch size `64`; the native runners use concurrency `64`. The experiment uses GPUs `0`, `1`, `2`, and `3`, with three measured repetitions per framework.

The H100 publication driver prepares the workload and launches this experiment through `../publication/orchestrate.py`.

## Outputs

Results are written under `results/`. Rows/s and end-to-end wall time are the reported performance metrics.
