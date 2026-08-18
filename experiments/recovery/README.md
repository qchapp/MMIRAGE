# Recovery

This experiment evaluates shard reuse and recomputation after deliberately terminating active workers during generation on four H100 GPUs.

## Workload

Source dataset: [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), split `train_sft`.

The recovery workload is the first **4,000** rows of the same deterministic UltraChat selection used by the scaling experiment: shuffle seed `20260813`, first user request extracted from each conversation, normalized prompts, duplicate prompt hashes removed. The 4,000 rows are split into **16 logical shards of 250 rows each**.

The generation task is identical to scaling:

```text
You are helping construct a public text dataset.

Rewrite the following user request into a helpful assistant response of 4 to 6 sentences.
Preserve the user's intent.

User request:
{prompt_text}
```

All paths use `Qwen/Qwen3-4B`, temperature `0`, and at most `256` new tokens. MMIRAGE uses batch size `64`; native competitors use concurrency `64`.

## Injected failures

At most four shard workers are active at once on GPUs `0`, `1`, `2`, and `3`. Every condition has three repetitions.

MMIRAGE runs:

- `baseline`: no worker is terminated.
- `fail_1`: logical shard **3** is terminated.
- `fail_4`: logical shards **1, 5, 9, and 13** are terminated.

Direct SGLang, DataTrove, NeMo Curator, Distilabel, and Ray Data LLM run the `fail_1` and `fail_4` conditions.

For a targeted shard, the controller first waits until the worker has entered its running phase. It then waits **30 seconds** and sends **SIGTERM** to that shard's worker/wrapper. Other workers in the initial phase are allowed to finish normally. The retry phase relaunches only shards that do not have a valid completion marker.

For MMIRAGE, completed shard outputs are snapshotted before and after retry so that reuse can be checked directly. For native competitors, validation checks that completed shard outputs are unchanged, retry is limited to incomplete or killed shards, and the final merged output contains no missing, duplicate, or unexpected IDs.

`run_local.py` controls the MMIRAGE local-shard experiment. `run_native_recovery_publication.py` invokes the native recovery controller and clears orphaned vLLM engine processes between waves when necessary. `extract_results.py` aggregates the recorded recovery evidence.

## Outputs

The H100 publication driver copies recovery summaries and JSON evidence into `results/`. The key recovery evidence is the set of shards reused unchanged, the set recomputed after failure, and final-output validity.
