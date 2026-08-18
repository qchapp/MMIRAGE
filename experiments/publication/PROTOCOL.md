# Publication protocol

This document summarizes the settings executed by `run_h100.sh`, `run_a100.sh`, and `orchestrate.py`. The experiment-specific READMEs give the corresponding workload construction details.

## Shared execution rules

- Three measured repetitions per reported cell.
- Workloads are prepared deterministically before timed execution.
- Hugging Face dataset and model revisions are resolved to commit SHAs and recorded in workload metadata; model snapshots are prefetched before timed execution.
- MMIRAGE uses batch size 64 in the scaling, recovery, text-shortening, and VLM experiments. Native text/VLM paths use concurrency 64.
- H100 scaling uses physical GPU IDs `0`, `0,1`, and `0,1,2,3` for the 1-, 2-, and 4-GPU points respectively, for every framework. The A100 scaling point uses `0,1,2,3`.

## Scaling

**Dataset.** [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), config/default dataset view, split `train_sft`. The workload contains 5,430 rows. Rows are selected by shuffling with seed `20260813`, extracting the first user message, normalizing trailing whitespace, removing duplicate prompt hashes, and taking the first 5,430 unique prompts.

**Frameworks.** MMIRAGE, Direct SGLang, DataTrove, and NeMo Curator.

**Hardware.** H100: 1 GPU = physical GPU `0`; 2 GPUs = physical GPUs `0,1`; 4 GPUs = physical GPUs `0,1,2,3`. A100: one four-GPU point on physical GPUs `0,1,2,3` using the exact H100-prepared workload and model revision.

**Model and decoding.** `Qwen/Qwen3-4B`, temperature `0`, maximum `256` new tokens.

**Transformation prompt.** The same semantic instruction is used by MMIRAGE and the native text runners:

```text
You are helping construct a public text dataset.

Rewrite the following user request into a helpful assistant response of 4 to 6 sentences.
Preserve the user's intent.

User request:
{prompt_text}
```

The Direct SGLang scaling condition is a complete execution-path baseline. The endpoint-matched experiment below is the controlled measurement of MMIRAGE wrapper overhead.

## Recovery

**Dataset.** The first 4,000 rows of the same deterministic `HuggingFaceH4/ultrachat_200k` selection used by scaling. They are partitioned into 16 deterministic logical shards of 250 rows each.

**Hardware.** Four H100 GPUs (`0,1,2,3`), with at most four shard workers active simultaneously.

**Model, decoding, and transformation.** Identical to scaling: `Qwen/Qwen3-4B`, temperature `0`, maximum `256` new tokens, and the UltraChat rewrite prompt shown above.

**Failure conditions.** MMIRAGE runs a clean baseline plus two injected-failure conditions. `fail_1` terminates logical shard `3`; `fail_4` terminates logical shards `1`, `5`, `9`, and `13`. Native competitors (Direct SGLang, DataTrove, NeMo Curator, Distilabel, and Ray Data LLM) run `fail_1` and `fail_4`.

For each targeted shard, the controller waits until that worker has entered its running phase, waits another 30 seconds, and sends `SIGTERM` to the shard worker/wrapper. The initial phase is allowed to finish for the other shards. Retry then relaunches only shards without a valid completion marker. MMIRAGE snapshots completed-shard outputs before and after retry; native validation checks that completed outputs are unchanged, only incomplete/killed shards are relaunched, and the merged output has no missing, duplicate, or unexpected IDs.

## Text shortening

**Dataset.** [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail), configuration `3.0.0`, split `train`. The workload contains 9,471 rows. Rows are shuffled with seed `20260813`; each article is truncated to at most 4,096 source characters, normalized, deduplicated by content hash, and the first 9,471 unique articles are retained.

**Frameworks/hardware.** MMIRAGE, DataTrove, and NeMo Curator on four H100 GPUs.

**Model and decoding.** `Qwen/Qwen3-4B`, temperature `0`, maximum `128` new tokens.

**Transformation prompt.** Applied exactly once per framework:

```text
Summarize the following news article in 2 to 3 sentences.
Keep the summary faithful to the facts in the article.

Article:
{prompt_text}
```

## VLM enrichment

**Dataset.** [`UCSC-VLAA/MedTrinity-25M`](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M), configuration `25M_demo`, split `train`. The workload contains 83 rows. The dataset is shuffled with seed `20260813`; the first 83 rows with unique non-empty sample IDs are retained and their images are saved as PNG files for reuse by all frameworks.

**Frameworks/hardware.** MMIRAGE, SGLang, DataTrove, and NeMo Curator on four H100 GPUs.

**Model and decoding.** `Qwen/Qwen3-VL-4B-Instruct`, temperature `0.1`, top-p `0.9`, maximum `1,024` new tokens.

**Transformation prompt.** The image and its original MedTrinity caption are supplied to the VLM with the following text instruction:

```text
Reformat the image description with markdown without adding anything else.
Add titles and structure your output.

Image description:
{caption}
```

MMIRAGE serializes the generated answer through its configured `conversations`/`modalities` output schema; the native VLM harness records the generated string as `formatted_description`. The transformation instruction and source image/caption are matched even though the result serialization is framework-specific.

## Endpoint-matched SGLang overhead

**Dataset.** [`simplescaling/s1K-1.1`](https://huggingface.co/datasets/simplescaling/s1K-1.1), split `train`, starting from row 0. The publication driver prepares 1,000 measured prompts and 16 additional warm-up prompts.

**Hardware.** One H100 and one A100, GPU `0` on each machine. The A100 experiment reuses the exact H100-prepared workload and locked model revision.

**Model and decoding.** `Qwen/Qwen3-4B`, concurrency `64`, temperature `0`, maximum `1,024` output tokens, three repetitions for each of raw SGLang and MMIRAGE-over-SGLang.

**Prompt construction.** For each dataset question, workload preparation first constructs:

```text
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

That string is placed in a user chat message and serialized once with the locked Qwen tokenizer using `apply_chat_template(..., add_generation_prompt=True)`. The resulting serialized `prompt` field is then sent unchanged by both the raw SGLang client and the MMIRAGE endpoint path.
