# Endpoint-matched SGLang overhead

This experiment measures MMIRAGE abstraction overhead relative to a raw SGLang client while holding the serving endpoint and serialized prompts fixed.

## Workload

Source dataset: [`simplescaling/s1K-1.1`](https://huggingface.co/datasets/simplescaling/s1K-1.1), split `train`, beginning at source row `0`.

The publication driver prepares **1,000 measured prompts** and **16 warm-up prompts**. For each dataset question, `scripts/prepare_workload.py` first constructs:

```text
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

That string is used as the content of a user chat message and serialized with the locked `Qwen/Qwen3-4B` tokenizer using `apply_chat_template(..., add_generation_prompt=True)`. The resulting `prompt` string is stored in the workload and sent unchanged by both measured paths.

## Execution

Both paths use the same `Qwen/Qwen3-4B` SGLang server, one GPU (`GPU 0`), concurrency `64`, temperature `0`, and a maximum output budget of `1,024` tokens. Raw SGLang and MMIRAGE-over-SGLang each run three measured repetitions.

The experiment is run once on H100 and once on A100. The A100 run reuses the exact H100-prepared prompts, warm-up prompts, and locked model revision.

`raw_sglang_client.py` sends the prepared `prompt` field directly to the completion endpoint. `run_mmirage_with_sglang_endpoint.py` exercises the MMIRAGE endpoint wrapper using that same serialized prompt. `run.py` starts the matched server/client repetitions and appends each completed path/repetition to `raw_results.csv` before writing aggregate summaries.

## Outputs

Results are written under `results/h100/` and `results/a100/`. This is the experiment used to quantify endpoint-matched MMIRAGE overhead; the Direct SGLang condition in [`../scaling/`](../scaling/) is a separate complete-runner comparison.
