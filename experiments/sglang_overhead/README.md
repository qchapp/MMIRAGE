# Endpoint-matched SGLang overhead

This experiment measures the overhead of the MMIRAGE abstraction relative to a raw client while holding the SGLang serving endpoint and serialized prompts fixed.

Both paths use the same `Qwen/Qwen3-4B` server, the same prepared tokenized chat-template prompts, one GPU, concurrency 64, temperature 0 and a 1,024-token maximum output budget. Each path has three measured repetitions. The experiment is run once on H100 and once on A100; the A100 run reuses the exact workload and locked model revision produced on H100.

`raw_sglang_client.py` sends each prepared prompt directly to the completion endpoint. `run_mmirage_with_sglang_endpoint.py` exercises the corresponding MMIRAGE wrapper against that endpoint. `run.py` controls the matched server/client repetitions and writes each completed repetition to `raw_results.csv` before producing aggregate outputs.

Use this experiment—not the complete Direct SGLang scaling runner—for claims about MMIRAGE abstraction/orchestration overhead.
