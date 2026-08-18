# Recovery

This experiment evaluates deterministic shard reuse and recomputation after injected worker failures on four H100 GPUs.

The shared UltraChat workload contains 4,000 records split into 16 logical shards with at most four active shards. MMIRAGE runs baseline, one-failure and four-failure conditions. Direct SGLang, DataTrove, NeMo Curator, Distilabel and Ray Data LLM run one-failure and four-failure conditions. Every framework/condition cell has three repetitions. Failure injection occurs 30 seconds after designated workers enter the running phase.

Shared generation settings are `Qwen/Qwen3-4B`, temperature 0 and maximum 256 new tokens. MMIRAGE uses batch size 64 and native competitors use concurrency 64. `run_local.py` is the retained MMIRAGE recovery controller; `run_native_recovery_publication.py` wraps the native controller with the orphan-vLLM cleanup required by the publication runs. `extract_results.py` produces the aggregate recovery evidence.

The publication driver persists summaries and JSON validation/evidence under `results/`. Interpret cross-framework recovery primarily through completed-shard reuse, recomputed work and final output validity. The common external relaunch controller and fixed failure time mean absolute wall time is not a framework-native recovery latency benchmark.
