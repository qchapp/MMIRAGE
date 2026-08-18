# Scaling

This experiment measures complete-path throughput for deterministic UltraChat rewriting with MMIRAGE, Direct SGLang, DataTrove and NeMo Curator.

The H100 suite evaluates 1, 2 and 4 GPUs with fixed physical subsets (0; 0–1; 0–3). The A100 suite evaluates only the four-GPU transfer point using the exact workload and model revision prepared on H100. Every cell has three repetitions.

Shared settings: `Qwen/Qwen3-4B`, temperature 0, maximum 256 new tokens, MMIRAGE batch size 64, native concurrency 64. Workload generation is in `scripts/prepare_workload.py`; the MMIRAGE single-node runner and native competitor runners live under `scripts/`. The canonical publication orchestration is `../publication/orchestrate.py` and should be launched through the publication shell drivers rather than by manually mixing cells.

Results are written below `results/h100/` and `results/a100/`. Rows/s and end-to-end wall time are the primary complete-runner metrics. Direct SGLang here is not the abstraction-overhead microbenchmark; see `../sglang_overhead/` for the endpoint-matched comparison.
