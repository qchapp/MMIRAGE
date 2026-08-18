# Publication protocol

This document defines the comparison contract used by the publication drivers. It is descriptive; the executable settings remain in the experiment configs and orchestration scripts.

## Shared rules

- Three measured repetitions per reported cell.
- Timed cells are serialized on a node to avoid GPU contention between frameworks.
- Workload preparation and model download occur before timed regions.
- H100 and A100 scaling use the same deterministic UltraChat rows and the same exact `Qwen/Qwen3-4B` revision.
- Frameworks receive the same semantic instruction, generation budget, and output contract within each comparison. Framework-native prompt/chat serialization may differ.
- Cross-framework input-token counters are not treated as directly comparable metrics.

## Scaling

Frameworks: MMIRAGE, Direct SGLang, DataTrove, NeMo Curator.

H100 points: 1, 2 and 4 GPUs. The physical subsets are fixed to GPU 0; GPUs 0–1; and GPUs 0–3 respectively for every framework. The A100 transfer experiment uses four GPUs only.

Model: `Qwen/Qwen3-4B`. Temperature: 0. Maximum new tokens: 256. MMIRAGE batch size: 64. Native concurrency: 64. Task: rewrite the UltraChat user request into a helpful four-to-six-sentence assistant response while preserving intent.

The Direct SGLang condition is a complete runner/path comparison. It is not the measurement used to quantify MMIRAGE wrapper overhead.

## Recovery

Hardware: four H100 GPUs. Workload: 4,000 records partitioned into 16 deterministic logical shards, with at most four active shards. Model: `Qwen/Qwen3-4B`. Temperature: 0. Maximum new tokens: 256. MMIRAGE batch size: 64. Native concurrency: 64.

MMIRAGE conditions: baseline, one failed shard, four failed shards. Native frameworks: Direct SGLang, DataTrove, NeMo Curator, Distilabel and Ray Data LLM, each with one-failure and four-failure conditions. Failure injection occurs 30 seconds after the designated worker reaches the running phase. Native execution uses the publication cleanup wrapper to terminate orphaned vLLM engine cores before later waves.

Cross-framework interpretation should prioritize completed-shard reuse, recomputed work, and final output validity. Absolute recovery wall time is not a framework-native recovery comparison because a common external relaunch controller is used and the fixed failure time does not correspond to identical useful-work progress across frameworks.

## Text shortening

Frameworks: MMIRAGE, DataTrove, NeMo Curator. Hardware: four H100 GPUs. Dataset: CNN/DailyMail. Model: `Qwen/Qwen3-4B`. Temperature: 0. Maximum new tokens: 128. MMIRAGE batch size and native concurrency: 64. Task: faithful two-to-three-sentence summarization.

Primary comparison metrics are rows/s and end-to-end wall time.

## VLM enrichment

Frameworks: MMIRAGE, SGLang, DataTrove, NeMo Curator. Hardware: four H100 GPUs. Dataset: MedTrinity demo subset. Model: `Qwen/Qwen3-VL-4B-Instruct`. Temperature: 0.1, top-p: 0.9, maximum new tokens: 1024, MMIRAGE batch size/native concurrency: 64.

The current NeMo VLM integration issues model requests row-sequentially internally, so it is an integration/task-completion result rather than a tuned NeMo throughput ceiling. MMIRAGE and native VLM paths also use different multimodal serialization stacks; token-rate results should therefore not be interpreted as an exact serving-engine comparison.

## Endpoint-matched SGLang overhead

This is the controlled abstraction-overhead experiment. Raw SGLang and MMIRAGE send the same prepared serialized prompts to the same SGLang completion endpoint. It uses one GPU, three repetitions, concurrency 64, temperature 0 and a 1,024-token output budget. The experiment runs on both H100 and A100 using the exact H100-prepared workload and locked model revision.
