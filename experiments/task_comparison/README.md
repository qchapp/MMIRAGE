# task_comparison

Same transformation workload, same model, same 4-GPU point, compared across MMIRAGE and framework-native pipelines. Subsets are committed-sized via `configs/workload_size.yaml` (written by `experiments/smoke/calibrate.py`).

| Task | Workload | Model | Frameworks | Runner |
| --- | --- | --- | --- | --- |
| `text_shortening` | cnn_dailymail (article → 2-3 sentence summary) | `Qwen/Qwen3-4B` | MMIRAGE, DataTrove, NeMo Curator | reuses `single_node_h100_scaling` runners |
| `vlm_enrichment` | MedTrinity-25M demo (image + caption → markdown caption) | `Qwen/Qwen3-VL-4B-Instruct` | MMIRAGE, SGLang, DataTrove, NeMo Curator | `vlm_enrichment/scripts/` |

Prerequisite on the pod: the environment and model/dataset downloads in `experiments/smoke/AGENT_PROMPT.md`.
