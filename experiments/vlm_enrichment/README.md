# VLM enrichment

This task-generalization experiment reformats MedTrinity image descriptions into structured markdown using MMIRAGE, SGLang, DataTrove and NeMo Curator on four H100 GPUs.

Every framework runs three repetitions with `Qwen/Qwen3-VL-4B-Instruct`, temperature 0.1, top-p 0.9 and maximum 1,024 new tokens. MMIRAGE uses batch size 64; native concurrency is 64. `scripts/prepare_workload.py` deterministically selects rows and stores the images used by all frameworks.

`run_mmirage_vlm.py` and `run_native_vlm_competitor.py` preserve the same output contract (`id`, `formatted_description`) and validate task completion. The current NeMo integration sends model queries row-sequentially internally; its throughput is therefore an integration result, not an optimized NeMo VLM ceiling. Framework-specific multimodal serialization also prevents interpreting token rate as a controlled serving-engine comparison.
