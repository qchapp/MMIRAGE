# Text shortening

This task-generalization experiment summarizes CNN/DailyMail articles with MMIRAGE, DataTrove and NeMo Curator on four H100 GPUs.

Every framework runs three repetitions with `Qwen/Qwen3-4B`, temperature 0 and a 128-token maximum output budget. MMIRAGE uses batch size 64; native competitors use concurrency 64. `scripts/prepare_workload.py` deterministically selects and normalizes the input articles. The MMIRAGE recipe and native prompt template apply the same two-to-three-sentence faithful-summary instruction exactly once.

The canonical publication driver invokes this experiment through `../publication/orchestrate.py`. Primary metrics are rows/s and end-to-end wall time; framework-native input-token counters should not be compared directly.
