# Experiment outputs

Generated workloads and results are not tracked in Git. The files below are the useful outputs to retain with reported measurements.

## Scaling

`experiments/scaling/workload/` contains workload metadata, `model_revisions.json`, and `publication_manifest.json`. Results are written below `experiments/scaling/results/h100/` and `experiments/scaling/results/a100/`, separated by framework. Preserve raw repetition summaries and aggregate CSV/JSON summaries.

## Recovery

The live controller root is `MMIRAGE_RECOVERY_ROOT` (default `/workspace/mmirage-recovery`). The H100 driver copies extracted summaries and JSON evidence into `experiments/recovery/results/`. Preserve `recovery_results.json`, `recovery_results.csv`, per-run summaries/validation files, and shard-status evidence.

## Text shortening and VLM enrichment

Preserve workload metadata/checksums, raw repetition summaries, and native validation JSON under `experiments/text_shortening/` and `experiments/vlm_enrichment/`.

## Endpoint overhead

Results are written to `experiments/sglang_overhead/results/h100/` and `experiments/sglang_overhead/results/a100/`. `raw_results.csv` is updated after each completed repetition/path. Completed runs also produce `summary.json`, `summary.csv`, and the generated table.

For reproducibility, retain workload hashes, dataset/model revisions, hardware metadata, raw repetition records, validation outputs, aggregate summaries, and recovery evidence alongside the reported results.
