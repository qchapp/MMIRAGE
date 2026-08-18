# Publication artifacts

Generated experiment outputs are not source code and should remain untracked. Internal or post-review archival should preserve the provenance and result artifacts below together with the exact source commit. A double-blind reviewer bundle must follow the anonymization exception at the end of this document instead.

## Scaling

`experiments/scaling/workload/` contains workload metadata, `model_revisions.json`, and `publication_manifest.json`. Results are written below `experiments/scaling/results/h100/` and `experiments/scaling/results/a100/`, separated by framework. Preserve raw repetition summaries plus aggregated CSV/JSON summaries.

## Recovery

The live controller root is `MMIRAGE_RECOVERY_ROOT` (default `/workspace/mmirage-recovery`). The publication H100 driver copies extracted recovery summaries and JSON evidence into `experiments/recovery/results/`. Preserve `recovery_results.json`, `recovery_results.csv`, per-run summaries/validation files, and shard-status evidence.

## Text and VLM

Preserve workload metadata/checksums and raw repetition summaries under `experiments/text_shortening/results/` and `experiments/vlm_enrichment/results/`. Native validation JSON is part of the evidence, not disposable logging.

## Endpoint overhead

Results are written to `experiments/sglang_overhead/results/h100/` and `experiments/sglang_overhead/results/a100/`. `raw_results.csv` is appended after each completed repetition/path; final `summary.json`, `summary.csv`, and the generated table are produced after the run completes.

## Minimum provenance bundle

For internal/post-review archival, preserve: exact source commit; workload metadata and hashes; dataset/model revisions; hardware metadata; raw repetition records; validation outputs; aggregate summaries; and recovery evidence. Large model caches and generated intermediate shard data are not part of the archival bundle.

## Double-blind submission exception

Do **not** expose a public repository commit SHA in the anonymous reviewer bundle. Run the refactor verifier before export, remove `.git/`, `experiments/publication/_verification/`, and `experiments/publication/verify_refactor.py`, and normally omit generated workloads/results entirely. If generated result/provenance JSON must be supplied to reviewers, redact or replace repository-identifying fields such as `git_commit` with an anonymous source-bundle identifier while leaving workload hashes, model/dataset revisions, hardware metadata, measurements, and validation evidence intact.
