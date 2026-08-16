# Experiments

Runbooks for the four main MMIRAGE benchmark experiments. All commands run from
the repository root inside the MMIRAGE GPU environment. Generated artifacts
(`workload/`, `results/`, `runs/`) are git-ignored; commit only code, configs,
and READMEs.

| Experiment | Measures | Primary metrics | Runbook |
|---|---|---|---|
| `a_matrix` | Consolidated A + B: single-node strong scaling (1/2/4 H100 and 4x A100), shard recovery, text and vlm task comparisons | `aggregate_output_tok_s`, `output_tok_s_per_gpu`, `rows_s`, `speedup_vs_1gpu`, recovery wall time | `experiments/a_matrix/README.md` |
| `task_comparison/text_shortening` | Article→summary transformation with MMIRAGE/DataTrove/NeMo Curator | per-framework wall time and throughput at fixed workload | `experiments/task_comparison/text_shortening/README.md` |
| `task_comparison/vlm_enrichment` | Image-caption enrichment with MMIRAGE/SGLang/DataTrove/NeMo Curator | per-framework wall time and throughput at fixed workload | `experiments/task_comparison/vlm_enrichment/README.md` |

`single_node_h100_scaling`, `shard_recovery` and `raw_sglang_overhead` are
superseded by `a_matrix` (their runners are still used as the execution
engine; the raw_sglang path is one framework of the scaling matrix).

## Running everything unattended

`bash experiments/run_all.sh` runs the whole suite end to end without human
intervention and **without any pod_a / pod_b split**: preflight checks (venv,
4 GPUs, HF token, all competitor interpreters), then smoke → calibrate → every
experiment on one 4-GPU pod, each stage isolated with its own log under
`experiments/run_all_logs/` and a final status table. `--only`/`--skip` select
stages (e.g. `bash experiments/run_all.sh --skip vlm`).

By default the MMIRAGE-only cells already covered by the 2026-08-15 fast-run
reproduction are reused, not rerun (see
`experiments/a_matrix/README.md#reusing-the-2026-08-15-fast-runs`): those
results are preserved, and `run_setup.py --reuse-fastruns` skips the
corresponding units. `--rerun-reused` reruns every cell from scratch.

The only node that runs separately is the 4x A100 point
(`bash experiments/run_a100.sh`); everything else (scaling 1/2/4, recovery,
text, vlm) is driven by run_all.sh on the 4x H100 pod.

## Monitoring

`python experiments/progress_tracker.py` renders a live dashboard for the
`run_setup.py` scheduler: per-unit status derived from the per-cell logs,
elapsed vs. expected time, GPU usage, and recovery progress. `--once` prints a
single snapshot, `--json` emits machine-readable output, `--setup <name>` /
`--pod pod_a|pod_b` filter the view.

## Sizes and the smoke calibrator

Each experiment's default workload size lives in its `configs/workload_size.yaml`
and is committed. `experiments/smoke/` contains the one-time calibrator that
measures per-cell timings at a small size and rewrites those size files so every
experiment fits its wall-clock budget. See `experiments/smoke/README.md`.

## Running on the EPFL pod

On the `meditron-fab-h100` pod, follow the literal agent runbook in
`experiments/smoke/AGENT_PROMPT.md`: it pins the environment (uv, HF token,
`MMIRAGE_RECOVERY_ROOT`, sglang/setuptools workaround), the exact smoke →
calibrate → re-prepare → run sequence per experiment, and the result/report
conventions.

## Repository policy

- Fixed recipes and execution configs live under `experiments/<name>/configs/`.
- Runnable entry points live under `experiments/<name>/scripts/`.
- Shared benchmark scaffolding lives in `experiments/_shared/` (helper code, not
  MMIRAGE runtime code). Runner entry points never import sibling experiment scripts.
- Generated outputs stay local and ignored; only code, configs, and READMEs are committed.
