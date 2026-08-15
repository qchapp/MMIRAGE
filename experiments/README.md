# Experiments

Runbooks for the four main MMIRAGE benchmark experiments. All commands run from
the repository root inside the MMIRAGE GPU environment. Generated artifacts
(`workload/`, `results/`, `runs/`) are git-ignored; commit only code, configs,
and READMEs.

| Experiment | Measures | Primary metrics | Runbook |
|---|---|---|---|
| `raw_sglang_overhead` | MMIRAGE throughput retention vs matched raw SGLang HTTP, 1 GPU | `output_tok_s_per_gpu`, `rows_s`, `throughput_retention`, `relative_orchestration_overhead` | `experiments/raw_sglang_overhead/README.md` |
| `single_node_h100_scaling` | Single-node multi-GPU strong scaling, 1/2/4 one-GPU shard workers | `aggregate_output_tok_s`, `output_tok_s_per_gpu`, `speedup_vs_1gpu`, `parallel_efficiency` | `experiments/single_node_h100_scaling/README.md` |
| `shard_recovery` | Shard-scoped recovery after deliberate worker termination, 16 shards / 4 conditions | `shards_recomputed_count`, `fraction_of_total_workload_recomputed`, recovery wall time | `experiments/shard_recovery/README.md` |
| `task_comparison/text_shortening` | Article→summary transformation with MMIRAGE/DataTrove/NeMo Curator | per-framework wall time and throughput at fixed workload | `experiments/task_comparison/text_shortening/README.md` |
| `task_comparison/vlm_enrichment` | Image-caption enrichment with MMIRAGE/SGLang/DataTrove/NeMo Curator | per-framework wall time and throughput at fixed workload | `experiments/task_comparison/vlm_enrichment/README.md` |

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
