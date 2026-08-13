# Experiments

Experiment-specific commands, configs, and runbooks live under `experiments/<name>/`. Do not use repository-level `scripts/` for experiment entry points.

Run every command from the repository root unless an experiment README says otherwise.

## Available Experiments

| Experiment | Purpose | Start here |
|---|---|---|
| `raw_sglang_overhead` | Measures AnonLib throughput retention against a matched raw SGLang HTTP baseline. | `experiments/raw_sglang_overhead/README.md` |
| `shard_recovery` | Measures shard-scoped recovery when selected shard workloads are terminated and only incomplete shards are retried. | `experiments/shard_recovery/README.md` |
| `single_node_h100_scaling` | Measures single-node multi-GPU strong scaling with independent one-GPU shard workers. | `experiments/single_node_h100_scaling/README.md` |
| `nemo_curator_comparison` | Compares AnonLib with NeMo Curator/Data Designer on a matched LLM-only multimodal ChartQA transformation. | `experiments/nemo_curator_comparison/README.md` |

Each experiment README is the source of truth for commands, output locations, artifact policy, and metadata to archive.

## Repository Policy

- Keep fixed recipes and execution configs under `experiments/<name>/configs/`.
- Keep runnable experiment commands under `experiments/<name>/scripts/`.
- Keep generated outputs local unless an experiment README explicitly says a small workload or fixture is committed.
- Common generated paths include `results/`, `runs/`, `recovery_root/`, plots, archives, and paper-evidence bundles.
- Shared helper code for experiment scripts lives in `experiments/_shared/`. These helpers are benchmark scaffolding, not AnonLib runtime code.
