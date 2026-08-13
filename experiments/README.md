# Experiments

This directory contains paper-support experiments. Each experiment is self-contained: documentation, fixed configs, workload preparation, execution scripts, and analysis scripts live under `experiments/<name>/`.

Experiment commands intentionally do not live under the repository-level `scripts/` directory. Run commands from the repository root using the paths documented by each experiment.

| Experiment | Purpose | Entry point |
|---|---|---|
| `raw_sglang_overhead` | Measures AnonLib throughput retention against a matched raw SGLang HTTP baseline. | `experiments/raw_sglang_overhead/scripts/run.py` |
| `shard_recovery` | Measures shard-scoped recovery when selected shard workloads are terminated and only incomplete shards are retried. | `experiments/shard_recovery/scripts/run_k8s.py` |
| `single_node_h100_scaling` | Measures single-node multi-GPU strong scaling with independent one-GPU shard workers. | `experiments/single_node_h100_scaling/scripts/run.py` |

Generated outputs should stay local unless an experiment README explicitly says a small workload or fixture is intentionally committed. Common generated paths include `results/`, `runs/`, `recovery_root/`, plots, archives, and paper-evidence bundles.

Shared helper code for experiment scripts lives in `experiments/_shared/`. These helpers are benchmark scaffolding, not AnonLib runtime code.
