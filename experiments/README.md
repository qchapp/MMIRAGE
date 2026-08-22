# Publication experiments

This directory contains the evaluation used in the paper. All commands below are run from the repository root.

| Experiment | Comparison | Hardware |
|---|---|---|
| [`scaling/`](scaling/) | UltraChat rewrite throughput: MMIRAGE, Direct SGLang, DataTrove, NeMo Curator | 1/2/4× H100; 4× A100 |
| [`recovery/`](recovery/) | shard reuse and recomputation after injected worker failures | 4× H100 |
| [`text_shortening/`](text_shortening/) | CNN/DailyMail summarization: MMIRAGE, DataTrove, NeMo Curator | 4× H100 |
| [`vlm_enrichment/`](vlm_enrichment/) | MedTrinity multimodal enrichment: MMIRAGE, SGLang, DataTrove, NeMo Curator | 4× H100 |
| [`sglang_overhead/`](sglang_overhead/) | endpoint-matched MMIRAGE vs raw SGLang | 1× H100; 1× A100 |

All reported cells use three repetitions. Exact source datasets, workload sizes, selection rules, transformation prompts, models, decoding settings, GPU placement, and failure conditions are documented in [`publication/PROTOCOL.md`](publication/PROTOCOL.md) and in each experiment's README. Each experiment README also gives commands for reproducing that experiment by itself without running the full hardware suite.

## Environments

The main MMIRAGE/SGLang environment and isolated competitor environments are described in [`publication/ENVIRONMENTS.md`](publication/ENVIRONMENTS.md). Exact requirement files are under `publication/environment/`.

The H100 run requires exactly four visible H100 GPUs and `HF_TOKEN` in the environment. The A100 run requires exactly four visible A100 GPUs and reuses the H100-prepared scaling and endpoint-overhead workloads.

## H100 evaluation

Inspect the planned commands without running inference:

```bash
bash experiments/publication/run_h100.sh --dry-run
```

Run the full H100 evaluation:

```bash
bash experiments/publication/run_h100.sh
```

The driver prepares deterministic workloads, resolves exact model revisions before timed execution, and runs scaling, recovery, text shortening, VLM enrichment, and endpoint-matched overhead. Generated results are written under each experiment's `results/` directory.

## A100 transfer point

The A100 node must use the H100-produced workload directories unchanged:

```text
experiments/scaling/workload/
experiments/sglang_overhead/workload/
```

Inspect the A100 commands:

```bash
bash experiments/publication/run_a100.sh --dry-run
```

Run the four-GPU scaling transfer point and one-GPU endpoint-overhead experiment:

```bash
bash experiments/publication/run_a100.sh
```

The A100 driver verifies the H100 workload hashes and model revision before execution.

## Monitoring

A read-only progress dashboard can be attached while either suite is running:

```bash
python experiments/progress_tracker.py --suite h100
python experiments/progress_tracker.py --suite a100
```

## Outputs

Expected result and provenance files are summarized in [`publication/ARTIFACTS.md`](publication/ARTIFACTS.md). Generated workloads, model outputs, and result directories are untracked.
