# Experiments

Runbooks for the MMIRAGE benchmark experiments. Commands are run from the
repository root inside the MMIRAGE GPU environment. Generated artifacts
(`workload/`, `results/`, `runs/`) are git-ignored; commit code, configs, and
documentation rather than generated outputs.

## Publication experiment overview

| Experiment | What it measures | Frameworks | GPU points |
|---|---|---|---|
| `a_matrix` (`gpu_scaling`) | strong scaling of UltraChat rewriting | MMIRAGE, Direct SGLang (`raw_sglang`), DataTrove, NeMo Curator | 1 / 2 / 4 H100 |
| `a_matrix` (`a100_4gpu`) | accelerator transfer of the same rewrite workload | same four | 4 A100 |
| `a_matrix` (`recovery`) | shard-scoped fault recovery | MMIRAGE + Direct SGLang, DataTrove, NeMo Curator, Distilabel, Ray Data LLM | 4 H100 |
| `a_matrix` (`text_shortening`) | article-to-summary task generalization | MMIRAGE, DataTrove, NeMo Curator | 4 H100 |
| `a_matrix` (`vlm_enrichment`) | multimodal enrichment task generalization | MMIRAGE, SGLang, DataTrove, NeMo Curator | 4 H100 |

Historical result directories may contain superseded measurements. The corrected
publication benchmark must be rerun with the current batch/concurrency and
failure-injection settings; the 2026-08-15 fast-run cells are provenance only.

## Publication entry points

For the corrected H100 suite:

```bash
# Non-destructive plan/preflight
bash run_all_h100_rerun.sh --dry-run

# Fresh publication run
bash run_all_h100_rerun.sh
```

For the four-GPU A100 transfer point, first copy the exact
`experiments/a_matrix/workload/` directory produced by the H100 run to the A100
node, then:

```bash
bash run_all_a100_rerun.sh --dry-run
bash run_all_a100_rerun.sh
```

Do not regenerate the A-MATRIX workload on the A100 node. The A100 driver
verifies `workload.jsonl` against `metadata.json` `workload_sha256`.

See `experiments/a_matrix/README.md` for the publication protocol and exact
settings.

## The A matrix

The primary UltraChat rewrite task is shared by H100 scaling, the A100 transfer
point, and recovery. Text shortening and VLM enrichment are separate
task-generalization experiments with their own workloads, prompts, and (for
VLM) model.

Within a comparison, frameworks receive the same workload rows, semantic
prompt/instruction, model, generation budget, and output contract.
Framework-native prompt/chat serialization may differ.

### Corrected H100 scaling

- 1 / 2 / 4 H100 GPUs;
- 3 repetitions;
- `Qwen/Qwen3-4B`;
- temperature 0;
- `max_new_tokens=256`;
- MMIRAGE batch 64;
- native concurrency 64;
- scaling units serialized to avoid shared-host contention.

### Corrected recovery

- 16 logical shards, at most 4 active;
- MMIRAGE: baseline / `fail_1` / `fail_4`;
- five native competitors: `fail_1` / `fail_4`;
- 3 repetitions per framework/condition cell;
- MMIRAGE loading batch 64;
- native concurrency 64;
- temperature 0;
- `max_new_tokens=256`;
- failure injection at 30 seconds.

There are 13 framework/condition cells and 39 condition-repetition executions.

### Text shortening

CNN/DailyMail article summarization with MMIRAGE, DataTrove, and NeMo Curator:

- 4 H100 GPUs;
- 3 repetitions;
- MMIRAGE batch 64;
- native concurrency 64;
- temperature 0;
- `max_new_tokens=128`.

### VLM enrichment

MedTrinity enrichment using `Qwen/Qwen3-VL-4B-Instruct`:

- 4 H100 GPUs;
- 3 repetitions;
- MMIRAGE batch 64;
- native concurrency 64;
- temperature 0.1;
- `top_p=0.9`;
- `max_new_tokens=1024`.

The current NeMo VLM integration is row-sequential internally and should be
treated as an integration/contract result rather than a fully tuned throughput
ceiling.

## Direct SGLang terminology

The A-MATRIX `raw_sglang` condition is a **complete Direct SGLang runner**. It
is a runner/path comparison.

`experiments/raw_sglang_overhead` is a separate endpoint-matched experiment and
is the appropriate source for claims about MMIRAGE abstraction/orchestration
overhead relative to the same SGLang serving endpoint.

## Manual `run_setup.py` usage

```bash
# H100 scaling plan
python experiments/a_matrix/scripts/run_setup.py \
  --setup gpu_scaling --serial --repetitions 3 --dry-run

# Recovery plan
python experiments/a_matrix/scripts/run_setup.py \
  --setup recovery --repetitions 3 --dry-run

# Text-shortening plan
python experiments/a_matrix/scripts/run_setup.py \
  --setup text_shortening --repetitions 3 --dry-run

# VLM plan
python experiments/a_matrix/scripts/run_setup.py \
  --setup vlm_enrichment --repetitions 3 --dry-run

# A100 plan, after copying the H100 workload
python experiments/a_matrix/scripts/run_setup.py \
  --setup a100_4gpu --repetitions 3 --dry-run
```

The generic `experiments/run_all.sh` and smoke calibrator remain useful
development/runbook tools, but the dedicated publication drivers should be used
for final measurements because they do not reuse historical cells or
recalibrate publication workload sizes.

## Monitoring

`python experiments/progress_tracker.py` renders a live dashboard for
`run_setup.py` execution. `--once` prints one snapshot and `--json` emits
machine-readable output.

```bash
python experiments/progress_tracker.py
python experiments/progress_tracker.py --once
python experiments/progress_tracker.py --once --setup recovery
```

## Environment and venvs

The experiments use multiple Python environments because competitor dependency
sets differ:

| Venv | Purpose |
|---|---|
| `.venv` | MMIRAGE core and SGLang |
| `.venv-datatrove` | DataTrove competitor |
| `.venv-nemo_curator` | NeMo Curator competitor |
| `.venv-distilabel` | Distilabel competitor |
| `.venv-ray_data_llm` | Ray Data LLM competitor |

The launchers use:

- `MMIRAGE_DATATROVE_PYTHON`;
- `MMIRAGE_NEMO_CURATOR_PYTHON`;
- `MMIRAGE_DISTILABEL_PYTHON`;
- `MMIRAGE_RAY_DATA_LLM_PYTHON`.

Python 3.12 requires `SETUPTOOLS_USE_DISTUTILS=local` for the affected vLLM
competitor environments.

## Repository policy

- Fixed recipes and execution configs live under `experiments/<name>/configs/`.
- Runnable entry points live under experiment/script directories.
- Shared benchmark scaffolding lives under `experiments/_shared/`.
- Generated outputs remain local and git-ignored.
- Persistent benchmark evidence should be archived with its run manifests,
  environment metadata, workload metadata/checksums, raw repetitions, and
  aggregated summaries.
