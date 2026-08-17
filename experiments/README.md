# Experiments

Runbooks for the MMIRAGE benchmark experiments. All commands run from the
repository root inside the MMIRAGE GPU environment. Generated artifacts
(`workload/`, `results/`, `runs/`) are git-ignored; commit only code, configs,
and READMEs.

## Experiment overview

| Experiment | What it measures | Frameworks | GPU points | Status |
|---|---|---|---|---|
| `a_matrix` (gpu_scaling) | Strong scaling of text rewriting (1→2→4 H100) | MMIRAGE, raw_sglang, DataTrove, NeMo Curator | 1 / 2 / 4 | **Complete** (all 12 units) |
| `a_matrix` (recovery) | Shard-scoped fault recovery (16 shards, 4 active) | MMIRAGE, raw_sglang, DataTrove, NeMo Curator, Distilabel, Ray Data LLM | 4 | **In progress** (9/13 cells; datatrove failed, distilabel running) |
| `a_matrix` (text_shortening) | Article→summary throughput at fixed workload | MMIRAGE, DataTrove, NeMo Curator | 4 | **Complete** (3 units) |
| `a_matrix` (vlm_enrichment) | Image-caption enrichment throughput | MMIRAGE, SGLang, DataTrove, NeMo Curator | 4 | **Complete** (4 units) |
| `single_node_h100_scaling` | Per-framework GPU scaling (superseded by `a_matrix`) | MMIRAGE, raw_sglang, DataTrove, NeMo Curator | 1 / 2 / 4 | Superseded |
| `shard_recovery` | Shard-recovery controllers and runners (superseded by `a_matrix`) | MMIRAGE + 5 native competitors | 4 | Superseded |
| `task_comparison/text_shortening` | Legacy text-shortening runner (superseded by `a_matrix`) | MMIRAGE, DataTrove, NeMo Curator | 4 | Superseded |
| `task_comparison/vlm_enrichment` | Legacy VLM-enrichment runner (superseded by `a_matrix`) | MMIRAGE, SGLang, DataTrove, NeMo Curator | 4 | Superseded |

## The A matrix (primary experiment)

The A matrix is the consolidated benchmark suite. Every experiment consumes the
**same task**: rewrite an UltraChat user prompt with a fixed prompt template,
model `Qwen/Qwen3-4B`, fixed shard split, and output contract.

### GPU scaling (`gpu_scaling`)

Strong scaling on H100 GPUs: same workload across 1, 2, and 4 GPUs for each
framework. Measures throughput (`aggregate_output_tok_s`, `rows_s`), per-GPU
efficiency (`output_tok_s_per_gpu`), and speedup vs. the 1-GPU baseline.

| Framework | GPU points | Reps | Status |
|---|---|---|---|
| MMIRAGE | 1, 2, 4 | 3 each | Complete |
| raw_sglang | 1, 2, 4 | 3 each | Complete |
| DataTrove | 1, 2, 4 | 3 each | Complete |
| NeMo Curator | 1, 2, 4 | 3 each | Complete |

### Recovery (`recovery`)

16 shards, 4 active simultaneously. Three conditions:

- **baseline**: clean run, no faults (MMIRAGE only)
- **fail_1**: 1 worker terminated mid-run, then retry
- **fail_4**: 4 workers terminated mid-run, then retry

MMIRAGE runs all three conditions (baseline + fail_1 + fail_4). The five native
competitors run fail_1 and fail_4 only (their recovery wall time is absolute,
no baseline needed) → 13 runs total.

| Framework | baseline | fail_1 | fail_4 | Status |
|---|---|---|---|---|
| MMIRAGE | ok (4m 26s) | ok (5m 16s) | ok (5m 18s) | **Complete** |
| raw_sglang | — | ok (5m 06s) | ok (4m 58s) | **Complete** |
| DataTrove | — | failed | failed | Failed (TRANSFORMERS_CACHE bug, fix committed) |
| NeMo Curator | — | ok (10m 36s) | ok (8m 24s) | **Complete** |
| Distilabel | — | running | queued | In progress |
| Ray Data LLM | — | queued | queued | Queued |

**Known issue**: DataTrove recovery cells failed because `TRANSFORMERS_CACHE`
overrode the `HF_HOME/hub` cache lookup. Fix committed (removes the env var);
datatrove cells need re-run after the current recovery run finishes.

### Text shortening (`text_shortening`)

Article→summary transformation with a `summarize`-style prompt. All frameworks
receive the same instruction through their native API.

| Framework | Status |
|---|---|
| MMIRAGE | Complete |
| DataTrove | Complete |
| NeMo Curator | Complete |

### VLM enrichment (`vlm_enrichment`)

Image-caption enrichment with a `VLM_REFORMAT_TEMPLATE` applied by native VLM
runners.

| Framework | Status |
|---|---|
| MMIRAGE | Complete |
| SGLang | Complete |
| DataTrove | Complete |
| NeMo Curator | Complete |

## Running everything unattended

`bash experiments/run_all.sh` runs the whole suite end to end without human
intervention:

1. **Preflight**: venv check, 4 GPUs visible, HF token, all competitor
   interpreters present.
2. **Smoke**: one fast synthetic run to verify runners work.
3. **Calibrate**: measure per-cell timings at small size, rewrite workload sizes
   to fit the wall-clock budget.
4. **Scaling**: all frameworks at 1/2/4 GPU points (12 units).
5. **Recovery**: MMIRAGE baseline + fail_1/fail_4 + all native competitors (13
   cells). Extracts `recovery_results.json` and persists results to
   `experiments/a_matrix/results/recovery/` on `/lightscratch`.
6. **Text**: MMIRAGE + DataTrove + NeMo Curator (3 units).
7. **VLM**: MMIRAGE + SGLang + DataTrove + NeMo Curator (4 units).

```bash
# Run everything
bash experiments/run_all.sh

# Skip specific stages
bash experiments/run_all.sh --skip vlm,text

# Run only recovery
bash experiments/run_all.sh --only recovery

# Rerun everything including previously-reused MMIRAGE cells
bash experiments/run_all.sh --rerun-reused
```

By default the MMIRAGE-only cells already covered by the 2026-08-15 fast-run
reproduction are reused, not rerun. `--rerun-reused` overrides this.

## Running individual experiments

Each experiment can be run in isolation:

```bash
# Scaling: all frameworks at 1/2/4 GPU points
python experiments/a_matrix/scripts/run_setup.py --setup gpu_scaling

# Recovery: MMIRAGE + all native competitors
python experiments/a_matrix/scripts/run_setup.py --setup recovery

# Text shortening: MMIRAGE + DataTrove + NeMo Curator
python experiments/a_matrix/scripts/run_setup.py --setup text_shortening

# VLM enrichment: MMIRAGE + SGLang + DataTrove + NeMo Curator
python experiments/a_matrix/scripts/run_setup.py --setup vlm_enrichment

# Extract recovery results only (after recovery runs)
python experiments/a_matrix/scripts/run_setup.py --setup recovery --extract
```

## Monitoring

`python experiments/progress_tracker.py` renders a live dashboard for the
`run_setup.py` scheduler: per-unit status derived from the per-cell logs,
elapsed vs. expected time, GPU usage, and recovery progress. `--once` prints a
single snapshot, `--json` emits machine-readable output, `--setup <name>`
filters the view.

```bash
# Live dashboard
python experiments/progress_tracker.py

# Single snapshot
python experiments/progress_tracker.py --once

# Recovery only
python experiments/progress_tracker.py --once --setup recovery

# Machine-readable
python experiments/progress_tracker.py --once --json
```

## Sizes and the smoke calibrator

Each experiment's default workload size lives in its `configs/workload_size.yaml`
and is committed. `experiments/smoke/` contains the one-time calibrator that
measures per-cell timings at a small size and rewrites those size files so every
experiment fits its wall-clock budget. See `experiments/smoke/README.md`.

## Environment and venvs

The experiments require multiple Python virtual environments, each with
different dependency versions:

| Venv | Purpose | Key dependency |
|---|---|---|
| `.venv` | MMIRAGE core, sglang | sglang, flashinfer 0.6.7.post2 |
| `.venv-datatrove` | DataTrove competitor | datatrove, vllm 0.27.1, flashinfer 0.6.16.post3 |
| `.venv-nemo_curator` | NeMo Curator competitor | nemo_curator, vllm 0.27.1, flashinfer 0.6.16.post3 |
| `.venv-distilabel` | Distilabel competitor | distilabel 1.5.3, vllm 0.27.1, flashinfer 0.6.16.post3 |
| `.venv-ray_data_llm` | Ray Data LLM competitor | ray, vllm 0.25.1, flashinfer 0.6.13 |

Python 3.12 requires `SETUPTOOLS_USE_DISTUTILS=local` (forced by run_all.sh
and run_setup.py), or the distilabel/ray_data_llm vllm imports fail on the
missing stdlib `distutils`. Never export `SETUPTOOLS_USE_DISTUTILS=stdlib` in
the launching shell.

## Repository policy

- Fixed recipes and execution configs live under `experiments/<name>/configs/`.
- Runnable entry points live under `experiments/<name>/scripts/`.
- Shared benchmark scaffolding lives in `experiments/_shared/` (helper code, not
  MMIRAGE runtime code). Runner entry points never import sibling experiment
  scripts.
- Generated outputs stay local and ignored; only code, configs, and READMEs are
  committed.
- Persistent results (scaling summaries, recovery results) are stored on
  `/lightscratch` (survives pod suspension). Intermediate data (model downloads,
  shard outputs) lives on `/workspace` (ephemeral).
