# A Matrix (Consolidated Comparison A + B)

The A matrix is the consolidated fast-run suite: the scaling experiments
(`gpu_scaling` and `a100_4gpu`), the shard-recovery experiment (`recovery`)
with a `stable_id` workload, and the Comparison-B task comparisons
(`text_shortening`, `vlm_enrichment`) with single-instruction fixes. Every
experiment consumes the **same task**: rewrite an UltraChat user prompt with a
fixed prompt template (`task.yaml`), model `Qwen/Qwen3-4B`, fixed shard split
and output contract.

| Setup | Frameworks | GPU points | Notes |
|---|---|---|---|
| `gpu_scaling` | mmirage, raw_sglang, datatrove, nemo_curator | 1 / 2 / 4 | S1 strong scaling on H100 |
| `a100_4gpu` | same 4 | 4 | S2 done node (4x A100) |
| `recovery` | mmirage + raw_sglang, datatrove, nemo_curator, distilabel, ray_data_llm | 4 | 16 shards; conditions baseline / fail_1 / fail_4 |
| `text_shortening` | mmirage, datatrove, nemo_curator | 4 | summarize-style prompt (B1 fix) |
| `vlm_enrichment` | mmirage, sglang, datatrove, nemo_curator | 4 | VLM reformat prompt (B2 fix) |

## Run

`bash experiments/run_all.sh` is the single entry point: on one 4-GPU node it
runs the whole matrix — smoke → calibrate → scaling (all GPU points 1/2/4) →
recovery → text → vlm, delegating the run stages to `scripts/run_setup.py`.
`--only`/`--skip` select stages. By default the MMIRAGE-only cells already
covered by the 2026-08-15 fast-runs reproduction are reused, not rerun — see
[Reusing the 2026-08-15 fast-runs](#reusing-the-2026-08-15-fast-runs).
`bash experiments/run_all.sh --rerun-reused` reruns every cell from scratch.

To run individual experiments directly:

```bash
# Scaling (all frameworks, 1/2/4 GPU points)
python experiments/a_matrix/scripts/run_setup.py --setup gpu_scaling

# Recovery (MMIRAGE + all native competitors, fail_1/fail_4 conditions)
python experiments/a_matrix/scripts/run_setup.py --setup recovery

# Text shortening (MMIRAGE + DataTrove + NeMo Curator)
python experiments/a_matrix/scripts/run_setup.py --setup text_shortening

# VLM enrichment (MMIRAGE + SGLang + DataTrove + NeMo Curator)
python experiments/a_matrix/scripts/run_setup.py --setup vlm_enrichment

# Extract recovery results after runs complete
python experiments/a_matrix/scripts/run_setup.py --setup recovery --extract
```

Monitor a live run with `python experiments/progress_tracker.py` (`--once` for
a single snapshot, `--json` for machine-readable output). Recovery additionally
needs the distilabel and ray_data_llm venvs on top of datatrove/nemo_curator;
their interpreters come from `MMIRAGE_DISTILABEL_PYTHON` /
`MMIRAGE_RAY_DATA_LLM_PYTHON` (run_all.sh sets defaults). Python 3.12 requires
`SETUPTOOLS_USE_DISTUTILS=local` (forced by run_all.sh and run_setup.py), or
the distilabel/ray_data_llm vllm imports fail on the missing stdlib `distutils` —
never export `SETUPTOOLS_USE_DISTUTILS=stdlib` in the launching shell.

`run_setup.py` supports `--dry-run` (prints the plan, verifies templates),
`--setup <name>` to run one setup, `--prepare`, `--extract`, `--overwrite`,
`--gpus`, and `--reuse-fastruns`. It is a thin scheduler: each unit delegates
to the existing runners (`single_node_h100_scaling/scripts/run.py`, the native
scaling wrappers, `shard_recovery/scripts/run_local.py` and the native recovery
harness). One scaling unit per framework runs at a time; units are pinned to
free physical GPUs; recovery/text/vlm units take all 4 GPUs. Per-stage logs
land in `experiments/run_all_logs/<stage>.log`; per-unit logs and a final
`status.json` live under `experiments/run_all_logs/a_matrix/`.

## Reusing the 2026-08-15 fast-runs

The MMIRAGE-only cells of the fast-run reproduction on 2026-08-15 (see
`RUNLOG.md`, pod `mmirage-exp-4gpu-0-0`) were produced using the OLD batch
configurations. They are **not** byte/config compatible with the corrected
publication batch=64 suite and must **not** be reused for the corrected
publication benchmark. The historical fast-run cells are retained as
provenance only.

To run the corrected publication benchmark, use `--rerun-reused` (via
`experiments/run_all.sh`) or omit `--reuse-fastruns` (via `run_setup.py`)
to rerun all cells from scratch with the corrected settings.

## Workload and sizes

One deterministic UltraChat workload (`experiments/a_matrix/workload/`,
git-ignored) serves every setup: same dataset, seed, selection and
normalization as the single-node scaling prep, keyed by `stable_id`. Two
independent sizes are calibrated:

* `configs/workload_size.yaml` `num_rows` — the scaling workloads (`gpu_scaling`
  and `a100_4gpu`).
* `configs/recovery_size.yaml` `recovery_num_rows` — the shared-root subset
  written to `<shared-root>/data/ultrachat_200k/` for the recovery controllers.

`prepare_workload.py --shared-root` writes that subset plus `id_order.jsonl`
keyed on `stable_id`; the MMIRAGE and native recovery controllers read them.

## Recovery conditions

`recovery` runs 16 shards with 4 active, conditions `baseline` (clean),
`fail_1` (1 worker terminated, then MMIRAGE retry) and `fail_4` (4 workers
terminated, then retry). MMIRAGE runs all three conditions; the five native
frameworks run `fail_1` and `fail_4` only (their recovery wall time is
absolute, no baseline needed) → 13 runs total.

## B fixes

* B1 (text): the MMIRAGE recipe is already correct; the natives now receive the
  same instruction through `--prompt-style summarize`
  (`SUMMARIZE_PROMPT_TEMPLATE` in `experiments/_shared/native_frameworks.py`).
* B2 (vlm): `VLM_REFORMAT_TEMPLATE` in `experiments/_shared/vlm_runners.py` is
  applied by the native VLM runners (sglang, datatrove, nemo_curator).

`run_setup.py` verifies these templates are byte-identical (modulo the
placeholder name) before launching anything and refuses to run otherwise.

## Superseded experiments

`experiments/single_node_h100_scaling` and `experiments/shard_recovery` are
kept as libraries/runbooks but are no longer run directly: scaling and recovery
now run through this experiment. `shard_recovery/scripts/prepare_workload.py` is
superseded by `prepare_workload.py --shared-root` and writes the old
`mmirage_id` schema; do not run it against the current shared root.

`A-MATRIX` `raw_sglang` evaluates a complete Direct SGLang runner.
`experiments/raw_sglang_overhead` is a separate endpoint-matched experiment
used to estimate MMIRAGE abstraction/orchestration overhead relative to the
same SGLang serving endpoint.
