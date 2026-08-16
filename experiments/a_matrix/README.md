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

`bash experiments/run_all.sh` is the single entry point: on one 4-GPU pod it
runs the whole matrix **with no pod separation** — smoke → calibrate → scaling
(all GPU points 1/2/4) → recovery → text → vlm, delegating the run stages to
`scripts/run_setup.py`. `--only`/`--skip` select stages. By default the
MMIRAGE-only cells already covered by the 2026-08-15 fast-runs reproduction are
reused, not rerun — see [Reusing the 2026-08-15 fast-runs](#reusing-the-2026-08-15-fast-runs).
`bash experiments/run_all.sh --rerun-reused` reruns every cell from scratch.

Optional parallelism: the identical work can be split across two 4-GPU nodes
with `run_setup.py --pod pod_a --reuse-fastruns` (scaling 1/2-GPU + recovery)
and `run_setup.py --pod pod_b --reuse-fastruns` (scaling 4-GPU + text + vlm),
per `schedule.yaml`. This is purely a manual optimization — nothing requires
the split. Run the shared workload preparation once first:

```
python experiments/a_matrix/scripts/prepare_workload.py \
  --output-dir experiments/a_matrix/workload \
  --shared-root "$MMIRAGE_RECOVERY_ROOT"
```

The 4x A100 point runs on its own node with `bash experiments/run_a100.sh`
(= `run_setup.py --setup a100_4gpu`); it reuses the same workload and sizes.
No fast-run exists on A100, so all four cells there are new.

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
`RUNLOG.md`, pod `mmirage-exp-4gpu-0-0`) are byte-consistent with their A-matrix
counterparts (same workload rows, model, batch size 256, 3 repetitions; the
prompt differs only in the trailing newline noted below) and are reused instead
of rerunning today:

| Reused unit | Fast-runs source | num_rows |
|---|---|---|
| `gpu_scaling/mmirage/gpu_1` | `experiments/single_node_h100_scaling/results/runs/gpu_1` | 5430 |
| `gpu_scaling/mmirage/gpu_2` | `experiments/single_node_h100_scaling/results/runs/gpu_2` | 5430 |
| `gpu_scaling/mmirage/gpu_4` | `experiments/single_node_h100_scaling/results/runs/gpu_4` | 5430 |
| `text_shortening/mmirage` | `experiments/task_comparison/text_shortening/results/runs/gpu_4` | 9471 |
| `vlm_enrichment/mmirage` | `experiments/task_comparison/vlm_enrichment/results/runs/gpu_4` | 83 |

Nothing else is reused: the scaling natives (raw_sglang, datatrove, nemo_curator)
never ran at these sizes/points, the A100 point has no prior run, recovery
switched from the `mmirage_id` schema/sizes/conditions to `stable_id`
(`recovery_num_rows`), and the text/vlm natives got new prompts (B1
`--prompt-style summarize`, B2 `VLM_REFORMAT_TEMPLATE`).

The mapping lives in `configs/reused_units.yaml`. `run_setup.py --reuse-fastruns`
skips these units and logs a notice (with a warning if the results are not
restored yet); `run_all.sh` passes the flag by default, so the reused results
are preserved by its stage cleanup. `--rerun-reused` (run_all) or omitting
`--reuse-fastruns` (run_setup) reruns them.

Reuse is valid only while:

* the calibrated sizes stay at the committed values (5430 / 9471 / 83). If a
  fresh `calibrate --apply` rewrites them (different hardware), the reused runs
  no longer match the new workload — rerun the reused cells instead;
* the archived results are restored into the locations above. The archive
  (`/lightscratch/users/azgaoui/anonlib/mmirage-fastrun-evidence-20260815/`)
  mirrors the repo tree; restore the scaling cells into the matrix layout with
  (adjust the source path to the unpacked archive):

```
for n in 1 2 4; do
  rsync -a <archive>/experiments/single_node_h100_scaling/results/runs/gpu_$n/ \
        experiments/a_matrix/results/gpu_scaling/mmirage/runs/gpu_$n/
done
```

  The text/vlm cells already sit at their matrix locations, so restoring the
  archive into the repo tree is sufficient for them.

One input caveat: the consolidated recipe renders the user prompt without the
trailing newline the 2026-08-15 recipe emitted (YAML `|-` vs `|`). It does not
affect the timing/throughput metrics the matrix reports.

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

`experiments/single_node_h100_scaling`, `experiments/shard_recovery` and
`experiments/raw_sglang_overhead` are kept as libraries/runbooks but are no
longer run directly: scaling and recovery now run through this experiment.
`raw_sglang_overhead` in particular is absorbed into the `gpu_scaling` matrix
as the `raw_sglang` framework. `shard_recovery/scripts/prepare_workload.py` is
superseded by `prepare_workload.py --shared-root` and writes the old
`mmirage_id` schema; do not run it against the current shared root.
