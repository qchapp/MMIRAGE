# MMIRAGE Shard Recovery

Measures how much work MMIRAGE avoids recomputing after selected shard workloads
are deliberately terminated. 16 logical shards, 4 conditions, at most 4
concurrent one-GPU shard workers. Kubernetes is only external orchestration; the
in-pod path is `run_local.py`, which emulates pod termination with local
`SIGTERM` and writes the same layout.

| Condition | Deliberately terminated shards |
|---|---|
| `baseline` | none |
| `fail_1` | `3` |
| `fail_4` | `1,5,9,13` |

Metrics: `shards_recomputed_count`, `completed_shards_reused`, `rows_recomputed`,
`fraction_of_total_workload_recomputed`,
`failure_plus_recovery_wall_time_seconds`, and the integrity check
`completed_shard_outputs_unchanged_after_retry`. Workload size lives in
`configs/workload_size.yaml` (`num_records`, written by
`experiments/smoke/calibrate.py`; prepare with no flags to use it).

## Environment

```
export MMIRAGE_RECOVERY_ROOT=/workspace/mmirage-recovery
export MMIRAGE_REPO=/workspace/MMIRAGE
cd "$MMIRAGE_REPO"
```

## Prepare

```
python experiments/shard_recovery/scripts/prepare_workload.py --output-root "$MMIRAGE_RECOVERY_ROOT"
```

Writes `$MMIRAGE_RECOVERY_ROOT/data/ultrachat_200k/{subset.jsonl,id_order.jsonl,manifest.json}`.

## Run the in-pod experiment (4 GPUs)

Baseline first (failure timings reuse its median shard runtime):

```
python experiments/shard_recovery/scripts/run_local.py run-condition \
  --condition baseline --rep 1 --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --max-active-shards 4 --gpu-ids 0,1,2,3 --overwrite
```

Then for each of `fail_1`, `fail_4` and each repetition `--rep <r>`:

```
python experiments/shard_recovery/scripts/run_local.py run-condition \
  --condition <cond> --rep <r> --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --max-active-shards 4 --gpu-ids 0,1,2,3 --overwrite
python experiments/shard_recovery/scripts/run_local.py retry \
  --condition <cond> --rep <r> --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --max-active-shards 4 --gpu-ids 0,1,2,3
mmirage merge-dir \
  --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/<cond>/rep_<NN>/output" \
  --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/<cond>/rep_<NN>/merged"
```

`status` prints shard state (nonzero exit if shards incomplete). Do not overwrite
successful runs; increment `--rep` for repetitions.

## Extract results

```
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4 \
  --reps 1,2,3 \
  --config "$MMIRAGE_REPO/experiments/shard_recovery/configs/mmirage_recovery.yaml"
```

Writes `$MMIRAGE_RECOVERY_ROOT/results/recovery_results.csv` and `.json`. A
successful recovery reports zero missing/duplicate/unexpected IDs per condition.

## Kubernetes path

`run_k8s.py` provides the same commands against real pods (requires a
`ReadWriteMany` PVC mounted at the same path in controller and shard pods,
namespace RBAC, and the MMIRAGE GPU image). It is the paper's original harness;
`run_local.py` is the in-pod equivalent.

Confirm RBAC before preparing data — the controller creates one bare `Pod` per
shard, `exec`s into it to send `SIGTERM`, and reads its logs:

```bash
kubectl auth can-i create pods -n "$NAMESPACE"
kubectl auth can-i create pods/exec -n "$NAMESPACE"
kubectl auth can-i get pods/log -n "$NAMESPACE"
```

If any prints `no`, use the `run_local.py` path above instead; it runs the same
conditions with local `SIGTERM` signals and writes the layout
`extract_results.py` reads. When the image mounts the repo at a different path
than your working tree, pass `--repo-dir-in-container` (and `--config-in-container`
for the YAML) so pod-side commands find the config; see `run_k8s.py --help`.

## Native competitor recovery

`run_native_recovery_competitor.py` runs the same conditions with DataTrove /
NeMo Curator / Distilabel / Ray Data LLM workers (no Kubernetes), using the
scaling experiment's `native_shard_worker.py --prompt-style raw --id-field
stable_id`, each framework in its own venv (see the scaling README section on
environments). The worker is launched with `SETUPTOOLS_USE_DISTUTILS=local`
forced so the distilabel / ray_data_llm vllm imports work on Python 3.12 (no
stdlib `distutils`). Benchmark-level equivalence contract: same input + expected
ID order, same shard sets, `SIGTERM` emulated kills, retry only incomplete
shards, completed-shard hashes preserved, merge in input order. `--dry-run`
prints the manifest. Report framework-native retry semantics separately from the
normalized benchmark shard retry.

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `retry` exits nonzero | Some shard statuses remain incomplete/blocked. | `status`, inspect `state/shard_<id>/status.json` and `raw_logs/<phase>/`. |
| `merge-dir` fails | Shard output dirs missing/incomplete. | `status`, retry, re-merge. |
| Extractor reports missing/duplicate IDs | Merge doesn't match prepared input order. | Preserve the run dir, inspect `merged/` + `id_order.jsonl`; don't overwrite before diagnosing. |
| GPU OOM / crash on 4-way concurrency | Wrong GPU or memory pressure. | Check `nvidia-smi`; run `--max-active-shards 2` and record it in metadata. |
| distilabel shard fails: `ValueError: ... KV cache ... larger than the available KV cache memory` | vLLM's default `gpu_memory_utilization` leaves too little KV-cache headroom for the 40 960-token Qwen3-4B context while 4 engines share the node. | Already handled in `native_frameworks.py` (`run_distilabel` forces `0.85`); do not override with a higher value on 80 GiB GPUs. |
| distilabel shard fails: `DistilabelUserError` about the `instruction` column | distilabel 1.5.3's default template reads `{{ instruction }}`, but the worker feeds `columns=["prompt"]`. | Already handled (`template="{{ prompt }}"` in `run_distilabel`). |
| distilabel worker returns 0 rows / empty output | A stale distilabel execution cache short-circuits `Pipeline.run`. | Already handled (`use_cache=False` in `run_distilabel`). |
| distilabel worker fails: `TypeError` on `logits_processors` in `SamplingParams` | distilabel always passes that kwarg; modern vLLM (msgspec `Struct`) removed it. | Already handled (`_SamplingParamsCompat` shim in `native_frameworks.py`). |
| distilabel retry rounds never converge: shard outputs contain exactly 2x the input rows (neighbour-shard indices) | distilabel hashes the pipeline definition, not the dataset, so concurrent workers on one host share one execution cache and race on the batch manager, polluting each shard's leaf dataset. | Already handled (`run_distilabel` gives every worker its own per-shard `cache_dir`, wiped per call). If you see it again, confirm each shard writes to its own `<state>/shard_<id>/distilabel_cache/`. |
| distilabel / ray_data_llm worker fails with `"vLLM is not installed"` | Python 3.12 venv missing the `distutils` shim: `SETUPTOOLS_USE_DISTUTILS=stdlib` (or unset) in the launching shell skips the `.pth` shim, so `setuptools` → `distutils.filelist` fails and vllm looks absent. | Launch via `run_setup.py` / `run_all.sh` (they force `local`), or export `SETUPTOOLS_USE_DISTUTILS=local` in the shell; see the log `.../shard_0/status.json` traceback ending in `setuptools/monkey.py`. |

## Reproducibility

Keep the git commit, `--rep`/`--max-active-shards`/`--kill-after-seconds`,
`data/ultrachat_200k/manifest.json`, all `controller/phase_*.json` and
`raw_logs/`, and the final `recovery_results.{csv,json}`. Interpretation
boundary: this shows shard-scoped recovery and preserved completed shard work,
not a native Kubernetes backend.
