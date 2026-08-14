# MMIRAGE Shard Recovery Experiment

This experiment measures how much work MMIRAGE avoids recomputing after selected shard workloads are deliberately terminated. Kubernetes is only external orchestration: each pod still runs MMIRAGE's normal local shard path, state files, retry detection, statistics, and merge command.

Run all commands from the repository root inside the MMIRAGE container unless a step says otherwise.

## What It Measures

The workload has 16 logical shards. The controller runs four conditions:

| Condition | Deliberately terminated shards |
|---|---|
| `baseline` | none |
| `fail_1` | `3` |
| `fail_4` | `1,5,9,13` |
| `fail_8` | `0,2,4,6,8,10,12,14` |

For failure conditions, the controller terminates selected running pods with `SIGTERM`, then retries only shards MMIRAGE marks incomplete.

Primary metrics:

- `shards_recomputed_count`: retry shards launched after deliberate termination.
- `completed_shards_reused`: shards that completed before retry and were not recomputed.
- `rows_recomputed`: rows processed by retry shards.
- `fraction_of_total_workload_recomputed`: recomputed rows divided by final merged rows.
- `failure_plus_recovery_wall_time_seconds`: initial failure phase plus retry phases.
- `estimated_gpu_seconds_wasted`: approximate GPU seconds spent in deliberately killed pods.
- `completed_shard_outputs_unchanged_after_retry`: integrity check that successful shard outputs were preserved.

This experiment does not measure native Kubernetes fault tolerance, scheduler quality, image pull time, PVC provisioning time, or model-cache warm-up. Kubernetes is used only to launch and terminate isolated one-shard MMIRAGE processes.

## Files

| Path | Purpose |
|---|---|
| `configs/mmirage_recovery.yaml` | Fixed 16-shard MMIRAGE workload. |
| `configs/native_competitors.yaml` | Native-mode completion settings for DataTrove, NeMo Curator, Distilabel, and Ray Data LLM recovery-equivalent baselines. |
| `scripts/prepare_workload.py` | Downloads and freezes the public Hugging Face workload. |
| `scripts/run_k8s.py` | Kubernetes controller. |
| `scripts/run_local.py` | Optional local fallback runner with the same output layout. |
| `scripts/run_pod.py` | One-shard wrapper used by Kubernetes pods and the local fallback. |
| `scripts/extract_results.py` | Aggregates final CSV and JSON results. |
| `scripts/plan_native_competitor_recovery.py` | Emits dry-run manifests for native competitor recovery-equivalent runs. |
| `scripts/run_native_recovery_competitor.py` | Local native competitor recovery controller (no Kubernetes); runs the initial phase, emulates kills with `SIGTERM`, retries incomplete shards, merges, and validates. |
| `environment/` | Per-framework requirement pins for the native competitor environments. |

## Prerequisites

- Repository available inside the container. Set `MMIRAGE_REPO` below to its path; the images built
  from `docker/` place it at `/workspace/MMIRAGE`.
- Python environment with MMIRAGE GPU dependencies, `sglang==0.5.10`, CUDA-compatible PyTorch, `datasets`, and `kubectl`.
- Kubernetes or Run:ai namespace where the controller can create pods and run `kubectl exec` in them.
- Shared `ReadWriteMany` PVC mounted at the same path in the controller container and shard pods.
- H100 access. The default controller budget runs at most four one-GPU shard pods at once.

Use Bash for the commands below. Set these variables once, replacing only `IMAGE` unless your cluster requires different names:

```bash
export MMIRAGE_REPO=/workspace/MMIRAGE
cd "$MMIRAGE_REPO"
export NAMESPACE=mmirage-recovery
export PVC_NAME=mmirage-recovery-pvc
export IMAGE=<your-mmirage-gpu-image>
export MMIRAGE_RECOVERY_ROOT=/workspace/mmirage-recovery
export MAX_ACTIVE_SHARDS=4
```

If your cluster uses an H100 node selector, also set:

```bash
export GPU_PRODUCT_LABEL=NVIDIA-H100
```

Confirm the namespace grants pod permissions before preparing any data. The controller creates one
bare `Pod` per shard, `exec`s into it to send `SIGTERM`, and reads its logs:

```bash
kubectl auth can-i create pods -n "$NAMESPACE"
kubectl auth can-i create pods/exec -n "$NAMESPACE"
kubectl auth can-i get pods/log -n "$NAMESPACE"
```

If any prints `no`, use the Local Fallback at the end of this document. It runs the same 16-shard
experiment with local `SIGTERM` signals, needs no cluster permissions, and writes the layout
`extract_results.py` reads.

Build the reusable controller arguments after setting the variables:

```bash
COMMON_K8S_ARGS=(
  --namespace "$NAMESPACE"
  --pvc "$PVC_NAME"
  --image "$IMAGE"
  --shared-root "$MMIRAGE_RECOVERY_ROOT"
  --max-active-shards "$MAX_ACTIVE_SHARDS"
  --config "$MMIRAGE_REPO/experiments/shard_recovery/configs/mmirage_recovery.yaml"
  --config-in-container "$MMIRAGE_REPO/experiments/shard_recovery/configs/mmirage_recovery.yaml"
  --repo-dir-in-container "$MMIRAGE_REPO"
)
if [ -n "${GPU_PRODUCT_LABEL:-}" ]; then
  COMMON_K8S_ARGS+=(--gpu-product-label "$GPU_PRODUCT_LABEL")
fi
```

## Native Competitor Recovery (No Kubernetes)

The native competitor recovery runs the same recovery experiment as the K8s baseline but with DataTrove, NeMo Curator, Distilabel, or Ray Data LLM workers on the local node, no Kubernetes, and a deliberately emulated pod termination (`SIGTERM` to the designated failure shards). It is implemented and runnable via `scripts/run_native_recovery_competitor.py`, the local equivalent of `run_local.py` for the native backends.

The comparison uses a benchmark-level equivalence contract:

- same prepared input workload and expected ID order (`--shared-root/data/ultrachat_200k/`)
- same 16 logical shard IDs, split into waves of `--max-active-shards` (default 4) workers
- same killed shard sets for `baseline`, `fail_1`, `fail_4`, and `fail_8`
- same model family and decoding settings where the framework exposes them
- kills are emulated with `SIGTERM` to the worker process group after `--kill-after-seconds` (default 20)
- retry only shards without a valid completion marker
- preserve completed shard output hashes across retry
- merge outputs in original input order

The worker itself is the scaling experiment's `experiments/single_node_h100_scaling/scripts/native_shard_worker.py` invoked with `--prompt-style raw --id-field mmirage_id`.

Report Ray task retry, DataTrove checkpointing, NeMo pipeline retry, and Distilabel pipeline retry separately from the normalized benchmark shard retry. Do not claim any competitor has native MMIRAGE-equivalent shard state unless the final implementation uses that competitor's own public API to expose it.

### Environments

The four frameworks need their own Python environments, exactly as for the scaling experiment (see `experiments/single_node_h100_scaling/README.md` section 6.2): `.venv-datatrove`, `.venv-nemo`, `.venv-distilabel`, `.venv-ray`. Each run is executed with that framework's venv python.

### Dry-run / plan

Print the run manifest without launching any worker:

```bash
python experiments/shard_recovery/scripts/plan_native_competitor_recovery.py \
  --framework all \
  --condition all \
  --rep 1 \
  --gpu-ids 0,1,2,3
```

or

```bash
.venv-datatrove/bin/python experiments/shard_recovery/scripts/run_native_recovery_competitor.py \
  --framework datatrove --condition fail_4 --rep 1 \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" --gpu-ids 0,1,2,3 --dry-run
```

### Run a condition

Prepare the workload first (section 2), then run a condition with the framework venv:

```bash
.venv-datatrove/bin/python experiments/shard_recovery/scripts/run_native_recovery_competitor.py \
  --framework datatrove \
  --condition fail_4 \
  --rep 1 \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --gpu-ids 0,1,2,3
```

Run each framework with its own venv (`nemo_curator`, `distilabel`, `ray_data_llm`) and each condition (`baseline`, `fail_1`, `fail_4`, `fail_8`) and repetition. The controller runs the initial phase, snapshots completed shard outputs, retries incomplete shards in rounds (up to `--max-rounds`, default 3), merges in expected ID order, and writes:

```text
$MMIRAGE_RECOVERY_ROOT/native_competitors/<framework>/<condition>/rep_<R>/
  controller/run_manifest.json
  controller/phase_initial.json
  controller/phase_retry_<N>.json
  controller/completed_shards_before_retry.json
  state/shard_<i>/{input.jsonl,output.jsonl,running.json,status.json,worker.log}
  raw_logs/<phase>/...
  merged/merged.jsonl
  summary.json
  validation.json
```

`validation.json` checks `no_missing_ids`, `no_duplicate_ids`, `no_unexpected_ids`, `order_after_merge_matches_expected`, `completed_shard_outputs_unchanged_after_retry`, and `retry_only_incomplete_or_killed_shards`. The exit code is nonzero if validation fails. Metrics for the paper (shards recomputed, rows recomputed, fraction recomputed, initial and retry wall times) are in `summary.json`.

## 1. Create Or Confirm The PVC

If your namespace and PVC already exist, skip this step after confirming the PVC is mounted at `$MMIRAGE_RECOVERY_ROOT` in the controller container.

If your cluster has a default storage class that supports `ReadWriteMany`, this generic command is sufficient:

```bash
kubectl create namespace "$NAMESPACE" || true
kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $PVC_NAME
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 2Ti
EOF
```

If your cluster requires a storage class, add `storageClassName: <your-storage-class>` under `spec:` before applying.

## 2. Prepare Data

This writes the fixed JSONL workload, input order manifest, and provenance metadata to the shared root.

```bash
python experiments/shard_recovery/scripts/prepare_workload.py \
  --output-root "$MMIRAGE_RECOVERY_ROOT" \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --num-records 65536 \
  --seed 20260813
```

Expected files:

```text
$MMIRAGE_RECOVERY_ROOT/data/ultrachat_200k/subset.jsonl
$MMIRAGE_RECOVERY_ROOT/data/ultrachat_200k/id_order.jsonl
$MMIRAGE_RECOVERY_ROOT/data/ultrachat_200k/manifest.json
```

## 3. Run The Clean Baseline

Run the baseline first. Later failure timings use the median baseline shard runtime when available.

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition baseline \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}" \
  --overwrite
```

Merge the baseline:

```bash
mmirage merge-dir \
  --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/baseline/rep_01/output" \
  --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/baseline/rep_01/merged"
```

## 4. Run Failure Conditions

Run `fail_1`:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition fail_1 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}" \
  --overwrite

python experiments/shard_recovery/scripts/run_k8s.py retry \
  --condition fail_1 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}"

mmirage merge-dir \
  --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_1/rep_01/output" \
  --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_1/rep_01/merged"
```

Run `fail_4`:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition fail_4 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}" \
  --overwrite

python experiments/shard_recovery/scripts/run_k8s.py retry \
  --condition fail_4 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}"

mmirage merge-dir \
  --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_4/rep_01/output" \
  --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_4/rep_01/merged"
```

Run `fail_8`:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition fail_8 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}" \
  --overwrite

python experiments/shard_recovery/scripts/run_k8s.py retry \
  --condition fail_8 \
  --rep 1 \
  "${COMMON_K8S_ARGS[@]}"

mmirage merge-dir \
  --input-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_8/rep_01/output" \
  --output-dir "$MMIRAGE_RECOVERY_ROOT/runs/fail_8/rep_01/merged"
```

## 5. Check Status

Use this if a run appears stuck or before extracting results:

```bash
python experiments/shard_recovery/scripts/run_k8s.py status \
  --condition fail_4 \
  --rep 1 \
  --shared-root "$MMIRAGE_RECOVERY_ROOT"
```

The command exits with nonzero status if shards are incomplete.

## 6. Extract Results

Run this after every completed condition has been merged:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4,fail_8 \
  --reps 1 \
  --config "$MMIRAGE_REPO/experiments/shard_recovery/configs/mmirage_recovery.yaml"
```

Expected outputs:

```text
$MMIRAGE_RECOVERY_ROOT/results/recovery_results.csv
$MMIRAGE_RECOVERY_ROOT/results/recovery_results.json
```

The extractor reports recomputed rows exactly. Token recomputation is reported as the output tokens generated by successful retry shards, because MMIRAGE does not record token progress inside a killed pod.

Expected tree after one completed repetition of every condition:

```text
$MMIRAGE_RECOVERY_ROOT/
  data/
    ultrachat_200k/
      subset.jsonl
      id_order.jsonl
      manifest.json
  runs/
    baseline/
      rep_01/
        controller/
          phase_initial.json
          initial_w01_pods.yaml
        raw_logs/
        output/
        state/
        merged/
    fail_1/
      rep_01/
        controller/
          phase_initial.json
          completed_shards_before_retry.json
          completed_shards_after_retry.json
          phase_retry_1.json
        raw_logs/
        output/
        state/
        merged/
    fail_4/
    fail_8/
  results/
    recovery_results.csv
    recovery_results.json
```

For a successful recovery run, `recovery_results.json` should report zero missing, duplicate, and unexpected IDs for each condition.

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Controller cannot create pods | Namespace, RBAC, image, or quota is wrong. | Check `kubectl -n "$NAMESPACE" get events`, verify `IMAGE`, and confirm the namespace allows pod creation. |
| Pods start but cannot see data | PVC is not mounted at the same path in controller and shard pods. | Confirm `$MMIRAGE_RECOVERY_ROOT` exists in both places and that `subset.jsonl` is visible from a pod. |
| Pods stay pending | No matching H100 nodes, bad GPU product label, or insufficient quota. | Check pod events and adjust `GPU_PRODUCT_LABEL` or resource quota. |
| `retry` exits nonzero | Some shard statuses remain incomplete or blocked. | Run the `status` command and inspect `state/shard_<id>/status.json` plus `raw_logs/<phase>/`. |
| `merge-dir` fails | One or more shard output directories are missing or incomplete. | Run `status`, retry failed shards, then rerun `mmirage merge-dir`. |
| Extractor reports missing or duplicate IDs | Merge output does not match the prepared input order. | Preserve the run directory, inspect `merged/`, `id_order.jsonl`, and `recovery_results.json`; do not overwrite the failed run before diagnosis. |

## Reproducibility Metadata

Keep these with any reported result:

- Git commit of the branch used to run the controller.
- Container image tag or digest passed as `IMAGE`.
- Kubernetes namespace, PVC name, storage class if applicable, and GPU product label.
- `MAX_ACTIVE_SHARDS`, `--rep`, `--baseline-rep`, and any explicit `--kill-after-seconds`.
- `data/ultrachat_200k/manifest.json`, including dataset revision and checksums.
- All `controller/phase_*.json`, `controller/*_pods.yaml`, and `raw_logs/` files for each condition.
- Final `results/recovery_results.csv` and `results/recovery_results.json`.

## Paper Artifact Mapping

Use these generated files for paper artifacts:

| Paper artifact | Source file |
|---|---|
| Per-condition recovery table | `$MMIRAGE_RECOVERY_ROOT/results/recovery_results.csv` |
| Full structured metrics and integrity detail | `$MMIRAGE_RECOVERY_ROOT/results/recovery_results.json` |
| Audit trail for failure timing and pod termination | `$MMIRAGE_RECOVERY_ROOT/runs/<condition>/rep_<NN>/controller/phase_*.json` |
| Raw Kubernetes logs for appendix/debugging | `$MMIRAGE_RECOVERY_ROOT/runs/<condition>/rep_<NN>/raw_logs/` |

## Repetitions

Do not overwrite successful runs when collecting repetitions. Increment `--rep` and keep all repetitions:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition --condition baseline --rep 2 "${COMMON_K8S_ARGS[@]}" --overwrite
```

Extract multiple repetitions with:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4,fail_8 \
  --reps 1,2,3 \
  --config "$MMIRAGE_REPO/experiments/shard_recovery/configs/mmirage_recovery.yaml"
```

## Local Fallback

Use `scripts/run_local.py` only if Kubernetes pod creation or `kubectl exec` is unavailable. It writes the same run directory layout but emulates pod termination with local `SIGTERM` signals.

```bash
python experiments/shard_recovery/scripts/run_local.py run-condition \
  --condition baseline \
  --rep 1 \
  --shared-root "$MMIRAGE_RECOVERY_ROOT" \
  --max-active-shards 4 \
  --gpu-ids 0,1,2,3 \
  --overwrite
```

Use the same `run-condition`, `retry`, `merge-dir`, and `extract_results.py` sequence as the Kubernetes run, replacing `run_k8s.py` with `run_local.py`.

Pass `--config` to `extract_results.py` as shown in step 6.

## Interpretation Boundary

This experiment supports the claim that MMIRAGE recovery is shard-scoped and preserves completed shard work. It does not show that MMIRAGE has a native Kubernetes backend or Kubernetes-level fault tolerance.

Raw pod logs and pod JSON are retained under:

```text
$MMIRAGE_RECOVERY_ROOT/runs/<condition>/rep_<NN>/raw_logs/<phase>/
$MMIRAGE_RECOVERY_ROOT/runs/<condition>/rep_<NN>/controller/
```
