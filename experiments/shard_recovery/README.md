# AnonLib Shard Recovery Experiment

This experiment measures how much work AnonLib avoids recomputing after selected shard workloads are deliberately terminated. Kubernetes is only external orchestration: each pod still runs AnonLib's normal local shard path, state files, retry detection, statistics, and merge command.

Run all commands from the repository root inside the AnonLib container unless a step says otherwise.

## What It Measures

The workload has 16 logical shards. The controller runs four conditions:

| Condition | Deliberately terminated shards |
|---|---|
| `baseline` | none |
| `fail_1` | `3` |
| `fail_4` | `1,5,9,13` |
| `fail_8` | `0,2,4,6,8,10,12,14` |

For failure conditions, the controller terminates selected running pods with `SIGTERM`, then retries only shards AnonLib marks incomplete.

## Files

| Path | Purpose |
|---|---|
| `configs/anonlib_recovery.yaml` | Fixed 16-shard AnonLib workload. |
| `scripts/prepare_workload.py` | Downloads and freezes the public Hugging Face workload. |
| `scripts/run_k8s.py` | Kubernetes controller. |
| `scripts/run_local.py` | Optional local fallback runner with the same output layout. |
| `scripts/run_pod.py` | One-shard wrapper used by Kubernetes pods and the local fallback. |
| `scripts/extract_results.py` | Aggregates final CSV and JSON results. |

## Prerequisites

- Repository available inside the container at `/workspace/ANONLIB`.
- Python environment with AnonLib GPU dependencies, `sglang==0.5.10`, CUDA-compatible PyTorch, `datasets`, and `kubectl`.
- Kubernetes or Run:ai namespace where the controller can create pods and run `kubectl exec` in them.
- Shared `ReadWriteMany` PVC mounted at the same path in the controller container and shard pods.
- H100 access. The default controller budget runs at most four one-GPU shard pods at once.

Use Bash for the commands below. Set these variables once, replacing only `IMAGE` unless your cluster requires different names:

```bash
cd /workspace/ANONLIB
export NAMESPACE=anonlib-recovery
export PVC_NAME=anonlib-recovery-pvc
export IMAGE=<your-anonlib-gpu-image>
export ANONLIB_RECOVERY_ROOT=/workspace/anonlib-recovery
export MAX_ACTIVE_SHARDS=4
```

If your cluster uses an H100 node selector, also set:

```bash
export GPU_PRODUCT_LABEL=NVIDIA-H100
```

Build the reusable controller arguments after setting the variables:

```bash
COMMON_K8S_ARGS=(
  --namespace "$NAMESPACE"
  --pvc "$PVC_NAME"
  --image "$IMAGE"
  --shared-root "$ANONLIB_RECOVERY_ROOT"
  --max-active-shards "$MAX_ACTIVE_SHARDS"
)
if [ -n "${GPU_PRODUCT_LABEL:-}" ]; then
  COMMON_K8S_ARGS+=(--gpu-product-label "$GPU_PRODUCT_LABEL")
fi
```

## 1. Create Or Confirm The PVC

If your namespace and PVC already exist, skip this step after confirming the PVC is mounted at `$ANONLIB_RECOVERY_ROOT` in the controller container.

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
  --output-root "$ANONLIB_RECOVERY_ROOT" \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --num-records 65536 \
  --seed 20260813
```

Expected files:

```text
$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/subset.jsonl
$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/id_order.jsonl
$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/manifest.json
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
anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/baseline/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/baseline/rep_01/merged"
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

anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_1/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_1/rep_01/merged"
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

anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/merged"
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

anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_8/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_8/rep_01/merged"
```

## 5. Check Status

Use this if a run appears stuck or before extracting results:

```bash
python experiments/shard_recovery/scripts/run_k8s.py status \
  --condition fail_4 \
  --rep 1 \
  --shared-root "$ANONLIB_RECOVERY_ROOT"
```

The command exits with nonzero status if shards are incomplete.

## 6. Extract Results

Run this after every completed condition has been merged:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4,fail_8 \
  --reps 1
```

Expected outputs:

```text
$ANONLIB_RECOVERY_ROOT/results/recovery_results.csv
$ANONLIB_RECOVERY_ROOT/results/recovery_results.json
```

The extractor reports recomputed rows exactly. Token recomputation is reported as the output tokens generated by successful retry shards, because AnonLib does not record token progress inside a killed pod.

## Repetitions

Do not overwrite successful runs when collecting repetitions. Increment `--rep` and keep all repetitions:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition --condition baseline --rep 2 "${COMMON_K8S_ARGS[@]}" --overwrite
```

Extract multiple repetitions with:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4,fail_8 \
  --reps 1,2,3
```

## Local Fallback

Use `scripts/run_local.py` only if Kubernetes pod creation or `kubectl exec` is unavailable. It writes the same run directory layout but emulates pod termination with local `SIGTERM` signals.

```bash
python experiments/shard_recovery/scripts/run_local.py run-condition \
  --condition baseline \
  --rep 1 \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --max-active-shards 4 \
  --gpu-ids 0,1,2,3 \
  --overwrite
```

Use the same `run-condition`, `retry`, `merge-dir`, and `extract_results.py` sequence as the Kubernetes run, replacing `run_k8s.py` with `run_local.py`.

## Interpretation Boundary

This experiment supports the claim that AnonLib recovery is shard-scoped and preserves completed shard work. It does not show that AnonLib has a native Kubernetes backend or Kubernetes-level fault tolerance.

Raw pod logs and pod JSON are retained under:

```text
$ANONLIB_RECOVERY_ROOT/runs/<condition>/rep_<NN>/raw_logs/<phase>/
$ANONLIB_RECOVERY_ROOT/runs/<condition>/rep_<NN>/controller/
```
