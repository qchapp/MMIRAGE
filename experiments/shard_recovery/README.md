# AnonLib Shard Recovery on Kubernetes

This experiment quantifies how much work AnonLib avoids recomputing when Kubernetes terminates only a subset of shard pods. Kubernetes is used only as external orchestration. AnonLib still runs its existing local shard execution path inside each pod and uses its existing shard state, retry-count, statistics, and merge mechanisms.

## Paper Context

The current paper claims AnonLib provides declarative transformation, portable inference/execution settings, independent sharded execution, deterministic merging, shard-scoped retry, and runtime statistics. Its Results section already reports a 16-node MedTrinity run, an observed recovery event where 5 of 32 failed shards were relaunched without rerunning 27 completed shards, and a separate one-GPU Qwen3-4B DataTrove-style benchmark. It explicitly does not yet provide controlled recovery-overhead or recomputed-work measurements.

The semester report contains the historical implementation details for shard-level `status.json`, bounded retry, merge commands, and runtime statistics. It also repeats the observed 5-of-32 MedTrinity recovery event.

Recorded discrepancy: the current paper resolves the distributed MedTrinity allocation as 16 nodes x 1 GPU. The semester report appendix says the 16-node Slurm run used 16 nodes with 4 GH200 GPUs per node, and the report body mentions 16 nodes/64 GPUs for scale-out. This experiment is separate and should report the actual H100 Kubernetes allocation used for each run.

## What This Supports

The experiment supports the claim that AnonLib recovery is shard-scoped and preserves completed shard work. It does not support a claim that AnonLib has a native Kubernetes backend or Kubernetes-level fault tolerance.

## Files

- `configs/anonlib_recovery.yaml`: fixed 16-shard AnonLib workload using `Qwen/Qwen3-4B`, SGLang, one tensor-parallel GPU, deterministic temperature `0.0`, and stats enabled by the pod wrapper.
- `scripts/prepare_workload.py`: downloads and freezes a public Hugging Face subset.
- `scripts/run_k8s.py`: creates one Kubernetes pod per shard, terminates deterministic selected pods, and retries only shards selected from AnonLib state.
- `scripts/run_local.py`: optional single-terminal local runner with the same state layout, for clusters where Kubernetes pod orchestration cannot be used.
- `scripts/run_pod.py`: pod/local entrypoint that runs `anonlib.shard_process` and records deliberate termination as shard failure in AnonLib state.
- `scripts/extract_results.py`: writes `recovery_results.csv` and `recovery_results.json`.

## Dataset

Default dataset: `HuggingFaceH4/ultrachat_200k`, split `train_sft`.

The preparation script records the Hugging Face dataset commit if available, deterministically shuffles with seed `20260813`, selects `65536` records by default, and writes both the JSONL workload and an ID-order manifest. The selected subset gives 4096 records per shard for the fixed 16-shard workload. If this dataset becomes unavailable or its schema changes incompatibly, use another public text dataset and record the reason in `manifest.json`; do not silently change the dataset in reported results.

## Prerequisites

- An interactive Kubernetes or Run:ai job terminal inside the AnonLib container image. All commands below are intended to be run from that terminal, not from the host workstation.
- The container should contain this repository at `/workspace/ANONLIB`, or you must pass matching `--config`, `--config-in-container`, and `--repo-dir-in-container` paths.
- The container image must include AnonLib GPU dependencies, including `anonlib[gpu]`, `sglang==0.5.10`, CUDA-compatible PyTorch, `datasets`, and `kubectl`.
- A Kubernetes namespace and a shared PVC mounted read/write by all shard pods.
- Access to NVIDIA H100 nodes through your Kubernetes or Run:ai environment.
- Your claimed interactive job should have the same shared PVC mounted at `/workspace/anonlib-recovery`, or set `ANONLIB_RECOVERY_ROOT` to the mounted absolute path.

The generic GPU resource request is `nvidia.com/gpu: 1`. The controller defaults to `--max-active-shards 4`, matching the intended four-H100 experiment budget by running at most four one-GPU shard pods at a time. It still evaluates a fixed 16-shard workload by launching shards in four waves. If your Run:ai cluster uses extra queues, priorities, tolerations, or node labels, add them to rendered manifests or extend controller flags locally, but do not describe those as AnonLib features.

Start from the repository root inside the container:

```bash
cd /workspace/ANONLIB
```

## 1. Prepare PVC

Provision a shared `ReadWriteMany` PersistentVolumeClaim mounted at the shared root and writable by all shard pods. The exact steps are cluster-specific and are not part of the experiment controller.

## 2. Prepare Data

Run this from the interactive AnonLib container terminal. The output root must be an absolute path mounted in the container and shard pods:

```bash
export ANONLIB_RECOVERY_ROOT=/workspace/anonlib-recovery
python experiments/shard_recovery/scripts/prepare_workload.py \
  --output-root "$ANONLIB_RECOVERY_ROOT" \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --num-records 65536 \
  --seed 20260813
```

The script writes:

- `$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/subset.jsonl`
- `$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/id_order.jsonl`
- `$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/manifest.json`

## 3. Common Variables

Set these once for all controller commands:

```bash
export NAMESPACE=anonlib-recovery
export PVC_NAME=anonlib-recovery-pvc
export IMAGE=<your-anonlib-gpu-image>
export ANONLIB_RECOVERY_ROOT=/workspace/anonlib-recovery
export MAX_ACTIVE_SHARDS=4
```

Optional H100 node label example, if your cluster uses this label:

```bash
export GPU_PRODUCT_LABEL=NVIDIA-H100
```

If your cluster does not use `nvidia.com/gpu.product`, omit `--gpu-product-label`. Do not pass an empty value.

## 4. Clean Baseline

Launch the clean 0-failure condition first. Failure timings for later conditions are derived from this run when available.

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition baseline \
  --rep 1 \
  --namespace "$NAMESPACE" \
  --pvc "$PVC_NAME" \
  --image "$IMAGE" \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --max-active-shards "$MAX_ACTIVE_SHARDS" \
  --overwrite
```

If your cluster requires an H100 node selector, add `--gpu-product-label "$GPU_PRODUCT_LABEL"` to the controller commands.

Merge the baseline outputs:

```bash
anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/baseline/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/baseline/rep_01/merged"
```

## 5. Failure Conditions

The deterministic failed shard sets are:

- `fail_1`: shard `3`
- `fail_4`: shards `1,5,9,13`
- `fail_8`: shards `0,2,4,6,8,10,12,14`

The controller waits until the selected pods are running, sleeps for about 45 percent of the median clean-run shard runtime, then uses `kubectl exec` to send `SIGTERM` to PID 1 in those pods. This deliberately terminates the selected Kubernetes pod workload without corrupting model outputs. The wrapper records this as AnonLib shard failure so the normal shard-state retry mechanism can identify incomplete shards. If clean-run stats are unavailable, the fallback is 120 seconds; override with `--kill-after-seconds` if needed.

The controller intentionally refuses to fall back to deleting the pod object if `kubectl exec` cannot deliver the signal, because the experiment must preserve raw logs and terminal pod status. Fix pod exec permissions or use a cluster-approved equivalent that preserves logs and AnonLib state.

Launch `fail_1` initial phase:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition \
  --condition fail_1 \
  --rep 1 \
  --namespace "$NAMESPACE" \
  --pvc "$PVC_NAME" \
  --image "$IMAGE" \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --max-active-shards "$MAX_ACTIVE_SHARDS" \
  --overwrite
```

Retry only AnonLib-incomplete shards:

```bash
python experiments/shard_recovery/scripts/run_k8s.py retry \
  --condition fail_1 \
  --rep 1 \
  --namespace "$NAMESPACE" \
  --pvc "$PVC_NAME" \
  --image "$IMAGE" \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --max-active-shards "$MAX_ACTIVE_SHARDS"
```

Merge after retry:

```bash
anonlib merge-dir \
  --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_1/rep_01/output" \
  --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_1/rep_01/merged"
```

Repeat the same pattern for `fail_4` and optionally `fail_8`:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition --condition fail_4 --rep 1 --namespace "$NAMESPACE" --pvc "$PVC_NAME" --image "$IMAGE" --shared-root "$ANONLIB_RECOVERY_ROOT" --max-active-shards "$MAX_ACTIVE_SHARDS" --overwrite
python experiments/shard_recovery/scripts/run_k8s.py retry --condition fail_4 --rep 1 --namespace "$NAMESPACE" --pvc "$PVC_NAME" --image "$IMAGE" --shared-root "$ANONLIB_RECOVERY_ROOT" --max-active-shards "$MAX_ACTIVE_SHARDS"
anonlib merge-dir --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/output" --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/merged"
```

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition --condition fail_8 --rep 1 --namespace "$NAMESPACE" --pvc "$PVC_NAME" --image "$IMAGE" --shared-root "$ANONLIB_RECOVERY_ROOT" --max-active-shards "$MAX_ACTIVE_SHARDS" --overwrite
python experiments/shard_recovery/scripts/run_k8s.py retry --condition fail_8 --rep 1 --namespace "$NAMESPACE" --pvc "$PVC_NAME" --image "$IMAGE" --shared-root "$ANONLIB_RECOVERY_ROOT" --max-active-shards "$MAX_ACTIVE_SHARDS"
anonlib merge-dir --input-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_8/rep_01/output" --output-dir "$ANONLIB_RECOVERY_ROOT/runs/fail_8/rep_01/merged"
```

## 6. Status and Stats

Check AnonLib shard status for any run:

```bash
python experiments/shard_recovery/scripts/run_k8s.py status \
  --condition fail_4 \
  --rep 1 \
  --shared-root "$ANONLIB_RECOVERY_ROOT"
```

Collect AnonLib stats directly:

```bash
ANONLIB_RECOVERY_INPUT_JSONL="$ANONLIB_RECOVERY_ROOT/data/ultrachat_200k/subset.jsonl" \
ANONLIB_RECOVERY_STATE_DIR="$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/state" \
ANONLIB_RECOVERY_OUTPUT_DIR="$ANONLIB_RECOVERY_ROOT/runs/fail_4/rep_01/output" \
anonlib stats --config experiments/shard_recovery/configs/anonlib_recovery.yaml
```

## 7. Extract Results

After merging all completed conditions:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4,fail_8 \
  --reps 1
```

Outputs:

- `$ANONLIB_RECOVERY_ROOT/results/recovery_results.csv`
- `$ANONLIB_RECOVERY_ROOT/results/recovery_results.json`

The extractor reports exact row recomputation and the closest defensible token recomputation quantity currently available: output tokens generated by the successful retry shards. AnonLib does not record token progress inside a killed pod, so exact token-level lost work before termination is not invented.

## Repetitions

Do not report the best run. To repeat a condition, increment `--rep` and retain every repetition:

```bash
python experiments/shard_recovery/scripts/run_k8s.py run-condition --condition baseline --rep 2 --namespace "$NAMESPACE" --pvc "$PVC_NAME" --image "$IMAGE" --shared-root "$ANONLIB_RECOVERY_ROOT" --max-active-shards "$MAX_ACTIVE_SHARDS" --overwrite
```

Then extract multiple repetitions:

```bash
python experiments/shard_recovery/scripts/extract_results.py \
  --shared-root "$ANONLIB_RECOVERY_ROOT" \
  --conditions baseline,fail_1,fail_4 \
  --reps 1,2,3
```

The JSON output includes individual runs and mean +/- standard deviation summaries.

## Raw Logs

The controller preserves raw pod logs and pod JSON under each run directory:

```text
$ANONLIB_RECOVERY_ROOT/runs/<condition>/rep_<NN>/raw_logs/<phase>/
$ANONLIB_RECOVERY_ROOT/runs/<condition>/rep_<NN>/controller/
```

Retain these directories with the CSV/JSON results. They are required to audit pod termination time, shard status transitions, retry counts, and final merge integrity.
