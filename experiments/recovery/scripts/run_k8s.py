#!/usr/bin/env python3
"""Kubernetes shard-recovery controller run from the MMIRAGE container terminal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONTAINER_REPO = Path("/workspace/MMIRAGE")
DEFAULT_SHARED_ROOT = "/workspace/mmirage-recovery"
sys.path.insert(0, str(REPO_ROOT / "src"))

from mmirage.cli_utils.status import check_failed_shards, status_exit_code  # noqa: E402
from mmirage.config.utils import load_mmirage_config  # noqa: E402
from mmirage.shard_utils import read_status, shard_state_dir  # noqa: E402

CONDITION_FAILURE_SHARDS = {
    "baseline": [],
    "fail_1": [3],
    "fail_4": [1, 5, 9, 13],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--namespace", required=True)
        p.add_argument("--pvc", required=True)
        p.add_argument("--image", required=True)
        p.add_argument("--shared-root", default=os.environ.get("MMIRAGE_RECOVERY_ROOT", DEFAULT_SHARED_ROOT))
        p.add_argument("--config", default=str(DEFAULT_CONTAINER_REPO / "experiments" / "recovery" / "configs" / "mmirage_recovery.yaml"))
        p.add_argument("--config-in-container", default=str(DEFAULT_CONTAINER_REPO / "experiments" / "recovery" / "configs" / "mmirage_recovery.yaml"))
        p.add_argument("--repo-dir-in-container", default=str(DEFAULT_CONTAINER_REPO))
        p.add_argument("--image-pull-policy", default="IfNotPresent")
        p.add_argument("--service-account", default=None)
        p.add_argument("--gpu-resource-name", default="nvidia.com/gpu")
        p.add_argument("--gpu-product-label", default=None)
        p.add_argument("--cpu-request", default="8")
        p.add_argument("--memory-request", default="64Gi")
        p.add_argument("--kubectl-context", default=None)
        p.add_argument("--max-active-shards", type=int, default=4)
        p.add_argument("--wait-timeout-seconds", type=int, default=21600)
        p.add_argument("--termination-grace-period-seconds", type=int, default=120)

    run_p = subparsers.add_parser("run-condition", help="Launch the initial clean or failure phase")
    add_common(run_p)
    run_p.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    run_p.add_argument("--rep", type=int, default=1)
    run_p.add_argument("--overwrite", action="store_true")
    run_p.add_argument("--kill-after-seconds", type=float, default=None)
    run_p.add_argument("--baseline-rep", type=int, default=1)

    retry_p = subparsers.add_parser("retry", help="Relaunch only shards MMIRAGE marks incomplete")
    add_common(retry_p)
    retry_p.add_argument("--condition", choices=[c for c in CONDITION_FAILURE_SHARDS if c != "baseline"], required=True)
    retry_p.add_argument("--rep", type=int, default=1)
    retry_p.add_argument("--max-rounds", type=int, default=3)

    status_p = subparsers.add_parser("status", help="Print MMIRAGE shard status for one run")
    status_p.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    status_p.add_argument("--rep", type=int, default=1)
    status_p.add_argument("--shared-root", default=os.environ.get("MMIRAGE_RECOVERY_ROOT", DEFAULT_SHARED_ROOT))
    status_p.add_argument("--config", default=str(DEFAULT_CONTAINER_REPO / "experiments" / "recovery" / "configs" / "mmirage_recovery.yaml"))

    return parser.parse_args()


def require_container_terminal(args: argparse.Namespace) -> None:
    if not Path(args.shared_root).is_absolute():
        raise RuntimeError(
            f"--shared-root must be an absolute path inside the container, got {args.shared_root!r}. "
            f"Recommended: {DEFAULT_SHARED_ROOT}"
        )
    config = getattr(args, "config", None)
    if config and not Path(config).exists():
        raise RuntimeError(
            f"MMIRAGE config not found at {config}. Run this from the MMIRAGE container terminal "
            f"with the repository available at {DEFAULT_CONTAINER_REPO}, or pass --config explicitly."
        )
    if shutil.which("kubectl") is None:
        raise RuntimeError(
            "kubectl is not available in this container terminal. Install it in the image or run from an image that includes it."
        )
    if getattr(args, "max_active_shards", 1) < 1:
        raise RuntimeError("--max-active-shards must be at least 1")


def run_dir(shared_root: str, condition: str, rep: int) -> Path:
    return Path(shared_root) / "runs" / condition / f"rep_{rep:02d}"


def runtime_env(shared_root: str, condition: str, rep: int) -> Dict[str, str]:
    rd = run_dir(shared_root, condition, rep)
    return {
        "MMIRAGE_RECOVERY_ROOT": shared_root,
        "MMIRAGE_RECOVERY_RUN_DIR": str(rd),
        "MMIRAGE_RECOVERY_INPUT_JSONL": str(Path(shared_root) / "data" / "ultrachat_200k" / "subset.jsonl"),
        "MMIRAGE_RECOVERY_STATE_DIR": str(rd / "state"),
        "MMIRAGE_RECOVERY_OUTPUT_DIR": str(rd / "output"),
        "HF_HOME": str(Path(shared_root) / "hf"),
    }


def load_cfg(config: str, env: Dict[str, str]) -> Any:
    old = os.environ.copy()
    os.environ.update(env)
    try:
        return load_mmirage_config(config)
    finally:
        os.environ.clear()
        os.environ.update(old)


def kubectl_base(args: argparse.Namespace) -> List[str]:
    command = ["kubectl"]
    context = getattr(args, "kubectl_context", None)
    if context:
        command.extend(["--context", context])
    return command


def kubectl(args: argparse.Namespace, extra: Sequence[str], *, check: bool = True, input_text: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*kubectl_base(args), *extra],
        input=input_text,
        text=True,
        capture_output=True,
        check=check,
    )


def run_label(condition: str, rep: int) -> str:
    return f"{condition.replace('_', '-')}-r{rep:02d}"


def pod_name(condition: str, rep: int, phase: str, shard: int) -> str:
    return f"mmirage-rec-{condition.replace('_', '-')}-r{rep:02d}-{phase}-s{shard}"


def pod_manifest(args: argparse.Namespace, condition: str, rep: int, phase: str, shard: int) -> Dict[str, Any]:
    rd = run_dir(args.shared_root, condition, rep)
    labels = {
        "app.kubernetes.io/name": "mmirage-shard-recovery",
        "mmirage.run": run_label(condition, rep),
        "mmirage.condition": condition,
        "mmirage.phase": phase,
        "mmirage.shard-id": str(shard),
    }
    container = {
        "name": "shard",
        "image": args.image,
        "imagePullPolicy": args.image_pull_policy,
        "workingDir": args.repo_dir_in_container,
        "command": ["python"],
        "args": [
            f"{args.repo_dir_in_container}/experiments/recovery/scripts/run_pod.py",
            "--config",
            args.config_in_container,
            "--shard-id",
            str(shard),
        ],
        "env": [
            {"name": "SLURM_ARRAY_TASK_ID", "value": str(shard)},
            {"name": "MMIRAGE_COLLECT_STATS", "value": "1"},
            {"name": "MMIRAGE_RECOVERY_ROOT", "value": args.shared_root},
            {"name": "MMIRAGE_RECOVERY_RUN_DIR", "value": str(rd)},
            {"name": "MMIRAGE_RECOVERY_INPUT_JSONL", "value": str(Path(args.shared_root) / "data" / "ultrachat_200k" / "subset.jsonl")},
            {"name": "MMIRAGE_RECOVERY_STATE_DIR", "value": str(rd / "state")},
            {"name": "MMIRAGE_RECOVERY_OUTPUT_DIR", "value": str(rd / "output")},
            {"name": "HF_HOME", "value": str(Path(args.shared_root) / "hf")},
            {"name": "TRANSFORMERS_CACHE", "value": str(Path(args.shared_root) / "hf" / "transformers")},
        ],
        "resources": {
            "requests": {"cpu": args.cpu_request, "memory": args.memory_request, args.gpu_resource_name: "1"},
            "limits": {args.gpu_resource_name: "1"},
        },
        "volumeMounts": [{"name": "shared", "mountPath": args.shared_root}],
    }
    spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": args.termination_grace_period_seconds,
        "containers": [container],
        "volumes": [{"name": "shared", "persistentVolumeClaim": {"claimName": args.pvc}}],
    }
    if args.service_account:
        spec["serviceAccountName"] = args.service_account
    if args.gpu_product_label:
        spec["nodeSelector"] = {"nvidia.com/gpu.product": args.gpu_product_label}
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name(condition, rep, phase, shard), "namespace": args.namespace, "labels": labels},
        "spec": spec,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def apply_pods(args: argparse.Namespace, condition: str, rep: int, phase: str, shards: Sequence[int]) -> Path:
    manifest = {"apiVersion": "v1", "kind": "List", "items": [pod_manifest(args, condition, rep, phase, shard) for shard in shards]}
    manifest_path = run_dir(args.shared_root, condition, rep) / "controller" / f"{phase}_pods.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    kubectl(args, ["apply", "-f", str(manifest_path)])
    return manifest_path


def pod_json(args: argparse.Namespace, condition: str, rep: int, phase: str) -> Dict[str, Any]:
    selector = f"mmirage.run={run_label(condition, rep)},mmirage.phase={phase}"
    result = kubectl(args, ["get", "pods", "-n", args.namespace, "-l", selector, "-o", "json"], check=True)
    return json.loads(result.stdout)


def pod_phase(item: Dict[str, Any]) -> str:
    return str(item.get("status", {}).get("phase", "Unknown"))


def terminal_phase(phase: str) -> bool:
    return phase in {"Succeeded", "Failed"}


def wait_for_pods(args: argparse.Namespace, condition: str, rep: int, phase: str, expected: int) -> Dict[str, Any]:
    deadline = time.monotonic() + args.wait_timeout_seconds
    last_payload: Dict[str, Any] = {"items": []}
    while time.monotonic() < deadline:
        last_payload = pod_json(args, condition, rep, phase)
        items = last_payload.get("items", [])
        if len(items) >= expected and all(terminal_phase(pod_phase(item)) for item in items):
            return last_payload
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {expected} pod(s) in phase {phase}")


def wait_until_pods_running(args: argparse.Namespace, condition: str, rep: int, phase: str, shards: Sequence[int]) -> None:
    wanted = {pod_name(condition, rep, phase, shard) for shard in shards}
    deadline = time.monotonic() + args.wait_timeout_seconds
    while time.monotonic() < deadline:
        payload = pod_json(args, condition, rep, phase)
        running = {item["metadata"]["name"] for item in payload.get("items", []) if pod_phase(item) == "Running"}
        finished = {item["metadata"]["name"] for item in payload.get("items", []) if terminal_phase(pod_phase(item))}
        if wanted <= (running | finished):
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for selected pods to start: {sorted(wanted)}")


def kill_pods(args: argparse.Namespace, condition: str, rep: int, phase: str, shards: Sequence[int]) -> List[Dict[str, Any]]:
    events = []
    for shard in shards:
        name = pod_name(condition, rep, phase, shard)
        event = {"shard_id": shard, "pod": name, "requested_at": utc_now(), "method": "exec kill -TERM 1"}
        result = kubectl(args, ["exec", "-n", args.namespace, name, "--", "/bin/sh", "-c", "kill -TERM 1"], check=False)
        event["returncode"] = result.returncode
        event["stdout"] = result.stdout
        event["stderr"] = result.stderr
        if result.returncode != 0:
            events.append(event)
            raise RuntimeError(
                f"Failed to signal pod {name}. Refusing to delete the pod object because raw logs and terminal status must be preserved. stderr={result.stderr!r}"
            )
        events.append(event)
    return events


def parse_kubernetes_time(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def pod_runtime_seconds(item: Dict[str, Any]) -> Optional[float]:
    statuses = item.get("status", {}).get("containerStatuses") or []
    for status in statuses:
        state = status.get("state", {})
        terminated = state.get("terminated")
        if terminated:
            started = parse_kubernetes_time(terminated.get("startedAt"))
            finished = parse_kubernetes_time(terminated.get("finishedAt"))
            if started is not None and finished is not None:
                return max(0.0, round(finished - started, 3))
    return None


def collect_logs(args: argparse.Namespace, condition: str, rep: int, phase: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_dir = run_dir(args.shared_root, condition, rep) / "raw_logs" / phase
    raw_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    for item in payload.get("items", []):
        name = item["metadata"]["name"]
        logs = kubectl(args, ["logs", "-n", args.namespace, name, "--timestamps"], check=False)
        (raw_dir / f"{name}.log").write_text(logs.stdout + logs.stderr, encoding="utf-8", errors="replace")
        pod_dump = kubectl(args, ["get", "pod", "-n", args.namespace, name, "-o", "json"], check=False)
        (raw_dir / f"{name}.pod.json").write_text(pod_dump.stdout + pod_dump.stderr, encoding="utf-8", errors="replace")
        collected.append({"pod": name, "phase": pod_phase(item), "runtime_seconds": pod_runtime_seconds(item), "log_path": str(raw_dir / f"{name}.log")})
    return collected


def chunked(items: Sequence[int], size: int) -> Iterable[List[int]]:
    for offset in range(0, len(items), size):
        yield list(items[offset : offset + size])


def dir_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    digest = hashlib_sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8") + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def hashlib_sha256() -> Any:
    import hashlib

    return hashlib.sha256()


def snapshot_completed(args: argparse.Namespace, condition: str, rep: int, cfg: Any, label: str) -> None:
    rd = run_dir(args.shared_root, condition, rep)
    state_root = cfg.loading_params.get_state_root()
    output_root = Path(os.environ["MMIRAGE_RECOVERY_OUTPUT_DIR"])
    rows = []
    for shard_id in range(cfg.loading_params.get_num_shards()):
        status = read_status(shard_state_dir(shard_id, state_root))
        if status.status != "success":
            continue
        rows.append({"shard_id": shard_id, "output_dir": str(output_root / f"shard_{shard_id}"), "sha256": dir_sha256(output_root / f"shard_{shard_id}")})
    write_json(rd / "controller" / f"completed_shards_{label}.json", {"created_at": utc_now(), "shards": rows})


def baseline_kill_after(shared_root: str, rep: int, explicit: Optional[float]) -> Dict[str, Any]:
    if explicit is not None:
        return {"seconds": explicit, "source": "explicit --kill-after-seconds"}
    state_root = run_dir(shared_root, "baseline", rep) / "state"
    runtimes = []
    for shard_id in range(16):
        status = read_status(shard_state_dir(shard_id, str(state_root)))
        if status.status == "success" and status.stats and status.stats.runtime_seconds:
            runtimes.append(float(status.stats.runtime_seconds))
    if runtimes:
        runtimes.sort()
        median = runtimes[len(runtimes) // 2]
        return {"seconds": max(30.0, round(median * 0.45, 3)), "source": "0.45 * median clean-run shard runtime"}
    return {"seconds": 120.0, "source": "fallback because clean-run stats were unavailable"}


def run_phase(args: argparse.Namespace, condition: str, rep: int, phase: str, shards: Sequence[int], fail_shards: Sequence[int], kill_after: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start = time.monotonic()
    started_at = utc_now()
    manifest_paths: List[str] = []
    kill_events: List[Dict[str, Any]] = []
    pod_records: List[Dict[str, Any]] = []
    if fail_shards:
        assert kill_after is not None
    fail_set = set(fail_shards)
    for wave_index, wave_shards in enumerate(chunked(list(shards), args.max_active_shards), start=1):
        wave_phase = f"{phase}_w{wave_index:02d}"
        manifest_path = apply_pods(args, condition, rep, wave_phase, wave_shards)
        manifest_paths.append(str(manifest_path))
        wave_fail_shards = [shard for shard in wave_shards if shard in fail_set]
        if wave_fail_shards:
            wait_until_pods_running(args, condition, rep, wave_phase, wave_fail_shards)
            time.sleep(float(kill_after["seconds"]))
            kill_events.extend(kill_pods(args, condition, rep, wave_phase, wave_fail_shards))
        payload = wait_for_pods(args, condition, rep, wave_phase, len(wave_shards))
        pod_records.extend(collect_logs(args, condition, rep, wave_phase, payload))
    finished_at = utc_now()
    summary = {
        "condition": condition,
        "rep": rep,
        "phase": phase,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(time.monotonic() - start, 3),
        "max_active_shards": args.max_active_shards,
        "manifest_paths": manifest_paths,
        "launched_shards": list(shards),
        "failed_shards_requested": list(fail_shards),
        "kill_after": kill_after,
        "kill_events": kill_events,
        "pods": pod_records,
    }
    write_json(run_dir(args.shared_root, condition, rep) / "controller" / f"phase_{phase}.json", summary)
    return summary


def handle_run_condition(args: argparse.Namespace) -> int:
    require_container_terminal(args)
    rd = run_dir(args.shared_root, args.condition, args.rep)
    if rd.exists():
        if not args.overwrite:
            raise RuntimeError(f"Run directory already exists: {rd}. Use --overwrite for a new repetition directory only if you intend to replace it.")
        shutil.rmtree(rd)
    rd.mkdir(parents=True, exist_ok=True)

    env = runtime_env(args.shared_root, args.condition, args.rep)
    old = os.environ.copy()
    os.environ.update(env)
    try:
        cfg = load_mmirage_config(args.config)
        fail_shards = CONDITION_FAILURE_SHARDS[args.condition]
        kill_after = baseline_kill_after(args.shared_root, args.baseline_rep, args.kill_after_seconds) if fail_shards else None
        summary = run_phase(args, args.condition, args.rep, "initial", list(range(cfg.loading_params.get_num_shards())), fail_shards, kill_after)
        failed, status_summary = check_failed_shards(cfg)
        summary["mmirage_status_after_phase"] = status_summary.__dict__
        summary["mmirage_retryable_shards_after_phase"] = failed
        write_json(rd / "controller" / "phase_initial.json", summary)
        if fail_shards:
            snapshot_completed(args, args.condition, args.rep, cfg, "before_retry")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if args.condition == "baseline" and status_exit_code(failed, status_summary) == 0 else 0
    finally:
        os.environ.clear()
        os.environ.update(old)


def handle_retry(args: argparse.Namespace) -> int:
    require_container_terminal(args)
    env = runtime_env(args.shared_root, args.condition, args.rep)
    old = os.environ.copy()
    os.environ.update(env)
    try:
        cfg = load_mmirage_config(args.config)
        for round_idx in range(1, args.max_rounds + 1):
            failed, summary = check_failed_shards(cfg)
            if status_exit_code(failed, summary) == 0:
                snapshot_completed(args, args.condition, args.rep, cfg, "after_retry")
                print(json.dumps({"status": "already_complete", "summary": summary.__dict__}, indent=2, sort_keys=True))
                return 0
            if not failed:
                print(json.dumps({"status": "blocked", "summary": summary.__dict__}, indent=2, sort_keys=True))
                return 1
            phase = f"retry_{round_idx}"
            run_phase(args, args.condition, args.rep, phase, failed, [])
        failed, summary = check_failed_shards(cfg)
        snapshot_completed(args, args.condition, args.rep, cfg, "after_retry")
        result = {"summary": summary.__dict__, "retryable_shards": failed}
        print(json.dumps(result, indent=2, sort_keys=True))
        return status_exit_code(failed, summary)
    finally:
        os.environ.clear()
        os.environ.update(old)


def handle_status(args: argparse.Namespace) -> int:
    require_container_terminal(args)
    env = runtime_env(args.shared_root, args.condition, args.rep)
    cfg = load_cfg(args.config, env)
    failed, summary = check_failed_shards(cfg)
    print(json.dumps({"summary": summary.__dict__, "retryable_shards": failed}, indent=2, sort_keys=True))
    return status_exit_code(failed, summary)


def main() -> None:
    args = parse_args()
    if args.command == "run-condition":
        sys.exit(handle_run_condition(args))
    if args.command == "retry":
        sys.exit(handle_retry(args))
    if args.command == "status":
        sys.exit(handle_status(args))
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
