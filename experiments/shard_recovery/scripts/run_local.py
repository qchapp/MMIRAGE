#!/usr/bin/env python3
"""Local shard-recovery runner (no Kubernetes).

Runs the same ANONLIB shard-recovery experiment as
``run_k8s.py`` but from a single pod terminal without creating
Kubernetes pods. Each logical shard is a local subprocess that runs the same
pod entrypoint (``run_pod.py``); one GPU is pinned per
concurrent shard through ``CUDA_VISIBLE_DEVICES``. The deliberate pod
termination of the Kubernetes experiment is emulated by sending ``SIGTERM`` to
the selected wrapper processes, which the wrapper records as ANONLIB shard
failure exactly as it does for a terminated Kubernetes pod.

The controller/raw_logs directory layout written here matches
``run_k8s.py`` so that ``extract_results.py`` consumes the results unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_k8s import (  # noqa: E402
    CONDITION_FAILURE_SHARDS,
    baseline_kill_after,
    chunked,
    dir_sha256,
    load_cfg,
    pod_name,
    run_dir,
    run_label,
    runtime_env,
    snapshot_completed,
    utc_now,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ROOT = "/workspace/anonlib-recovery"
DEFAULT_GPU_IDS = "0,1,2,3"

sys.path.insert(0, str(REPO_ROOT / "src"))

from anonlib.cli_utils.status import check_failed_shards, status_exit_code  # noqa: E402
from anonlib.config.utils import load_anonlib_config  # noqa: E402
from anonlib.shard_utils import read_status, shard_state_dir  # noqa: E402


POD_ENTRYPOINT = EXPERIMENT_DIR / "scripts" / "run_pod.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--shared-root", default=os.environ.get("ANONLIB_RECOVERY_ROOT", DEFAULT_SHARED_ROOT))
        p.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "anonlib_recovery.yaml"))
        p.add_argument("--max-active-shards", type=int, default=4)
        p.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS)
        p.add_argument("--wait-timeout-seconds", type=int, default=21600)
        p.add_argument("--kill-after-seconds", type=float, default=None)

    run_p = subparsers.add_parser("run-condition", help="Launch the initial clean or failure phase")
    add_common(run_p)
    run_p.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    run_p.add_argument("--rep", type=int, default=1)
    run_p.add_argument("--overwrite", action="store_true")
    run_p.add_argument("--baseline-rep", type=int, default=1)

    retry_p = subparsers.add_parser("retry", help="Relaunch only shards ANONLIB marks incomplete")
    add_common(retry_p)
    retry_p.add_argument("--condition", choices=["fail_1", "fail_4", "fail_8"], required=True)
    retry_p.add_argument("--rep", type=int, default=1)
    retry_p.add_argument("--max-rounds", type=int, default=3)

    status_p = subparsers.add_parser("status", help="Print ANONLIB shard status for one run")
    status_p.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    status_p.add_argument("--rep", type=int, default=1)
    status_p.add_argument("--shared-root", default=os.environ.get("ANONLIB_RECOVERY_ROOT", DEFAULT_SHARED_ROOT))
    status_p.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "anonlib_recovery.yaml"))

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
            f"ANONLIB config not found at {config}. Run from the ANONLIB container terminal "
            f"with the repository available, or pass --config explicitly."
        )
    if getattr(args, "max_active_shards", 1) < 1:
        raise RuntimeError("--max-active-shards must be at least 1")
    gpu_ids = parse_gpu_ids(getattr(args, "gpu_ids", None))
    if not gpu_ids:
        raise RuntimeError("--gpu-ids must contain at least one GPU id")


def parse_gpu_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["0"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def shard_env(
    shared_root: str, condition: str, rep: int, gpu_id: str
) -> Dict[str, str]:
    env = runtime_env(shared_root, condition, rep)
    env.update(
        {
            "HF_HOME": str(Path(shared_root) / "hf"),
            "TRANSFORMERS_CACHE": str(Path(shared_root) / "hf" / "transformers"),
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def launch_shard(args: argparse.Namespace, condition: str, rep: int, wave_phase: str, shard: int, gpu_id: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(shard_env(args.shared_root, condition, rep, gpu_id))
    command = [
        sys.executable,
        str(POD_ENTRYPOINT),
        "--config",
        str(args.config),
        "--shard-id",
        str(shard),
    ]
    raw_dir = run_dir(args.shared_root, condition, rep) / "raw_logs" / wave_phase
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{pod_name(condition, rep, wave_phase, shard)}.log"
    stdout_handle = log_path.open("ab")
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def wait_for_pod_running(
    state_root: str,
    shard: int,
    previous_started_at: Optional[str],
    proc: subprocess.Popen[str],
    timeout_seconds: int,
) -> None:
    """Wait until ANONLIB marks the shard running (pod Running equivalent)."""
    state_dir = shard_state_dir(shard, state_root)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        status = read_status(state_dir)
        if status.status == "running" and status.started_at != previous_started_at:
            return
        time.sleep(2)


def collect_wave_logs(
    args: argparse.Namespace,
    condition: str,
    rep: int,
    wave_phase: str,
    shards: Sequence[int],
    procs: Dict[int, subprocess.Popen[str]],
    launched_at: Dict[int, float],
    gpu_by_shard: Dict[int, str],
) -> List[Dict[str, Any]]:
    raw_dir = run_dir(args.shared_root, condition, rep) / "raw_logs" / wave_phase
    raw_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    for shard in shards:
        proc = procs[shard]
        return_code = proc.wait()
        name = pod_name(condition, rep, wave_phase, shard)
        log_path = raw_dir / f"{name}.log"
        with log_path.open("ab") as handle:
            handle.write(
                f"\n[controller] shard={shard} wave_phase={wave_phase} returncode={return_code} exit={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode("utf-8")
            )
        runtime = round(max(0.0, time.monotonic() - launched_at[shard]), 3)
        phase = "Succeeded" if return_code == 0 else "Failed"
        collected.append(
            {
                "pod": name,
                "shard_id": shard,
                "phase": phase,
                "runtime_seconds": runtime,
                "returncode": return_code,
                "gpu_id": gpu_by_shard.get(shard),
                "log_path": str(log_path),
            }
        )
    return collected


def kill_shards(
    args: argparse.Namespace,
    condition: str,
    rep: int,
    wave_phase: str,
    shards: Sequence[int],
    procs: Dict[int, subprocess.Popen[str]],
) -> List[Dict[str, Any]]:
    events = []
    for shard in shards:
        proc = procs[shard]
        name = pod_name(condition, rep, wave_phase, shard)
        event = {
            "shard_id": shard,
            "pod": name,
            "requested_at": utc_now(),
            "method": "SIGTERM to local shard wrapper",
        }
        try:
            proc.send_signal(signal.SIGTERM)
            event["returncode"] = 0
        except Exception as exc:  # pragma: no cover
            event["returncode"] = 1
            event["stderr"] = str(exc)
            raise RuntimeError(f"Failed to signal local shard {shard}: {exc}")
        events.append(event)
    return events


def run_phase(
    args: argparse.Namespace,
    condition: str,
    rep: int,
    phase: str,
    shards: Sequence[int],
    fail_shards: Sequence[int],
    kill_after: Optional[Dict[str, Any]] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    start = time.monotonic()
    started_at = utc_now()
    state_root = cfg.loading_params.get_state_root() if cfg is not None else run_dir(args.shared_root, condition, rep) / "state"
    kill_events: List[Dict[str, Any]] = []
    pod_records: List[Dict[str, Any]] = []
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    fail_set = set(fail_shards)
    if fail_shards:
        assert kill_after is not None

    previous_started_at: Dict[int, Optional[str]] = {}
    for shard in shards:
        previous_started_at[shard] = read_status(shard_state_dir(shard, str(state_root))).started_at

    for wave_index, wave_shards in enumerate(chunked(list(shards), args.max_active_shards), start=1):
        wave_phase = f"{phase}_w{wave_index:02d}"
        procs: Dict[int, subprocess.Popen[str]] = {}
        launched_at: Dict[int, float] = {}
        gpu_by_shard: Dict[int, str] = {}
        for offset, shard in enumerate(wave_shards):
            gpu_id = gpu_ids[offset % len(gpu_ids)]
            gpu_by_shard[shard] = gpu_id
            procs[shard] = launch_shard(args, condition, rep, wave_phase, shard, gpu_id)
            launched_at[shard] = time.monotonic()

        wave_fail_shards = [shard for shard in wave_shards if shard in fail_set]
        if wave_fail_shards:
            for shard in wave_fail_shards:
                wait_for_pod_running(
                    str(state_root), shard, previous_started_at.get(shard), procs[shard], args.wait_timeout_seconds
                )
            time.sleep(float(kill_after["seconds"]))
            kill_events.extend(kill_shards(args, condition, rep, wave_phase, wave_fail_shards, procs))

        pod_records.extend(
            collect_wave_logs(args, condition, rep, wave_phase, wave_shards, procs, launched_at, gpu_by_shard)
        )

    summary = {
        "condition": condition,
        "rep": rep,
        "phase": phase,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": round(time.monotonic() - start, 3),
        "max_active_shards": args.max_active_shards,
        "gpu_ids": gpu_ids,
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
            raise RuntimeError(f"Run directory already exists: {rd}. Use --overwrite to replace it.")
        shutil.rmtree(rd)
    rd.mkdir(parents=True, exist_ok=True)

    env = runtime_env(args.shared_root, args.condition, args.rep)
    old = os.environ.copy()
    os.environ.update(env)
    try:
        cfg = load_anonlib_config(args.config)
        fail_shards = CONDITION_FAILURE_SHARDS[args.condition]
        kill_after = baseline_kill_after(args.shared_root, args.baseline_rep, args.kill_after_seconds) if fail_shards else None
        summary = run_phase(
            args, args.condition, args.rep, "initial", list(range(cfg.loading_params.get_num_shards())), fail_shards, kill_after, cfg=cfg
        )
        failed, status_summary = check_failed_shards(cfg)
        summary["anonlib_status_after_phase"] = status_summary.__dict__
        summary["anonlib_retryable_shards_after_phase"] = failed
        write_json(rd / "controller" / "phase_initial.json", summary)
        if fail_shards:
            snapshot_completed(args, args.condition, args.rep, cfg, "before_retry")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        os.environ.clear()
        os.environ.update(old)


def handle_retry(args: argparse.Namespace) -> int:
    require_container_terminal(args)
    env = runtime_env(args.shared_root, args.condition, args.rep)
    old = os.environ.copy()
    os.environ.update(env)
    try:
        cfg = load_anonlib_config(args.config)
        rd = run_dir(args.shared_root, args.condition, args.rep)
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
            run_phase(args, args.condition, args.rep, phase, failed, [], cfg=cfg)
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
