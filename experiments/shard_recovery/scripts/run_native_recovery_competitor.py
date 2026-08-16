#!/usr/bin/env python3
"""Local native competitor shard-recovery controller (no Kubernetes).

Equivalent of ``run_local.py`` for the native framework baselines. Logical
shards are local worker subprocesses pinned to GPUs via ``CUDA_VISIBLE_DEVICES``;
the deliberate MMIRAGE pod termination is emulated with ``SIGTERM`` to the
designated failure shards after they report running. Shards without a valid
completion marker are relaunched in retry rounds, outputs are merged in the
expected id order, and the recovery output contract is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SHARED_DIR = REPO_ROOT / "experiments" / "_shared"
WORKER_SCRIPT = REPO_ROOT / "experiments" / "single_node_h100_scaling" / "scripts" / "native_shard_worker.py"

CONDITION_FAILURE_SHARDS = {
    "baseline": [],
    "fail_1": [3],
    "fail_4": [1, 5, 9, 13],
}
FRAMEWORKS = {"datatrove", "nemo_curator", "distilabel", "ray_data_llm", "raw_sglang"}
TOTAL_SHARDS = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunked(items: Sequence[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=sorted(FRAMEWORKS), required=True)
    parser.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--shared-root", default="/workspace/mmirage-recovery")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-active-shards", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--kill-after-seconds", type=float, default=20.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unimplemented", action="store_true")
    return parser.parse_args()


def output_contract() -> Dict[str, Any]:
    return {
        "final_row_order": "must_match_expected_id_order",
        "required_fields": ["stable_id", "prompt_text", "answer"],
        "validation": [
            "no_missing_ids",
            "no_duplicate_ids",
            "no_unexpected_ids",
            "completed_shard_outputs_unchanged_after_retry",
            "retry_only_incomplete_or_killed_shards",
        ],
    }


def run_root(args: argparse.Namespace) -> Path:
    return Path(args.shared_root) / "native_competitors" / args.framework / args.condition / f"rep_{args.rep:02d}"


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    if args.max_active_shards < 1:
        raise ValueError("--max-active-shards must be at least 1")
    return {
        "created_at": utc_now(),
        "framework": args.framework,
        "native_mode": True,
        "condition": args.condition,
        "rep": args.rep,
        "shared_root": args.shared_root,
        "gpu_ids": args.gpu_ids,
        "total_shards": TOTAL_SHARDS,
        "max_active_shards": args.max_active_shards,
        "initially_failed_shards": CONDITION_FAILURE_SHARDS[args.condition],
        "retry_policy": "relaunch shards without valid completion marker",
        "model": args.model,
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "kill_method": "SIGTERM to worker process group",
        "kill_after_seconds": args.kill_after_seconds,
        "max_retry_rounds": args.max_rounds,
        "output_contract": output_contract(),
    }


def write_shard_inputs(args: argparse.Namespace, rows: List[Dict[str, Any]], shards: Sequence[int], num_shards: int) -> List[Path]:
    paths = []
    for shard_id in shards:
        start = (len(rows) * shard_id) // num_shards
        end = (len(rows) * (shard_id + 1)) // num_shards
        input_path = args.state_dir / f"shard_{shard_id}" / "input.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("w", encoding="utf-8") as handle:
            for row in rows[start:end]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(input_path)
    return paths


def shard_complete(args: argparse.Namespace, shard_id: int, expected_ids: List[str]) -> bool:
    status = read_json(args.state_dir / f"shard_{shard_id}" / "status.json")
    output_path = args.state_dir / f"shard_{shard_id}" / "output.jsonl"
    if status.get("status") != "success":
        return False
    if not output_path.exists():
        return False
    output_ids = [str(row.get("stable_id")) for row in read_jsonl(output_path)]
    return len(output_ids) == len(expected_ids) and set(output_ids) == set(expected_ids)


def launch_worker(
    args: argparse.Namespace,
    shard_id: int,
    gpu_id: str,
    phase: str,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "local"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{SHARED_DIR}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else f"{REPO_ROOT}{os.pathsep}{SHARED_DIR}"
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_HOME"] = str(Path(args.shared_root) / "hf")
    env["TRANSFORMERS_CACHE"] = str(Path(args.shared_root) / "hf" / "transformers")
    env["PYTHONUNBUFFERED"] = "1"
    state_dir = args.state_dir
    command = [
        args.python,
        str(WORKER_SCRIPT),
        "--framework",
        args.framework,
        "--input-jsonl",
        str(state_dir / f"shard_{shard_id}" / "input.jsonl"),
        "--output-jsonl",
        str(state_dir / f"shard_{shard_id}" / "output.jsonl"),
        "--status-json",
        str(state_dir / f"shard_{shard_id}" / "status.json"),
        "--model",
        args.model,
        "--temperature",
        "0.0",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--concurrency",
        str(args.concurrency),
        "--prompt-style",
        "rewrite",
        "--id-field",
        "stable_id",
        "--gpu-id",
        str(gpu_id),
    ]
    log_dir = args.run_root / "raw_logs" / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / f"native-{args.condition}-r{args.rep:02d}-{phase}-s{shard_id}.log").open("ab")
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return proc


def wait_for_running(args: argparse.Namespace, shard_id: int, proc: subprocess.Popen[str], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    running_path = args.state_dir / f"shard_{shard_id}" / "running.json"
    while time.monotonic() < deadline:
        if proc.poll() is not None or running_path.exists():
            return
        time.sleep(1)


def kill_worker(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def run_phase(
    args: argparse.Namespace,
    phase: str,
    shards: Sequence[int],
    fail_shards: Sequence[int],
    expected_by_shard: Dict[int, List[str]],
    wait_timeout: int = 21600,
) -> Dict[str, Any]:
    started = time.monotonic()
    gpu_ids = args.gpu_ids
    fail_set = set(fail_shards)
    kill_events: List[Dict[str, Any]] = []
    pod_records: List[Dict[str, Any]] = []
    for wave_index, wave_shards in enumerate(chunked(list(shards), args.max_active_shards), start=1):
        wave_phase = f"{phase}_w{wave_index:02d}"
        procs: Dict[int, subprocess.Popen[str]] = {}
        gpu_by_shard: Dict[int, str] = {}
        launched_at: Dict[int, float] = {}
        for offset, shard_id in enumerate(wave_shards):
            gpu_id = gpu_ids[offset % len(gpu_ids)]
            gpu_by_shard[shard_id] = gpu_id
            procs[shard_id] = launch_worker(args, shard_id, gpu_id, wave_phase)
            launched_at[shard_id] = time.monotonic()
        wave_fail = [shard_id for shard_id in wave_shards if shard_id in fail_set]
        if wave_fail:
            for shard_id in wave_fail:
                wait_for_running(args, shard_id, procs[shard_id], args.kill_after_seconds * 4 + 120)
            time.sleep(args.kill_after_seconds)
            for shard_id in wave_fail:
                if procs[shard_id].poll() is None:
                    kill_worker(procs[shard_id])
                    kill_events.append({"shard_id": shard_id, "phase": wave_phase, "requested_at": utc_now(), "method": "SIGTERM to worker process group"})
        for shard_id in wave_shards:
            proc = procs[shard_id]
            returncode = proc.wait()
            runtime = round(max(0.0, time.monotonic() - launched_at[shard_id]), 3)
            complete = shard_complete(args, shard_id, expected_by_shard[shard_id])
            pod_records.append(
                {
                    "shard_id": shard_id,
                    "phase": wave_phase,
                    "phase_result": "Succeeded" if returncode == 0 else "Failed",
                    "completed": complete,
                    "returncode": returncode,
                    "runtime_seconds": runtime,
                    "gpu_id": gpu_by_shard[shard_id],
                }
            )
    wall = round(time.monotonic() - started, 3)
    summary = {
        "phase": phase,
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "wall_seconds": wall,
        "max_active_shards": args.max_active_shards,
        "launched_shards": list(shards),
        "failed_shards_requested": list(fail_shards),
        "kill_after_seconds": args.kill_after_seconds,
        "kill_events": kill_events,
        "shards": pod_records,
    }
    write_json(args.run_root / "controller" / f"phase_{phase}.json", summary)
    return summary


def snapshot_completed_shards(args: argparse.Namespace, expected_by_shard: Dict[int, List[str]]) -> Dict[str, Any]:
    snapshot = {}
    for shard_id in range(TOTAL_SHARDS):
        output_path = args.state_dir / f"shard_{shard_id}" / "output.jsonl"
        if shard_complete(args, shard_id, expected_by_shard[shard_id]):
            snapshot[str(shard_id)] = {"completed": True, "output_sha256": file_sha256(output_path), "rows": len(read_jsonl(output_path))}
    payload = {"condition": args.condition, "rep": args.rep, "snapshot": snapshot}
    write_json(args.run_root / "controller" / "completed_shards_before_retry.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    args.python = sys.executable
    args.gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not args.gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one GPU id")
    args.run_root = run_root(args)
    args.state_dir = args.run_root / "state"

    manifest = build_manifest(args)
    planned_only = args.dry_run or args.allow_unimplemented
    manifest["execution_status"] = "planned_native_recovery_run_not_executed" if planned_only else "completed"
    if planned_only:
        write_json(args.run_root / "controller" / "dry_run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.run_root.exists() and any(args.run_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Run directory already exists: {args.run_root} (pass --overwrite to replace it)")
        shutil.rmtree(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_json(args.run_root / "controller" / "run_manifest.json", manifest)

    try:
        rows = read_jsonl(Path(args.shared_root) / "data" / "ultrachat_200k" / "subset.jsonl")
        order_rows = read_jsonl(Path(args.shared_root) / "data" / "ultrachat_200k" / "id_order.jsonl")
    except Exception as exc:
        raise SystemExit(f"Failed to read recovery workload under --shared-root: {exc}")

    expected_ids = [str(row["stable_id"]) for row in sorted(order_rows, key=lambda row: int(row["order_index"]))]
    if len(rows) != len(expected_ids):
        raise SystemExit(f"Workload rows ({len(rows)}) do not match id_order rows ({len(expected_ids)})")
    expected_by_shard: Dict[int, List[str]] = {}
    for shard_id in range(TOTAL_SHARDS):
        start = (len(rows) * shard_id) // TOTAL_SHARDS
        end = (len(rows) * (shard_id + 1)) // TOTAL_SHARDS
        expected_by_shard[shard_id] = expected_ids[start:end]

    write_shard_inputs(args, rows, list(range(TOTAL_SHARDS)), TOTAL_SHARDS)
    fail_shards = CONDITION_FAILURE_SHARDS[args.condition]

    overall_started = time.monotonic()
    initial = run_phase(args, "initial", list(range(TOTAL_SHARDS)), fail_shards, expected_by_shard)
    snapshot = snapshot_completed_shards(args, expected_by_shard)
    completed_before_retry = {int(k): v["output_sha256"] for k, v in snapshot["snapshot"].items()}

    retried_shards: List[int] = []
    retry_summaries = []
    rounds_needed = 0
    initial_incomplete = [shard_id for shard_id in range(TOTAL_SHARDS) if not shard_complete(args, shard_id, expected_by_shard[shard_id])]
    incomplete = list(initial_incomplete)
    for round_idx in range(1, args.max_rounds + 1):
        incomplete = [shard_id for shard_id in range(TOTAL_SHARDS) if not shard_complete(args, shard_id, expected_by_shard[shard_id])]
        if not incomplete:
            break
        rounds_needed = round_idx
        retried_shards.extend(incomplete)
        retry_summaries.append(run_phase(args, f"retry_{round_idx}", incomplete, [], expected_by_shard))

    final_incomplete = [shard_id for shard_id in range(TOTAL_SHARDS) if not shard_complete(args, shard_id, expected_by_shard[shard_id])]
    overall_wall = round(time.monotonic() - overall_started, 3)

    merged_rows: List[Dict[str, Any]] = []
    for shard_id in range(TOTAL_SHARDS):
        output_path = args.state_dir / f"shard_{shard_id}" / "output.jsonl"
        if output_path.exists():
            merged_rows.extend(read_jsonl(output_path))
    order_by_id = {str(row["stable_id"]): int(row["order_index"]) for row in order_rows}
    merged_rows.sort(key=lambda row: order_by_id.get(str(row["stable_id"]), 1 << 60))
    merged_path = args.run_root / "merged" / "merged.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    merged_ids = [str(row.get("stable_id")) for row in merged_rows]
    missing = sorted(set(expected_ids) - set(merged_ids))
    unexpected = sorted(set(merged_ids) - set(expected_ids))
    duplicates = sorted({item for item in merged_ids if merged_ids.count(item) > 1})

    changed_completed: List[int] = []
    only_incomplete_or_killed: bool = True
    killed = set(fail_shards)
    for shard_id in completed_before_retry:
        if shard_id in retried_shards:
            changed_completed.append(shard_id)
            only_incomplete_or_killed = False
    for shard_id in retried_shards:
        if shard_id not in initial_incomplete and shard_id not in killed:
            only_incomplete_or_killed = False
    unchanged = True
    for shard_id, sha_before in completed_before_retry.items():
        if shard_id in retried_shards:
            continue
        current = file_sha256(args.state_dir / f"shard_{shard_id}" / "output.jsonl")
        if current != sha_before:
            unchanged = False

    checks = {
        "no_missing_ids": not missing,
        "no_duplicate_ids": not duplicates,
        "no_unexpected_ids": not unexpected,
        "order_after_merge_matches_expected": merged_ids == expected_ids,
        "completed_shard_outputs_unchanged_after_retry": unchanged,
        "retry_only_incomplete_or_killed_shards": only_incomplete_or_killed,
    }
    validation = {
        "valid": all(checks.values()),
        "checks": checks,
        "final_row_count": len(merged_ids),
        "missing_ids": missing,
        "duplicate_ids": duplicates,
        "unexpected_ids": unexpected,
        "order_after_merge_matches_expected": checks["order_after_merge_matches_expected"],
        "final_incomplete_shards": final_incomplete,
    }
    rows_recomputed = sum(len(expected_by_shard[shard_id]) for shard_id in retried_shards)
    summary = {
        "framework": args.framework,
        "condition": args.condition,
        "rep": args.rep,
        "native_mode": True,
        "execution_status": "completed",
        "metrics": {
            "shards_recomputed_count": len(set(retried_shards)),
            "completed_shards_reused": TOTAL_SHARDS - len(set(retried_shards)),
            "rows_recomputed": rows_recomputed,
            "fraction_of_total_workload_recomputed": round(rows_recomputed / len(rows), 6),
            "failure_plus_recovery_wall_time_seconds": overall_wall,
            "initial_phase_wall_seconds": initial["wall_seconds"],
            "retry_phases_wall_seconds": [summary_entry["wall_seconds"] for summary_entry in retry_summaries],
            "retry_rounds_needed": rounds_needed,
            "final_incomplete_shards": final_incomplete,
        },
        "validation": validation,
        "completed_shards_before_retry_count": len(completed_before_retry),
        "retried_shards": sorted(set(retried_shards)),
    }
    write_json(args.run_root / "summary.json", summary)
    write_json(args.run_root / "validation.json", validation)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
