#!/usr/bin/env python3
"""Extract ANONLIB shard recovery metrics into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONTAINER_REPO = Path("/workspace/ANONLIB")
DEFAULT_SHARED_ROOT = "/workspace/anonlib-recovery"
sys.path.insert(0, str(REPO_ROOT / "src"))

from anonlib.cli_utils.status import collect_bench_stats  # noqa: E402
from anonlib.config.utils import load_anonlib_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", default=os.environ.get("ANONLIB_RECOVERY_ROOT", DEFAULT_SHARED_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONTAINER_REPO / "experiments" / "shard_recovery" / "configs" / "anonlib_recovery.yaml"))
    parser.add_argument("--conditions", default="baseline,fail_1,fail_4,fail_8")
    parser.add_argument("--reps", default="1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def require_container_terminal(args: argparse.Namespace) -> None:
    if not Path(args.shared_root).is_absolute():
        raise RuntimeError(
            f"--shared-root must be an absolute path inside the container, got {args.shared_root!r}. "
            f"Recommended: {DEFAULT_SHARED_ROOT}"
        )
    if not Path(args.config).exists():
        raise RuntimeError(
            f"ANONLIB config not found at {args.config}. Run from the ANONLIB container terminal "
            f"with the repository available at {DEFAULT_CONTAINER_REPO}, or pass --config explicitly."
        )


def run_dir(shared_root: str, condition: str, rep: int) -> Path:
    return Path(shared_root) / "runs" / condition / f"rep_{rep:02d}"


def runtime_env(shared_root: str, condition: str, rep: int) -> Dict[str, str]:
    rd = run_dir(shared_root, condition, rep)
    return {
        "ANONLIB_RECOVERY_ROOT": shared_root,
        "ANONLIB_RECOVERY_RUN_DIR": str(rd),
        "ANONLIB_RECOVERY_INPUT_JSONL": str(Path(shared_root) / "data" / "ultrachat_200k" / "subset.jsonl"),
        "ANONLIB_RECOVERY_STATE_DIR": str(rd / "state"),
        "ANONLIB_RECOVERY_OUTPUT_DIR": str(rd / "output"),
    }


def load_cfg(config: str, env: Dict[str, str]) -> Any:
    old = os.environ.copy()
    os.environ.update(env)
    try:
        return load_anonlib_config(config)
    finally:
        os.environ.clear()
        os.environ.update(old)


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def phase_summaries(rd: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((rd / "controller").glob("phase_*.json")):
        payload = read_json(path)
        if payload:
            rows.append(payload)
    return rows


def dir_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8") + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def expected_ids(shared_root: str) -> List[str]:
    path = Path(shared_root) / "data" / "ultrachat_200k" / "id_order.jsonl"
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["anonlib_id"])
    return ids


def load_merged_ids(rd: Path) -> Dict[str, Any]:
    merged_dir = rd / "merged"
    if not merged_dir.exists():
        return {"final_row_count": None, "ids": [], "error": f"Missing merged dataset at {merged_dir}"}
    try:
        ds = load_from_disk(str(merged_dir))
        ids = list(ds["anonlib_id"])
        return {"final_row_count": len(ids), "ids": ids, "error": None}
    except Exception as exc:
        return {"final_row_count": None, "ids": [], "error": str(exc)}


def missing_duplicate_order(actual_ids: Sequence[str], expected: Sequence[str]) -> Dict[str, Any]:
    actual_set = set(actual_ids)
    expected_set = set(expected)
    seen = set()
    duplicates = []
    for item in actual_ids:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    return {
        "missing_id_count": len(missing),
        "duplicate_id_count": len(duplicates),
        "unexpected_id_count": len(unexpected),
        "missing_ids_first20": missing[:20],
        "duplicate_ids_first20": duplicates[:20],
        "unexpected_ids_first20": unexpected[:20],
        "order_matches_original_input": list(actual_ids) == list(expected),
    }


def completed_unchanged(rd: Path) -> Optional[bool]:
    snapshot = read_json(rd / "controller" / "completed_shards_before_retry.json")
    if not snapshot:
        return None
    for item in snapshot.get("shards", []):
        shard_id = item["shard_id"]
        before = item.get("sha256")
        current = dir_sha256(rd / "output" / f"shard_{shard_id}")
        if before != current:
            return False
    return True


def pod_gpu_seconds(phases: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    for phase in phases:
        for pod in phase.get("pods", []):
            runtime = pod.get("runtime_seconds")
            if runtime is not None:
                total += float(runtime)
    return round(total, 3)


def wasted_gpu_seconds(phases: Sequence[Dict[str, Any]]) -> float:
    killed_pods = set()
    elapsed_by_pod: Dict[str, float] = {}
    for phase in phases:
        kill_after = phase.get("kill_after") or {}
        fallback_seconds = float(kill_after.get("seconds") or 0.0)
        for event in phase.get("kill_events", []):
            killed_pods.add(event.get("pod"))
            elapsed_by_pod[event.get("pod")] = fallback_seconds
    total = 0.0
    for phase in phases:
        for pod in phase.get("pods", []):
            name = pod.get("pod")
            if name in killed_pods:
                total += float(pod.get("runtime_seconds") or elapsed_by_pod.get(name) or 0.0)
    return round(total, 3)


def wall_seconds(phases: Sequence[Dict[str, Any]]) -> float:
    return round(sum(float(phase.get("wall_seconds") or 0.0) for phase in phases), 3)


def retry_launched_shards(phases: Sequence[Dict[str, Any]]) -> List[int]:
    shards = set()
    for phase in phases:
        if str(phase.get("phase", "")).startswith("retry_"):
            shards.update(int(item) for item in phase.get("launched_shards", []))
    return sorted(shards)


def summarize_run(shared_root: str, config: str, condition: str, rep: int, baseline_wall: Optional[float], expected: Sequence[str]) -> Dict[str, Any]:
    rd = run_dir(shared_root, condition, rep)
    env = runtime_env(shared_root, condition, rep)
    cfg = load_cfg(config, env)
    old = os.environ.copy()
    os.environ.update(env)
    try:
        stats = collect_bench_stats(cfg)
    finally:
        os.environ.clear()
        os.environ.update(old)

    phases = phase_summaries(rd)
    merged = load_merged_ids(rd)
    integrity = missing_duplicate_order(merged["ids"], expected) if merged["ids"] else {
        "missing_id_count": None,
        "duplicate_id_count": None,
        "unexpected_id_count": None,
        "missing_ids_first20": [],
        "duplicate_ids_first20": [],
        "unexpected_ids_first20": [],
        "order_matches_original_input": None,
    }
    recomputed = retry_launched_shards(phases)
    per_shard = stats.get("per_shard", [])
    recomputed_rows = 0
    recomputed_output_tokens = 0
    total_output_tokens = 0
    for entry in per_shard:
        shard_id = entry.get("shard_id")
        s = entry.get("stats") or {}
        if s.get("output_tokens") is not None:
            total_output_tokens += int(s.get("output_tokens") or 0)
        if shard_id in recomputed:
            recomputed_rows += int(s.get("rows_processed") or 0)
            recomputed_output_tokens += int(s.get("output_tokens") or 0)
    final_rows = merged.get("final_row_count")
    condition_wall = wall_seconds(phases)
    initial_phase = next((phase for phase in phases if phase.get("phase") == "initial"), {})
    initially_failed = initial_phase.get("anonlib_retryable_shards_after_phase") or initial_phase.get("failed_shards_requested") or []
    result = {
        "condition": condition,
        "rep": rep,
        "total_shards": cfg.loading_params.get_num_shards(),
        "initially_failed_shards": initially_failed,
        "initially_failed_shard_count": len(initially_failed),
        "shards_recomputed": recomputed,
        "shards_recomputed_count": len(recomputed),
        "completed_shards_reused": cfg.loading_params.get_num_shards() - len(recomputed),
        "clean_baseline_wall_time_seconds": baseline_wall,
        "failure_plus_recovery_wall_time_seconds": condition_wall,
        "additional_wall_time_seconds": round(condition_wall - baseline_wall, 3) if baseline_wall is not None else None,
        "gpu_seconds_spent": pod_gpu_seconds(phases),
        "estimated_gpu_seconds_wasted": wasted_gpu_seconds(phases),
        "rows_recomputed": recomputed_rows,
        "output_tokens_recomputed_closest_available": recomputed_output_tokens if total_output_tokens else None,
        "token_recomputation_note": "ANONLIB status records final successful shard token counts, not token progress lost inside a killed pod. The recomputed token count is therefore the output tokens generated by successful retry shards, not exact token-level lost work before termination.",
        "fraction_of_total_workload_recomputed": round(recomputed_rows / final_rows, 6) if final_rows else None,
        "final_row_count": final_rows,
        "missing_id_count": integrity["missing_id_count"],
        "duplicate_id_count": integrity["duplicate_id_count"],
        "unexpected_id_count": integrity["unexpected_id_count"],
        "order_after_merge_matches_original_input_order": integrity["order_matches_original_input"],
        "completed_shard_outputs_unchanged_after_retry": completed_unchanged(rd),
        "anonlib_stats": stats,
        "merge_error": merged.get("error"),
        "integrity_detail": integrity,
    }
    return result


def mean_std(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.mean(clean), 6), "std": round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "condition",
        "rep",
        "total_shards",
        "initially_failed_shard_count",
        "shards_recomputed_count",
        "completed_shards_reused",
        "clean_baseline_wall_time_seconds",
        "failure_plus_recovery_wall_time_seconds",
        "additional_wall_time_seconds",
        "gpu_seconds_spent",
        "estimated_gpu_seconds_wasted",
        "rows_recomputed",
        "output_tokens_recomputed_closest_available",
        "fraction_of_total_workload_recomputed",
        "final_row_count",
        "missing_id_count",
        "duplicate_id_count",
        "unexpected_id_count",
        "order_after_merge_matches_original_input_order",
        "completed_shard_outputs_unchanged_after_retry",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    metrics = [
        "failure_plus_recovery_wall_time_seconds",
        "additional_wall_time_seconds",
        "gpu_seconds_spent",
        "estimated_gpu_seconds_wasted",
        "rows_recomputed",
        "fraction_of_total_workload_recomputed",
    ]
    return {
        condition: {metric: mean_std(row.get(metric) for row in condition_rows) for metric in metrics}
        for condition, condition_rows in grouped.items()
    }


def main() -> None:
    args = parse_args()
    require_container_terminal(args)
    conditions = parse_str_list(args.conditions)
    reps = parse_int_list(args.reps)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.shared_root) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_ids(args.shared_root)

    baseline_by_rep: Dict[int, float] = {}
    for rep in reps:
        phases = phase_summaries(run_dir(args.shared_root, "baseline", rep))
        if phases:
            baseline_by_rep[rep] = wall_seconds(phases)

    rows = []
    for rep in reps:
        baseline_wall = baseline_by_rep.get(rep) or baseline_by_rep.get(1)
        for condition in conditions:
            rd = run_dir(args.shared_root, condition, rep)
            if not rd.exists():
                continue
            rows.append(summarize_run(args.shared_root, args.config, condition, rep, baseline_wall, expected))

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "interpretation": "Kubernetes terminates selected pods; ANONLIB recovery semantics are shard-scoped state detection and rerunning only incomplete shards. This is not evidence of a native Kubernetes backend.",
        "individual_runs": rows,
        "summary_mean_std": build_summary(rows),
    }
    (out_dir / "recovery_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "recovery_results.csv", rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
