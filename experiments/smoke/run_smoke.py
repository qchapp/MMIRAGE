#!/usr/bin/env python3
"""Run one small smoke cell per experiment and record its timing.

Each smoke cell is one repetition of the experiment's primary path on a tiny
workload subset (``rows_smoke`` in ``config.yaml``). The measured wall time and
model-load time are written to ``timing.json`` for ``calibrate.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SMOKE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_DIR.parents[1]
DEFAULT_CONFIG = SMOKE_DIR / "config.yaml"
DEFAULT_TIMING = SMOKE_DIR / "timing.json"
DEFAULT_SHARED_ROOT = "/workspace/mmirage-recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--timing", default=str(DEFAULT_TIMING))
    parser.add_argument("--shared-root", default=DEFAULT_SHARED_ROOT)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment names to smoke (default: all in config).",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Assume tiny workloads are already prepared.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run(command: str, verbose: bool) -> Dict[str, Any]:
    started = time.perf_counter()
    if verbose:
        print(f"$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def nested_get(data: Any, path: str) -> Any:
    node = data
    for key in path.split(".") if path else []:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def metric_value(node: Any, key: str) -> Optional[float]:
    if not key:
        return None
    if not isinstance(node, dict) or key not in node:
        return None
    value = node[key]
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_timing(cfg: Dict[str, Any], shared_root: str) -> Dict[str, Any]:
    spec = cfg["timing"]
    file_path = Path(str(spec["file"]).format(shared_root=shared_root))
    data = json.loads(file_path.read_text(encoding="utf-8"))
    node = nested_get(data, str(spec.get("path", "")))
    wall = metric_value(node, spec["wall"])
    load = metric_value(node, spec.get("load", ""))
    return {
        "file": str(file_path),
        "wall_seconds": wall,
        "model_load_seconds": load,
        "scope": str(spec.get("scope", "per_cell")),
    }


def main() -> int:
    args = parse_args()
    config = read_yaml(Path(args.config))
    experiments = config["experiments"]
    selected = [item.strip() for item in (args.experiments or "").split(",") if item.strip()]
    if selected:
        unknown = sorted(set(selected) - set(experiments))
        if unknown:
            raise SystemExit(f"Unknown experiments: {unknown}; valid: {sorted(experiments)}")
        names = [name for name in selected if name in experiments]
    else:
        names = list(experiments)

    shared_root = args.shared_root
    results: Dict[str, Any] = {}
    failures: list[str] = []
    for name in names:
        cfg = experiments[name]
        rows_smoke = int(cfg["rows_smoke"])
        print(f"[smoke] {name} (rows_smoke={rows_smoke})", flush=True)
        if not args.skip_prepare:
            prepare = str(cfg["prepare"]).strip().replace("\n", " ").format(rows=rows_smoke, shared_root=shared_root)
            outcome = run(prepare, args.verbose)
            if outcome["returncode"] != 0:
                print(f"  prepare failed:\n{outcome['stderr']}", file=sys.stderr)
                failures.append(name)
                continue
        measure = str(cfg["measure"]).strip().replace("\n", " ").format(rows=rows_smoke, shared_root=shared_root)
        outcome = run(measure, args.verbose)
        if outcome["returncode"] != 0:
            print(f"  measure failed:\n{outcome['stderr']}", file=sys.stderr)
            failures.append(name)
            continue
        try:
            timing = read_timing(cfg, shared_root)
        except Exception as exc:
            print(f"  timing read failed: {exc}", file=sys.stderr)
            failures.append(name)
            continue
        wall = timing["wall_seconds"]
        load = timing["model_load_seconds"] or 0.0
        results[name] = {
            "rows_smoke": rows_smoke,
            "wall_seconds": wall,
            "model_load_seconds": load,
            "generation_seconds": round(max(0.0, wall - load), 6) if wall is not None else None,
            "scope": timing["scope"],
            "timing_source": timing["file"],
        }
        print(
            f"  wall={results[name]['wall_seconds']}s load={load}s gen={results[name]['generation_seconds']}s",
            flush=True,
        )

    timing_path = Path(args.timing)
    previous_runs: Dict[str, Any] = {}
    previous_failed: list[str] = []
    if timing_path.exists():
        try:
            previous = json.loads(timing_path.read_text(encoding="utf-8"))
            previous_runs = dict(previous.get("runs") or {})
            previous_failed = list(previous.get("failed") or [])
        except (OSError, json.JSONDecodeError):
            pass
    merged_runs = {name: run for name, run in previous_runs.items() if name not in names}
    merged_runs.update(results)
    merged_failed = sorted((set(previous_failed) - set(names)) | set(failures))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shared_root": shared_root,
        "runs": merged_runs,
        "failed": merged_failed,
        "next": "python experiments/smoke/calibrate.py --apply",
    }
    timing_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
