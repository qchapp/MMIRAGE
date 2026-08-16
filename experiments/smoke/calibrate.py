#!/usr/bin/env python3
"""Calibrate per-experiment workload sizes from smoke timings.

Reads ``smoke/timing.json`` (written by ``run_smoke.py``) and for each
experiment computes the row count whose expected wall for the full run command
stays under the experiment's ``target_seconds`` budget.

The smoke measured the full run command on ``rows_smoke`` rows, which contains
``cells`` = (startup + generation) units (paths * repetitions). Per cell:

    budget_per_cell = target / (cells * recovery_factor)
    gen_at_size = budget_per_cell - startup
    recommended = rows_smoke * gen_at_size / gen_per_cell_smoke
    size = clamp(recommended, rows_smoke, max)

``startup`` is the measured model-load time and ``gen_per_cell_smoke`` is
``(wall_smoke / cells) - startup``. When generation time is ~0 the
``preferred`` size is kept. The result is written to
``<experiment>/configs/workload_size.yaml``, which the experiment's
``prepare_workload.py`` reads as its default size. Without ``--apply`` the
recommended sizes are printed and saved to ``calibration.json`` only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SMOKE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_DIR.parents[1]
DEFAULT_CONFIG = SMOKE_DIR / "config.yaml"
DEFAULT_TIMING = SMOKE_DIR / "timing.json"
DEFAULT_OUTPUT = SMOKE_DIR / "calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--timing", default=str(DEFAULT_TIMING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--apply", action="store_true", help="Write configs/workload_size.yaml in each experiment dir.")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def calibrate_experiment(name: str, cfg: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    rows_smoke = int(timing["rows_smoke"])
    wall = float(timing["wall_seconds"] or 0.0)
    load = float(timing.get("model_load_seconds") or 0.0)
    target = float(cfg["target_seconds"])
    cells = int(cfg.get("cells", 1))
    factor = float(cfg.get("recovery_factor", 1.0))
    size_cfg = cfg["size"]
    preferred = int(size_cfg["preferred"])
    maximum = int(size_cfg["max"])

    startup = load
    scope = str(timing.get("scope", "per_cell"))
    if scope == "whole_command":
        per_cell = wall / cells if wall > 0 and cells > 0 else 0.0
    else:
        per_cell = wall if wall > 0 else 0.0
    gen_per_cell = max(0.0, per_cell - startup)
    budget_per_cell = target / (cells * factor) if cells > 0 and factor > 0 else target

    if gen_per_cell > 0 and budget_per_cell > startup:
        recommended = math.ceil(rows_smoke * (budget_per_cell - startup) / gen_per_cell)
    else:
        recommended = preferred

    size = max(rows_smoke, min(maximum, recommended))
    expected_wall = cells * factor * (startup + gen_per_cell * (size / rows_smoke)) if rows_smoke > 0 else None
    over_budget = expected_wall is not None and expected_wall > target + 1.0

    return {
        "rows_smoke": rows_smoke,
        "smoke_wall_seconds": round(wall, 6),
        "smoke_model_load_seconds": round(load, 6),
        "smoke_scope": scope,
        "smoke_cells": cells,
        "per_cell_wall_seconds": round(per_cell, 6),
        "per_cell_generation_seconds": round(gen_per_cell, 6),
        "target_seconds": target,
        "recovery_factor": factor,
        "budget_per_cell_seconds": round(budget_per_cell, 6),
        "recommended": recommended,
        "size": size,
        "clamped": size != recommended,
        "expected_wall_seconds": round(expected_wall, 2) if expected_wall is not None else None,
        "over_budget": over_budget,
        "size_key": size_cfg["key"],
        "size_file": cfg.get("size_file", f"experiments/{name}/configs/workload_size.yaml"),
    }


def main() -> int:
    args = parse_args()
    config = read_yaml(Path(args.config))
    timing_payload = json.loads(Path(args.timing).read_text(encoding="utf-8"))
    timing_runs = timing_payload.get("runs", {})

    calibrations: Dict[str, Any] = {}
    warnings: list[str] = []
    for name, cfg in config["experiments"].items():
        if name not in timing_runs:
            warnings.append(f"{name}: no smoke timing available, skipped")
            continue
        calibration = calibrate_experiment(name, cfg, timing_runs[name])
        calibrations[name] = calibration
        if calibration["over_budget"]:
            warnings.append(
                f"{name}: expected full wall {calibration['expected_wall_seconds']}s exceeds "
                f"budget {calibration['target_seconds']}s at max size {calibration['size']}"
            )

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": bool(args.apply),
        "calibrations": calibrations,
        "warnings": warnings,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.apply:
        for name, calibration in calibrations.items():
            size_path = PROJECT_ROOT / calibration["size_file"]
            write_yaml(
                size_path,
                {calibration["size_key"]: calibration["size"]},
            )
            print(f"[applied] {calibration['size_file']} -> {calibration['size_key']}: {calibration['size']}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
