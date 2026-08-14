#!/usr/bin/env python3
"""Print dry-run manifests for native single-node scaling competitors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "native_competitors.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--framework", default="all", help="Framework name or 'all'.")
    parser.add_argument("--gpu-count", default="all", help="GPU point or 'all'.")
    parser.add_argument("--visible-gpus", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_frameworks(config: dict[str, Any], requested: str) -> list[str]:
    available = sorted((config.get("frameworks") or {}).keys())
    if requested == "all":
        return available
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown framework(s): {missing}. Available: {available}")
    return selected


def select_gpu_counts(config: dict[str, Any], requested: str) -> list[int]:
    available = [int(item) for item in config["execution"]["gpu_points"]]
    if requested == "all":
        return available
    selected = [int(item.strip()) for item in requested.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unsupported GPU point(s): {missing}. Available: {available}")
    return selected


def truncate_visible_gpus(raw: str, gpu_count: int) -> str:
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(gpu_ids) < gpu_count:
        raise ValueError(f"Need at least {gpu_count} visible GPU ids, got {gpu_ids}")
    return ",".join(gpu_ids[:gpu_count])


def build_manifest(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    frameworks = select_frameworks(config, args.framework)
    gpu_counts = select_gpu_counts(config, args.gpu_count)
    repetitions = args.repetitions or int(config["execution"].get("repetitions", 3))
    task = config["task"]
    output_root = config["execution"]["output_root"]

    runs = []
    for framework in frameworks:
        spec = config["frameworks"][framework]
        for gpu_count in gpu_counts:
            values = {
                "python": args.python,
                "workload_jsonl": task["source_workload"],
                "output_root": output_root,
                "gpu_count": gpu_count,
                "visible_gpus": truncate_visible_gpus(args.visible_gpus, gpu_count),
                "repetitions": repetitions,
            }
            runs.append(
                {
                    "framework": framework,
                    "gpu_count": gpu_count,
                    "visible_gpus": values["visible_gpus"],
                    "repetitions": repetitions,
                    "environment": spec.get("environment"),
                    "output_dir": spec.get("output_dir"),
                    "native_backend": spec.get("native_backend"),
                    "command": spec["command_template"].format_map(values).split(),
                }
            )
    return {
        "dry_run": True,
        "experiment": "single_node_h100_scaling",
        "native_mode": True,
        "task": task,
        "validation_contract": task["output_contract"],
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(read_yaml(Path(args.config)), args)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
