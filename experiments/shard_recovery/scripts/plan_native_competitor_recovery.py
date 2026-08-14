#!/usr/bin/env python3
"""Print dry-run manifests for native shard-recovery competitors."""

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
    parser.add_argument("--framework", default="all")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select(mapping: dict[str, Any], requested: str, label: str) -> list[str]:
    available = sorted(mapping)
    if requested == "all":
        return available
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown {label}(s): {missing}. Available: {available}")
    return selected


def build_manifest(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    frameworks = select(config["frameworks"], args.framework, "framework")
    conditions = select(config["conditions"], args.condition, "condition")
    runs = []
    for framework in frameworks:
        for condition in conditions:
            values = {
                "python": args.python,
                "framework": framework,
                "condition": condition,
                "rep": args.rep,
                "shared_root": config["execution"]["shared_root"],
                "gpu_ids": args.gpu_ids,
                "max_active_shards": config["task"]["max_active_shards"],
            }
            runs.append(
                {
                    "framework": framework,
                    "condition": condition,
                    "rep": args.rep,
                    "failed_shards": config["conditions"][condition],
                    "native_backend": config["frameworks"][framework]["native_backend"],
                    "environment": config["frameworks"][framework]["environment"],
                    "retry_policy": config["execution"]["retry_policy"],
                    "command": config["execution"]["command_template"].format_map(values).split(),
                }
            )
    return {
        "dry_run": True,
        "experiment": "shard_recovery",
        "native_mode": True,
        "task": config["task"],
        "conditions": config["conditions"],
        "validation_contract": config["task"]["output_contract"],
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
