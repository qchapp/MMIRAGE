#!/usr/bin/env python3
"""Dry-run-safe native competitor recovery harness skeleton."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONDITION_FAILURE_SHARDS = {
    "baseline": [],
    "fail_1": [3],
    "fail_4": [1, 5, 9, 13],
    "fail_8": [0, 2, 4, 6, 8, 10, 12, 14],
}
FRAMEWORKS = {"datatrove", "nemo_curator", "distilabel", "ray_data_llm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=sorted(FRAMEWORKS), required=True)
    parser.add_argument("--condition", choices=sorted(CONDITION_FAILURE_SHARDS), required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--shared-root", default="/workspace/anonlib-recovery")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-active-shards", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unimplemented", action="store_true")
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    failed = CONDITION_FAILURE_SHARDS[args.condition]
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    waves = []
    all_shards = list(range(16))
    for start in range(0, len(all_shards), args.max_active_shards):
        shards = all_shards[start : start + args.max_active_shards]
        waves.append({"phase": "initial", "shards": shards, "killed_shards": [item for item in shards if item in failed]})
    retry_shards = [] if args.condition == "baseline" else failed
    if retry_shards:
        waves.append({"phase": "retry_1", "shards": retry_shards, "killed_shards": []})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": args.framework,
        "native_mode": True,
        "execution_status": "planned_native_recovery_run_not_executed_by_this_pr",
        "condition": args.condition,
        "rep": args.rep,
        "shared_root": args.shared_root,
        "gpu_ids": gpu_ids,
        "total_shards": 16,
        "initially_failed_shards": failed,
        "retry_policy": "relaunch shards without valid completion marker",
        "planned_waves": waves,
        "output_contract": {
            "final_row_order": "must_match_expected_id_order",
            "validation": [
                "no_missing_ids",
                "no_duplicate_ids",
                "no_unexpected_ids",
                "completed_shard_outputs_unchanged_after_retry",
                "retry_only_incomplete_or_killed_shards",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    out_dir = Path(args.shared_root) / "native_competitors" / args.framework / args.condition / f"rep_{args.rep:02d}" / "controller"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.dry_run and not args.allow_unimplemented:
        raise SystemExit(
            "Native competitor recovery execution is planned but not implemented in this scaffold. "
            "Use --dry-run for manifests or wire the framework-specific pod runner."
        )


if __name__ == "__main__":
    main()
