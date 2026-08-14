#!/usr/bin/env python3
"""Native text-generation competitor harness skeleton for the scaling experiment.

This file records the exact task, shard split, and output contract for native
competitor runs. It intentionally refuses to execute GPU workloads unless a
framework-specific implementation fills in the native run path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FRAMEWORK_BACKENDS = {
    "datatrove": "DataTrove InferenceRunner with native vLLM backend",
    "nemo_curator": "NeMo Curator/Data Designer native pipeline executor",
    "distilabel": "Distilabel native pipeline with local vLLM task",
    "ray_data_llm": "Ray Data LLM native processor with vLLM engine",
    "raw_sglang": "direct SGLang engine/client baseline",
}


def parse_args(framework_override: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=sorted(FRAMEWORK_BACKENDS), required=framework_override is None)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--visible-gpus", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unimplemented", action="store_true", help="Write a manifest and exit zero even without native execution code.")
    args = parser.parse_args()
    if framework_override is not None:
        args.framework = framework_override
    return args


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def shard_ranges(total_rows: int, shards: int) -> list[dict[str, int]]:
    ranges = []
    for shard_id in range(shards):
        start = (total_rows * shard_id) // shards
        end = (total_rows * (shard_id + 1)) // shards
        ranges.append({"shard_id": shard_id, "start_row": start, "end_row": end, "rows": end - start})
    return ranges


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    workload = Path(args.workload_jsonl)
    if not workload.exists():
        raise FileNotFoundError(workload)
    gpu_ids = [item.strip() for item in args.visible_gpus.split(",") if item.strip()]
    if len(gpu_ids) < args.gpu_count:
        raise ValueError(f"Need {args.gpu_count} visible GPUs, got {gpu_ids}")
    rows = count_jsonl(workload)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": args.framework,
        "native_backend": FRAMEWORK_BACKENDS[args.framework],
        "native_mode": True,
        "execution_status": "planned_native_run_not_executed_by_this_pr",
        "workload_jsonl": str(workload),
        "input_rows": rows,
        "model": args.model,
        "decoding": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "gpu_count": args.gpu_count,
        "visible_gpus": gpu_ids[: args.gpu_count],
        "repetitions": args.repetitions,
        "shards": shard_ranges(rows, args.gpu_count),
        "output_contract": {
            "required_fields": ["stable_id", "source_index", "prompt_sha256", "prompt_text", "answer"],
            "row_order": "must_match_input_order",
            "validation": [
                "processed_rows_equals_input_rows",
                "stable_id_set_matches_input",
                "no_duplicate_stable_ids",
                "prompt_sha256_preserved",
                "schema_valid_for_every_row",
            ],
        },
    }


def main(framework_override: str | None = None) -> None:
    args = parse_args(framework_override)
    manifest = build_manifest(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{args.framework}_gpu{args.gpu_count}_native_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.dry_run and not args.allow_unimplemented:
        raise SystemExit(
            "Native execution is intentionally not run by this planning harness. "
            "Use --dry-run for manifests, or replace this skeleton with a framework-specific runner."
        )


if __name__ == "__main__":
    main()
