#!/usr/bin/env python3
"""Plan a native DataTrove ChartQA run for the comparison experiment.

The native DataTrove public benchmark path is text-generation oriented. This
runner records the exact multimodal contract and refuses to claim a completed
native run until the DataTrove multimodal path or a documented adapter is wired.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unimplemented", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    input_path = Path(args.input_jsonl)
    image_base = Path(args.image_base_path)
    rows = read_jsonl(input_path)
    missing_images = []
    for row in rows[:20]:
        image_path = row.get("image_path")
        if image_path and not (Path(image_path) if Path(str(image_path)).is_absolute() else image_base / str(image_path)).exists():
            missing_images.append(str(image_path))

    summary = {
        "framework": "datatrove",
        "returncode": 0,
        "native_mode": True,
        "execution_status": "planned_native_datatrove_chartqa_run_not_executed_by_this_pr",
        "input_jsonl": str(input_path),
        "image_base_path": str(image_base),
        "input_rows": len(rows),
        "sample_missing_images": missing_images,
        "model": args.model,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
        "output_contract": {
            "required_fields": ["id", "messages", "metadata"],
            "metadata_fields": ["reference_answer", "generated_answer_normalized", "source"],
            "validation": ["row_count_matches_input", "ids_match_input", "no_duplicate_ids", "row_order_matches_input", "nested_message_schema_valid"],
        },
        "native_backend_boundary": "Use DataTrove native multimodal inference if supported; otherwise document any adapter glue before reporting results.",
        "full_end_to_end_wall_seconds": round(time.perf_counter() - started, 6),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run and not args.allow_unimplemented:
        raise SystemExit(
            "Native DataTrove ChartQA execution is planned but not executed by this scaffold. "
            "Use --dry-run for validation manifests or wire the verified native multimodal path."
        )


if __name__ == "__main__":
    main()
