#!/usr/bin/env python3
"""Per-GPU shard worker for native text-generation competitor baselines.

Runs one framework-native generation pass over an input shard and writes:

* the contract JSONL (``--output-jsonl``), one row per input row, in input order
* a status JSON (``--status-json``) in the ANONLIB shard schema so the existing
  scaling aggregator can consume it unchanged

The worker is expected to be launched with ``CUDA_VISIBLE_DEVICES=<gpu>`` set by
the orchestrator; ``--gpu-id`` is recorded for metadata only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=sorted(_frameworks().keys()))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-style", choices=["rewrite", "raw"], default="rewrite")
    parser.add_argument("--id-field", default="stable_id", help="Workload id column carried into contract output rows.")
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


def _frameworks() -> dict[str, Any]:
    from experiments._shared.native_frameworks import FRAMEWORK_RUNNERS

    return FRAMEWORK_RUNNERS


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    status_path = Path(args.status_json)
    try:
        rows = load_rows(input_path)
        write_json(status_path.with_name("running.json"), {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu_id": args.gpu_id, "input_rows": len(rows)})
        runner = _frameworks()[args.framework]
        stats = runner(
            rows=rows,
            model=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            concurrency=args.concurrency,
            output_path=output_path,
            log_path=status_path.with_name("worker.log"),
            prompt_style=args.prompt_style,
            id_field=args.id_field,
        )
        status = {
            "status": "success",
            "gpu_id": args.gpu_id,
            "framework": args.framework,
            "stats": stats,
        }
        write_json(status_path, status)
        return 0
    except Exception as exc:
        status = {
            "status": "error",
            "gpu_id": args.gpu_id,
            "framework": args.framework,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "stats": {"rows_processed": 0, "input_tokens": 0, "output_tokens": 0},
        }
        write_json(status_path, status)
        return 1


if __name__ == "__main__":
    sys.exit(main())
