#!/usr/bin/env python3
"""Run native text-generation competitor baselines for the scaling experiment.

Splits the workload into one contiguous shard per visible GPU, launches one
framework-native worker subprocess per GPU (pinned via ``CUDA_VISIBLE_DEVICES``),
then aggregates every repetition with the MMIRAGE aggregator and validates the
output contract.

Usage mirrors the pre-existing wrapper contract; each framework wrapper
(``run_<framework>_scaling.py``) invokes this module with a fixed framework.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SHARED_DIR = PROJECT_ROOT / "experiments" / "_shared"
WORKER_SCRIPT = SCRIPTS_DIR / "native_shard_worker.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRAMEWORK_BACKENDS = {
    "datatrove": "DataTrove InferenceRunner with native vLLM backend",
    "nemo_curator": "NeMo Curator native pipeline with local vLLM server",
    "distilabel": "Distilabel native pipeline with local vLLM task",
    "ray_data_llm": "Ray Data LLM native processor with vLLM engine",
    "raw_sglang": "direct SGLang engine/client baseline",
}

OUTPUT_CONTRACT = {
    "required_fields": ["stable_id", "source_index", "prompt_sha256", "prompt_text", "answer"],
    "row_order": "must_match_input_order",
    "validation": [
        "processed_rows_equals_input_rows",
        "stable_id_set_matches_input",
        "no_duplicate_stable_ids",
        "prompt_sha256_preserved",
        "schema_valid_for_every_row",
    ],
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
    parser.add_argument("--concurrency", type=int, default=32, help="Generation batch size per shard worker.")
    parser.add_argument("--prompt-style", choices=["rewrite", "raw", "summarize"], default="rewrite")
    parser.add_argument("--worker-python", default=None, help="Python interpreter for shard workers (default: this interpreter).")
    parser.add_argument("--aggregate-only", action="store_true", help="Only re-aggregate an existing output root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing repetition run directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print the manifest and exit without running.")
    parser.add_argument(
        "--allow-unimplemented",
        action="store_true",
        help="Plan-only mode: write a manifest and exit zero without running (legacy compatibility flag).",
    )
    args = parser.parse_args()
    if framework_override is not None:
        args.framework = framework_override
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shard_ranges(total_rows: int, shards: int) -> list[dict[str, int]]:
    ranges = []
    for shard_id in range(shards):
        start = (total_rows * shard_id) // shards
        end = (total_rows * (shard_id + 1)) // shards
        ranges.append({"shard_id": shard_id, "start_row": start, "end_row": end, "rows": end - start})
    return ranges


def build_manifest(args: argparse.Namespace, planned_only: bool) -> dict[str, Any]:
    workload = Path(args.workload_jsonl)
    if not workload.exists():
        raise FileNotFoundError(workload)
    gpu_ids = [item.strip() for item in args.visible_gpus.split(",") if item.strip()]
    if len(gpu_ids) < args.gpu_count:
        raise ValueError(f"Need {args.gpu_count} visible GPUs, got {gpu_ids}")
    rows = len(read_jsonl(workload))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": args.framework,
        "native_backend": FRAMEWORK_BACKENDS[args.framework],
        "native_mode": True,
        "execution_status": "planned_native_run_not_executed" if planned_only else "implemented_and_runnable",
        "workload_jsonl": str(workload),
        "input_rows": rows,
        "model": args.model,
        "decoding": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "gpu_count": args.gpu_count,
        "visible_gpus": gpu_ids[: args.gpu_count],
        "repetitions": args.repetitions,
        "concurrency": args.concurrency,
        "prompt_style": args.prompt_style,
        "shards": shard_ranges(rows, args.gpu_count),
        "output_contract": OUTPUT_CONTRACT,
    }


def write_rep_inputs(run_dir: Path, rows: list[dict[str, Any]], gpu_count: int) -> list[Path]:
    paths = []
    for spec in shard_ranges(len(rows), gpu_count):
        input_path = run_dir / "state" / f"shard_{spec['shard_id']}" / "input.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("w", encoding="utf-8") as handle:
            for row in rows[spec["start_row"] : spec["end_row"]]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(input_path)
    return paths


def launch_and_wait(args: argparse.Namespace, run_dir: Path, gpu_ids: list[str]) -> dict[str, Any]:
    worker_python = args.worker_python or sys.executable
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{SHARED_DIR}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else f"{PROJECT_ROOT}{os.pathsep}{SHARED_DIR}"
    )

    started = time.perf_counter()
    processes = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        command = [
            worker_python,
            str(WORKER_SCRIPT),
            "--framework",
            args.framework,
            "--input-jsonl",
            str(run_dir / "state" / f"shard_{shard_id}" / "input.jsonl"),
            "--output-jsonl",
            str(run_dir / "state" / f"shard_{shard_id}" / "output.jsonl"),
            "--status-json",
            str(run_dir / "state" / f"shard_{shard_id}" / "status.json"),
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--concurrency",
            str(args.concurrency),
            "--prompt-style",
            args.prompt_style,
            "--gpu-id",
            str(gpu_id),
        ]
        log_path = logs_dir / f"shard_{shard_id}.log"
        log_handle = log_path.open("wb")
        worker_env = env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(
            command,
            env=worker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(
            {"process": process, "log_handle": log_handle, "log_path": log_path, "shard_id": shard_id, "gpu_id": gpu_id}
        )

    failures = []
    try:
        for item in processes:
            returncode = item["process"].wait()
            if returncode != 0:
                failures.append({"shard_id": item["shard_id"], "gpu_id": item["gpu_id"], "returncode": returncode, "log": str(item["log_path"])})
    except KeyboardInterrupt:
        for item in processes:
            if item["process"].poll() is None:
                item["process"].terminate()
        raise
    finally:
        for item in processes:
            item["log_handle"].close()
    wall_seconds = time.perf_counter() - started
    return {"wall_seconds": wall_seconds, "failures": failures}


def merge_shard_outputs(run_dir: Path, input_rows: list[dict[str, Any]], gpu_count: int) -> list[dict[str, Any]]:
    output_rows = []
    for shard_id in range(gpu_count):
        output_path = run_dir / "state" / f"shard_{shard_id}" / "output.jsonl"
        if output_path.exists():
            output_rows.extend(read_jsonl(output_path))
    output_rows.sort(key=lambda row: int(row.get("source_index", -1)))
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    merged_path = run_dir / "output" / "native_competitor_output.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_rows


def validate_output(input_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from experiments._shared.native_frameworks import validate_contract

    return validate_contract(input_rows, output_rows, required_fields=OUTPUT_CONTRACT["required_fields"])


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    def command_output(command: list[str]) -> str | None:
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
            return (result.stderr.strip() or result.stdout.strip()) or None
        except Exception as exc:
            return str(exc)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "single_node_h100_scaling",
        "native_mode": True,
        "framework": args.framework,
        "model": args.model,
        "decoding": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "python": sys.version.replace("\n", " "),
    }


def run_repetitions(args: argparse.Namespace, output_root: Path) -> None:
    from run import aggregate_repetition, write_all_outputs

    workload = Path(args.workload_jsonl)
    input_rows = read_jsonl(workload)
    gpu_ids = [item.strip() for item in args.visible_gpus.split(",") if item.strip()][: args.gpu_count]

    for repetition in range(1, args.repetitions + 1):
        run_dir = output_root / "runs" / f"gpu_{args.gpu_count}" / f"rep_{repetition}"
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"Run directory already exists: {run_dir} (pass --overwrite to replace it)")
            shutil.rmtree(run_dir)
        write_rep_inputs(run_dir, input_rows, args.gpu_count)
        result = launch_and_wait(args, run_dir, gpu_ids)
        if result["failures"]:
            for failure in result["failures"]:
                print(f"[{args.framework}] shard {failure['shard_id']} on GPU {failure['gpu_id']} failed with rc={failure['returncode']}: {failure['log']}", file=sys.stderr)
            raise SystemExit(f"{len(result['failures'])} shard worker(s) failed for {args.framework} gpu={args.gpu_count} rep={repetition}")

        merged = merge_shard_outputs(run_dir, input_rows, args.gpu_count)
        validation = validate_output(input_rows, merged)
        write_json(run_dir / "validation.json", validation)
        aggregate_repetition(run_dir, args.gpu_count, repetition, result["wall_seconds"])
        print(f"[{args.framework}] gpu={args.gpu_count} rep={repetition} wall={result['wall_seconds']:.2f}s rows={len(merged)} validation={'PASS' if validation['valid'] else 'FAIL'}")
        if not validation["valid"]:
            raise SystemExit(f"Output contract validation failed for {args.framework} gpu={args.gpu_count} rep={repetition}: {json.dumps(validation['checks'])}")

    write_all_outputs(output_root, environment_metadata(args))


def main(framework_override: str | None = None) -> None:
    args = parse_args(framework_override)
    planned_only = args.dry_run or args.allow_unimplemented
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, planned_only=planned_only)
    manifest_path = output_root / f"{args.framework}_gpu{args.gpu_count}_native_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if planned_only:
        return
    if args.aggregate_only:
        from run import write_all_outputs

        write_all_outputs(output_root, environment_metadata(args))
        return
    run_repetitions(args, output_root)


if __name__ == "__main__":
    main()
