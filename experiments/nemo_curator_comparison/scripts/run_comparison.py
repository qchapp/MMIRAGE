#!/usr/bin/env python3
"""Run balanced AnonLib vs NeMo Curator/Data Designer repetitions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_ORDER = ["anonlib", "nemo", "nemo", "anonlib", "anonlib", "nemo"]
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
EXPERIMENT_DIR = Path("experiments/nemo_curator_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-jsonl", default="experiments/nemo_curator_comparison/workload/chartqa/chartqa_subset.jsonl")
    parser.add_argument("--image-base-path", default="experiments/nemo_curator_comparison/workload/chartqa")
    parser.add_argument("--output-root", default="experiments/nemo_curator_comparison/results")
    parser.add_argument("--rows", type=int, default=None, help="Optional sanity label; input rows are not truncated here.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--anonlib-start-rep", type=int, default=1)
    parser.add_argument("--nemo-start-rep", type=int, default=1)
    parser.add_argument("--order", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--anonlib-python", default=sys.executable, help="Interpreter with the anonlib package (default: this interpreter).")
    parser.add_argument("--nemo-python", default=sys.executable, help="Interpreter with nemo_curator/data_designer (default: this interpreter).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing per-run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    parser.add_argument("--mock", action="store_true", help="Use a local mock OpenAI-compatible server for tiny smoke tests.")
    return parser.parse_args()


class GpuPoller:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)
        if not self.samples:
            return {"mean": None, "min": None, "max": None, "samples": 0}
        return {
            "mean": sum(self.samples) / len(self.samples),
            "min": min(self.samples),
            "max": max(self.samples),
            "samples": len(self.samples),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if values:
                    self.samples.append(sum(values) / len(values))
            except Exception:
                return
            self._stop.wait(self.interval_seconds)


def capture_environment(output_root: Path, args: argparse.Namespace) -> None:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "cwd": str(Path.cwd()),
    }
    commands = {
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
    }
    for key, command in commands.items():
        if shutil.which(command[0]) is None:
            env[key] = None
            continue
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        env[key] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")


def framework_command(framework: str, rep: int, args: argparse.Namespace, run_dir: Path) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "CHARTQA_MODEL_PATH": args.model,
            "CHARTQA_OPENAI_BASE_URL": args.base_url,
            "ANONLIB_CHARTQA_MODEL_PATH": args.model,
            "ANONLIB_CHARTQA_OPENAI_BASE_URL": args.base_url,
            "ANONLIB_CHARTQA_INPUT_JSONL": str(Path(args.workload_jsonl).resolve()),
            "ANONLIB_CHARTQA_IMAGE_BASE_PATH": str(Path(args.image_base_path).resolve()),
            "ANONLIB_CHARTQA_OUTPUT_DIR": str((run_dir / "output").resolve()),
            "ANONLIB_CHARTQA_STATE_DIR": str((run_dir / "state").resolve()),
            "ANONLIB_CHARTQA_REPORT_DIR": str((run_dir / "reports").resolve()),
            "ANONLIB_CHARTQA_BATCH_SIZE": str(args.batch_size),
            "ANONLIB_CHARTQA_CONCURRENCY": str(args.concurrency),
            "ANONLIB_CHARTQA_MAX_TOKENS": str(args.max_tokens),
            "ANONLIB_CHARTQA_MAX_RUNNING_REQUESTS": str(max(args.concurrency, args.batch_size, 64)),
            "HF_HOME": env.get("HF_HOME", str((Path.home() / "hf").resolve())),
            "NEMO_TELEMETRY_ENABLED": "false",
        }
    )
    if framework == "anonlib":
        return [
            args.anonlib_python,
            str(EXPERIMENT_DIR / "scripts/run_anonlib_with_openai_vision_endpoint.py"),
            "--config",
            str(EXPERIMENT_DIR / "configs/anonlib_chartqa.yaml"),
            "--summary-json",
            str(run_dir / "run_summary.json"),
        ], env
    if framework == "nemo":
        return [
            args.nemo_python,
            str(EXPERIMENT_DIR / "scripts/run_nemo_curator_pipeline.py"),
            "--input-jsonl",
            str(Path(args.workload_jsonl).resolve()),
            "--image-base-path",
            str(Path(args.image_base_path).resolve()),
            "--output-dir",
            str((run_dir / "output").resolve()),
            "--summary-json",
            str(run_dir / "run_summary.json"),
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--max-tokens",
            str(args.max_tokens),
            "--max-parallel-requests",
            str(args.concurrency),
        ], env
    raise ValueError(f"Unknown framework {framework!r}")


def run_one(framework: str, rep: int, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    run_dir = output_root / f"{framework}_rep{rep}"
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing run directory {run_dir}. Use --overwrite or a later start rep.")
        shutil.rmtree(run_dir)
    cmd, env = framework_command(framework, rep, args, run_dir)
    manifest = {"framework": framework, "rep": rep, "command": cmd, "run_dir": str(run_dir), "env_overrides": {k: env[k] for k in env if k.startswith(("CHARTQA_", "ANONLIB_CHARTQA_", "NEMO_"))}}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return {"framework": framework, "rep": rep, "returncode": None, "dry_run": True}
    run_dir.mkdir(parents=True)
    (run_dir / "command.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    poller = GpuPoller()
    poller.start()
    started = time.perf_counter()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(cmd, env=env, text=True, stdout=stdout, stderr=stderr, check=False)
    gpu = poller.stop()
    elapsed = time.perf_counter() - started
    summary = {"framework": framework, "rep": rep, "returncode": result.returncode, "launcher_wall_seconds": elapsed, "gpu_utilization": gpu}
    (run_dir / "launcher_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    capture_environment(output_root, args)
    requested_order = [item.strip() for item in args.order.split(",") if item.strip()]
    starts = {"anonlib": args.anonlib_start_rep, "nemo": args.nemo_start_rep}
    counts = {"anonlib": 0, "nemo": 0}
    results = []
    for framework in requested_order:
        if counts[framework] >= args.repetitions:
            continue
        counts[framework] += 1
        results.append(run_one(framework, starts[framework] + counts[framework] - 1, args, output_root))
    (output_root / "run_comparison_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    failures = [r for r in results if r.get("returncode") not in (0, None)]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
