#!/usr/bin/env python3
"""Run raw SGLang and MMIRAGE-over-SGLang overhead repetitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_CONFIG = "configs/config_overhead_s1k_sglang.yaml"
DEFAULT_SERVER_EXTRA_ARGS = [
    "--tp-size",
    "1",
    "--trust-remote-code",
    "--disable-custom-all-reduce",
    "--max-running-requests",
    "1000",
]


class GpuPoller:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: List[float] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 2)
        if not self.samples:
            return {"mean": None, "min": None, "max": None, "samples": 0}
        return {
            "mean": round(sum(self.samples) / len(self.samples), 1),
            "min": min(self.samples),
            "max": max(self.samples),
            "samples": len(self.samples),
        }

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            value = self._query()
            if value is not None:
                self.samples.append(value)

    @staticmethod
    def _query() -> Optional[float]:
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible and visible.lower() not in ("all", "nodevfiles"):
                command.append(f"--id={visible}")
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return None
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            return sum(values) / len(values) if values else None
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Physical GPU index to pin via CUDA_VISIBLE_DEVICES for the whole run "
        "(server and clients). Unset on single-GPU nodes.",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-limit", type=int, default=None)
    parser.add_argument("--server-command-json", default=None)
    parser.add_argument("--server-extra-args-json", default=None)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_output(command: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        return result.stderr.strip() or result.stdout.strip()
    except Exception:
        return None


def nvidia_metadata() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        names = sorted({line.split(",")[0].strip() for line in lines})
        return {
            "nvidia_smi": result.stdout.strip(),
            "nvidia_smi_error": result.stderr.strip(),
            "gpu_names": names,
        }
    except Exception as exc:
        return {"nvidia_smi": None, "nvidia_smi_error": str(exc), "gpu_names": []}


def server_command(args: argparse.Namespace) -> List[str]:
    if args.server_command_json:
        command = json.loads(args.server_command_json)
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("--server-command-json must be a JSON string list")
        return command
    extra_args = DEFAULT_SERVER_EXTRA_ARGS
    if args.server_extra_args_json:
        extra_args = json.loads(args.server_extra_args_json)
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError("--server-extra-args-json must be a JSON string list")
    executable = shutil.which("sglang")
    if executable:
        return [
            executable,
            "serve",
            "--model-path",
            args.model_path,
            "--host",
            args.host,
            "--port",
            str(args.port),
            *extra_args,
        ]
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        *extra_args,
    ]


def endpoint_ready(base_url: str) -> bool:
    for suffix in ("/v1/models", "/models", "/health"):
        try:
            with urllib.request.urlopen(f"{base_url}{suffix}", timeout=5) as response:
                response.read()
                return True
        except Exception:
            pass
    return False


def wait_for_server(proc: subprocess.Popen[bytes], base_url: str, timeout_seconds: int) -> float:
    start = time.monotonic()
    deadline = start + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {proc.returncode}")
        if endpoint_ready(base_url):
            return time.monotonic() - start
        time.sleep(2)
    raise RuntimeError(f"SGLang server did not become ready in {timeout_seconds}s")


def start_server(args: argparse.Namespace, run_dir: Path) -> tuple[subprocess.Popen[bytes], float, str, List[str]]:
    base_url = f"http://{args.host}:{args.port}"
    command = server_command(args)
    log_handle = (run_dir / "sglang_server.log").open("wb")
    proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    setattr(proc, "_mmirage_log_handle", log_handle)
    try:
        startup_seconds = wait_for_server(proc, base_url, args.startup_timeout_seconds)
    except Exception:
        stop_server(proc)
        raise
    return proc, startup_seconds, base_url, command


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    handle = getattr(proc, "_mmirage_log_handle", None)
    if handle is not None:
        handle.close()


def timed_subprocess(command: List[str], env: Optional[Dict[str, str]] = None) -> float:
    start = time.monotonic()
    subprocess.run(command, env=env, check=True)
    return time.monotonic() - start


def raw_command(args: argparse.Namespace, prompts_jsonl: Path, base_url: str, output_jsonl: Path, summary_json: Path, limit: Optional[int]) -> List[str]:
    command = [
        sys.executable,
        "scripts/benchmarks/raw_sglang_client.py",
        "--prompts-jsonl",
        str(prompts_jsonl),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-json",
        str(summary_json),
        "--base-url",
        base_url,
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--concurrency",
        str(args.concurrency),
        "--gpu-count",
        str(args.gpu_count),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.model_revision:
        command.extend(["--tokenizer-revision", args.model_revision])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def run_raw(args: argparse.Namespace, prompts_jsonl: Path, base_url: str, run_dir: Path, limit: Optional[int] = None) -> tuple[float, Dict[str, Any]]:
    poller = GpuPoller()
    poller.start()
    wall = timed_subprocess(raw_command(args, prompts_jsonl, base_url, run_dir / "outputs.jsonl", run_dir / "summary.json", limit))
    gpu = poller.stop()
    return wall, {**read_json(run_dir / "summary.json"), "gpu_utilization_pct": gpu["mean"]}


def run_mmirage(args: argparse.Namespace, prompts_jsonl: Path, base_url: str, run_dir: Path) -> float:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    source_path = str(project_root() / "src")
    env["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_path
    )
    env.update(
        {
            "MMIRAGE_OVERHEAD_MODEL_PATH": args.model_path,
            "MMIRAGE_OVERHEAD_SGLANG_BASE_URL": base_url,
            "MMIRAGE_OVERHEAD_STATE_DIR": str(run_dir / "state"),
            "MMIRAGE_OVERHEAD_PROMPTS_JSONL": str(prompts_jsonl),
            "MMIRAGE_OVERHEAD_OUTPUT_DIR": str(run_dir / "output"),
            "MMIRAGE_OVERHEAD_CONCURRENCY": str(args.concurrency),
            "MMIRAGE_OVERHEAD_TIMEOUT_SECONDS": "900",
        }
    )
    return timed_subprocess(
        [sys.executable, "scripts/benchmarks/run_mmirage_with_sglang_endpoint.py", "--config", args.config, "--summary-json", str(run_dir / "wrapper_summary.json")],
        env=env,
    )


def mmirage_metrics(run_dir: Path, full_wall_seconds: float) -> Dict[str, Any]:
    status = read_json(run_dir / "state" / "shard_0" / "status.json")
    stats = status.get("stats") or {}
    wrapper = read_json(run_dir / "wrapper_summary.json")
    success_count = int(stats.get("rows_processed") or 0)
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(run_dir / "output" / "shard_0"))
        success_count = sum(
            1 for answer in ds["answer"] if isinstance(answer, str) and answer.strip()
        )
    except Exception:
        pass
    return {
        "generation_wall_seconds": stats.get("inference_runtime_seconds"),
        "full_end_to_end_wall_seconds": round(full_wall_seconds, 6),
        "rows": stats.get("rows_processed"),
        "rows_per_second": stats.get("throughput_rows_per_sec"),
        "total_output_tokens": stats.get("output_tokens"),
        "output_tokens_per_second_per_gpu": stats.get("tokens_per_sec_per_gpu"),
        "gpu_utilization_pct": stats.get("gpu_util_mean"),
        "success_count": success_count,
        "wrapper_full_wall_seconds": wrapper.get("full_end_to_end_wall_seconds"),
    }


def write_raw_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "repetition",
        "path",
        "output_tokens_per_second_per_gpu",
        "rows_per_second",
        "total_output_tokens",
        "gpu_utilization_pct",
        "model_loading_seconds",
        "generation_wall_seconds",
        "full_end_to_end_wall_seconds",
        "success_count",
        "server_command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def mean_std(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.mean(clean), 6), "std": round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0}


def build_summary(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    grouped = {"raw_sglang": [], "mmirage_sglang": []}
    for row in rows:
        grouped[row["path"]].append(row)
    metric_names = [
        "output_tokens_per_second_per_gpu",
        "rows_per_second",
        "total_output_tokens",
        "gpu_utilization_pct",
        "model_loading_seconds",
        "generation_wall_seconds",
        "full_end_to_end_wall_seconds",
        "success_count",
    ]
    metrics = {
        path_name: {name: mean_std(row.get(name) for row in path_rows) for name in metric_names}
        for path_name, path_rows in grouped.items()
    }
    raw_mean = metrics["raw_sglang"]["output_tokens_per_second_per_gpu"]["mean"]
    mm_mean = metrics["mmirage_sglang"]["output_tokens_per_second_per_gpu"]["mean"]
    retention = round(mm_mean / raw_mean, 6) if raw_mean and mm_mean else None
    overhead = round(1.0 - retention, 6) if retention is not None else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "end-to-end empirical overhead estimate, not a pure CPU orchestration microbenchmark",
        "metrics": metrics,
        "throughput_retention": retention,
        "relative_orchestration_overhead": overhead,
        "metadata": metadata,
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "metric", "mean", "std"])
        for path_name, metrics in summary["metrics"].items():
            for metric_name, value in metrics.items():
                writer.writerow([path_name, metric_name, value["mean"], value["std"]])
        writer.writerow(["comparison", "throughput_retention", summary["throughput_retention"], ""])
        writer.writerow(["comparison", "relative_orchestration_overhead", summary["relative_orchestration_overhead"], ""])


def latex_cell(value: Dict[str, Optional[float]]) -> str:
    if value["mean"] is None:
        return "--"
    return f"{value['mean']:.2f} $\\pm$ {value['std']:.2f}"


def write_latex_table(path: Path, summary: Dict[str, Any]) -> None:
    raw = summary["metrics"]["raw_sglang"]
    mm = summary["metrics"]["mmirage_sglang"]
    line_end = "\\\\"
    table = "\n".join(
        [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            f"Path & Output tok/s/GPU & Rows/s & Gen. wall (s) & Success count {line_end}",
            "\\midrule",
            f"Raw SGLang & {latex_cell(raw['output_tokens_per_second_per_gpu'])} & {latex_cell(raw['rows_per_second'])} & {latex_cell(raw['generation_wall_seconds'])} & {latex_cell(raw['success_count'])} {line_end}",
            f"MMIRAGE over SGLang & {latex_cell(mm['output_tokens_per_second_per_gpu'])} & {latex_cell(mm['rows_per_second'])} & {latex_cell(mm['generation_wall_seconds'])} & {latex_cell(mm['success_count'])} {line_end}",
            "\\bottomrule",
            "\\end{tabular}",
            f"% Throughput retention: {summary['throughput_retention']}",
            f"% Relative orchestration overhead: {summary['relative_orchestration_overhead']}",
        ]
    )
    path.write_text(table + "\n", encoding="utf-8")


def write_plot_script(path: Path) -> None:
    path.write_text(
        '''#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

labels = ["Raw SGLang", "MMIRAGE"]
keys = ["raw_sglang", "mmirage_sglang"]
data = [[float(row["output_tokens_per_second_per_gpu"]) for row in rows if row["path"] == key] for key in keys]
plt.boxplot(data, labels=labels)
plt.ylabel("Output tok/s/GPU")
plt.title("Raw SGLang vs MMIRAGE over SGLang")
plt.tight_layout()
plt.savefig(root / "throughput_boxplot.png", dpi=200)
''',
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    os.chdir(project_root())
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    workload_dir = Path(args.workload_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_jsonl = workload_dir / "prompts.jsonl"
    warmup_jsonl = workload_dir / "warmup_prompts.jsonl"
    workload_metadata = read_json(workload_dir / "metadata.json")
    command = server_command(args)
    gpu_info = nvidia_metadata()
    gpu_label = "/".join(gpu_info.get("gpu_names") or ["unknown"])
    metadata = {
        "mmirage_commit": command_output(["git", "rev-parse", "HEAD"]),
        "sglang_version": command_output([sys.executable, "-c", "import sglang; print(getattr(sglang, '__version__', 'unknown'))"]),
        "qwen_model_revision": workload_metadata.get("model_revision_resolved"),
        "dataset_revision": workload_metadata.get("dataset_revision_resolved"),
        "gpu_cuda_driver": gpu_info,
        "server_arguments": command,
        "generation_settings": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "tokenizer": args.model_path,
        "concurrency": args.concurrency,
        "workload": workload_metadata,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": (
            f"one {gpu_label} target, Run:ai pod, no Slurm, same pod for both paths; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        ),
    }
    write_json(output_dir / "experiment_metadata.json", metadata)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "server_command": command, "metadata": metadata}, indent=2, sort_keys=True))
        return

    rows: List[Dict[str, Any]] = []
    for rep in range(1, args.repetitions + 1):
        for path_name in ("raw_sglang", "mmirage_sglang"):
            run_dir = output_dir / f"rep_{rep}" / path_name
            run_dir.mkdir(parents=True, exist_ok=True)
            proc, startup_seconds, base_url, server_cmd = start_server(args, run_dir)
            warmup_wall = 0.0
            try:
                if warmup_jsonl.exists():
                    warmup_dir = run_dir / "warmup"
                    warmup_dir.mkdir(parents=True, exist_ok=True)
                    if path_name == "raw_sglang":
                        warmup_wall, _ = run_raw(args, warmup_jsonl, base_url, warmup_dir, args.warmup_limit)
                    else:
                        warmup_wall = run_mmirage(args, warmup_jsonl, base_url, warmup_dir)
                if path_name == "raw_sglang":
                    measured_wall, metrics = run_raw(args, prompts_jsonl, base_url, run_dir, None)
                    row = {
                        "repetition": rep,
                        "path": path_name,
                        "output_tokens_per_second_per_gpu": metrics.get("output_tokens_per_second_per_gpu"),
                        "rows_per_second": metrics.get("rows_per_second"),
                        "total_output_tokens": metrics.get("total_output_tokens"),
                        "gpu_utilization_pct": metrics.get("gpu_utilization_pct"),
                        "model_loading_seconds": round(startup_seconds, 6),
                        "generation_wall_seconds": metrics.get("generation_wall_seconds"),
                        "full_end_to_end_wall_seconds": round(startup_seconds + warmup_wall + measured_wall, 6),
                        "success_count": metrics.get("success_count"),
                        "server_command": " ".join(server_cmd),
                    }
                else:
                    measured_wall = run_mmirage(args, prompts_jsonl, base_url, run_dir)
                    row = {
                        "repetition": rep,
                        "path": path_name,
                        "model_loading_seconds": round(startup_seconds, 6),
                        "server_command": " ".join(server_cmd),
                        **mmirage_metrics(run_dir, startup_seconds + warmup_wall + measured_wall),
                    }
                rows.append(row)
                write_raw_results_csv(output_dir / "raw_results.csv", rows)
            finally:
                stop_server(proc)

    summary = build_summary(rows, metadata)
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_latex_table(output_dir / "table.tex", summary)
    write_plot_script(output_dir / "plot_throughput.py")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
