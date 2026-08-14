#!/usr/bin/env python3
"""Run and aggregate a single-node ANONLIB multi-GPU strong-scaling point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUN_FIELDS = [
    "gpu_count",
    "repetition",
    "processed_rows",
    "total_input_tokens",
    "total_output_tokens",
    "end_to_end_wall_seconds",
    "model_loading_seconds",
    "model_loading_seconds_sum",
    "aggregate_output_tok_s",
    "output_tok_s_per_gpu",
    "rows_s",
    "steady_state_output_tok_s",
    "steady_state_output_tok_s_per_gpu",
    "steady_state_rows_s",
    "mean_gpu_utilization",
    "max_shard_runtime_seconds",
    "max_shard_inference_runtime_seconds",
    "successful_shards",
    "failed_shards",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--semantic-recipe", default=None)
    parser.add_argument("--workload-jsonl", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--visible-gpus", default=None, help="Comma-separated physical GPU IDs")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repetition-start", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_output(command: List[str], timeout: int = 30) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return (result.stderr.strip() or result.stdout.strip()) or None
    except Exception as exc:
        return str(exc)


def package_version(package: str) -> Optional[str]:
    code = (
        "import importlib.metadata as m\n"
        f"print(m.version('{package}'))\n"
    )
    return command_output([sys.executable, "-c", code])


def environment_metadata() -> Dict[str, Any]:
    torch_info = command_output(
        [
            sys.executable,
            "-c",
            "import torch; print({'version': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})",
        ]
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "single-node multi-GPU data-parallel ANONLIB scaling; not multi-node scaling",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "packages": {
            "datasets": package_version("datasets"),
            "huggingface_hub": package_version("huggingface-hub"),
            "sglang": package_version("sglang"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
        },
        "torch_runtime": torch_info,
        "nvidia_smi_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=10,
        ),
        "nvidia_smi": command_output(["nvidia-smi"], timeout=10),
    }


def resolve_path(raw: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_path(args.execution_config, Path.cwd())
    cfg = read_yaml(config_path)
    cfg["execution_config"] = str(config_path)
    if args.semantic_recipe is not None:
        cfg["semantic_recipe"] = args.semantic_recipe
    if args.workload_jsonl is not None:
        cfg["workload_jsonl"] = args.workload_jsonl
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.gpu_count is not None:
        cfg["gpu_count"] = args.gpu_count
    if args.visible_gpus is not None:
        cfg["visible_gpus"] = [int(item.strip()) for item in args.visible_gpus.split(",") if item.strip()]
    if args.repetitions is not None:
        cfg["repetitions"] = args.repetitions
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    return cfg


def concrete_anonlib_config(
    semantic_recipe: Path,
    workload_jsonl: Path,
    state_dir: Path,
    output_dir: Path,
    num_shards: int,
    shard_id: int,
    batch_size: int,
) -> Dict[str, Any]:
    recipe = read_yaml(semantic_recipe)
    recipe["loading_params"]["state_dir"] = str(state_dir)
    recipe["loading_params"]["datasets"][0]["path"] = str(workload_jsonl)
    recipe["loading_params"]["datasets"][0]["output_dir"] = str(output_dir)
    recipe["loading_params"]["num_shards"] = num_shards
    recipe["loading_params"]["shard_id"] = shard_id
    recipe["loading_params"]["batch_size"] = batch_size
    recipe.setdefault("execution_params", {})
    recipe["execution_params"].update({"mode": "local", "retry": False, "merge": False, "max_retries": 0})
    return recipe


def write_anonlib_config(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def launch_shards(
    run_dir: Path,
    semantic_recipe: Path,
    workload_jsonl: Path,
    gpu_ids: List[int],
    batch_size: int,
    extra_env: Dict[str, str],
    dry_run: bool,
) -> Dict[str, Any]:
    state_dir = run_dir / "state"
    output_dir = run_dir / "output"
    configs_dir = run_dir / "configs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        shard_config = concrete_anonlib_config(
            semantic_recipe=semantic_recipe,
            workload_jsonl=workload_jsonl,
            state_dir=state_dir,
            output_dir=output_dir,
            num_shards=len(gpu_ids),
            shard_id=shard_id,
            batch_size=batch_size,
        )
        config_path = configs_dir / f"shard_{shard_id}.yaml"
        write_anonlib_config(config_path, shard_config)
        command = [sys.executable, "-m", "anonlib.shard_process", "--config", str(config_path)]
        commands.append({"shard_id": shard_id, "gpu_id": gpu_id, "command": command, "config": str(config_path)})

    if dry_run:
        return {"dry_run": True, "commands": commands}

    processes = []
    source_path = str(PROJECT_ROOT / "src")
    for item in commands:
        env = os.environ.copy()
        env.update(extra_env)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{source_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else source_path
        env["CUDA_VISIBLE_DEVICES"] = str(item["gpu_id"])
        env["ANONLIB_COLLECT_STATS"] = "1"
        log_path = logs_dir / f"shard_{item['shard_id']}.log"
        log_handle = log_path.open("wb")
        proc = subprocess.Popen(
            item["command"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((proc, log_handle, log_path, item))

    failures = []
    try:
        for proc, log_handle, log_path, item in processes:
            returncode = proc.wait()
            log_handle.close()
            if returncode != 0:
                failures.append({"shard_id": item["shard_id"], "gpu_id": item["gpu_id"], "returncode": returncode, "log": str(log_path)})
    except KeyboardInterrupt:
        for proc, _, _, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        raise

    return {"dry_run": False, "commands": commands, "failures": failures}


def read_status(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "missing", "error": str(exc)}


def weighted_mean(values: Iterable[tuple[Optional[float], Optional[int]]]) -> Optional[float]:
    total_weight = 0
    total = 0.0
    for value, weight in values:
        if value is None:
            continue
        w = int(weight or 1)
        total += float(value) * w
        total_weight += w
    if total_weight == 0:
        return None
    return round(total / total_weight, 6)


def aggregate_repetition(run_dir: Path, gpu_count: int, repetition: int, wall_seconds: float) -> Dict[str, Any]:
    statuses = [read_status(run_dir / "state" / f"shard_{idx}" / "status.json") for idx in range(gpu_count)]
    successful = [status for status in statuses if status.get("status") == "success"]
    failed = [idx for idx, status in enumerate(statuses) if status.get("status") != "success"]
    stats = [status.get("stats") or {} for status in successful]

    rows = sum(int(item.get("rows_processed") or 0) for item in stats)
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in stats)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in stats)
    load_times = [float(item.get("model_load_seconds") or 0.0) for item in stats]
    shard_runtimes = [float(item.get("runtime_seconds") or 0.0) for item in stats]
    inference_runtimes = [float(item.get("inference_runtime_seconds") or 0.0) for item in stats]
    model_loading_seconds = max(load_times) if load_times else 0.0
    model_loading_seconds_sum = sum(load_times)
    steady_wall = max(0.0, wall_seconds - model_loading_seconds)
    mean_gpu_util = weighted_mean((item.get("gpu_util_mean"), item.get("gpu_util_samples")) for item in stats)

    row = {
        "gpu_count": gpu_count,
        "repetition": repetition,
        "processed_rows": rows,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "end_to_end_wall_seconds": round(wall_seconds, 6),
        "model_loading_seconds": round(model_loading_seconds, 6),
        "model_loading_seconds_sum": round(model_loading_seconds_sum, 6),
        "aggregate_output_tok_s": round(output_tokens / wall_seconds, 6) if wall_seconds > 0 else None,
        "output_tok_s_per_gpu": round(output_tokens / (wall_seconds * gpu_count), 6) if wall_seconds > 0 else None,
        "rows_s": round(rows / wall_seconds, 6) if wall_seconds > 0 else None,
        "steady_state_output_tok_s": round(output_tokens / steady_wall, 6) if steady_wall > 0 else None,
        "steady_state_output_tok_s_per_gpu": round(output_tokens / (steady_wall * gpu_count), 6) if steady_wall > 0 else None,
        "steady_state_rows_s": round(rows / steady_wall, 6) if steady_wall > 0 else None,
        "mean_gpu_utilization": mean_gpu_util,
        "max_shard_runtime_seconds": round(max(shard_runtimes), 6) if shard_runtimes else None,
        "max_shard_inference_runtime_seconds": round(max(inference_runtimes), 6) if inference_runtimes else None,
        "successful_shards": len(successful),
        "failed_shards": ";".join(str(idx) for idx in failed),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "rep_summary.json", {"row": row, "statuses": statuses})
    return row


def collect_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "runs").glob("gpu_*/rep_*/rep_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload["row"])
    rows.sort(key=lambda item: (int(item["gpu_count"]), int(item["repetition"])))
    return rows


def write_raw_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in RUN_FIELDS})


def mean_std(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value not in (None, "")]
    if not clean:
        return {"mean": None, "std": None}
    return {
        "mean": round(statistics.mean(clean), 6),
        "std": round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0,
    }


def build_summary(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    metric_names = [
        "processed_rows",
        "total_input_tokens",
        "total_output_tokens",
        "end_to_end_wall_seconds",
        "model_loading_seconds",
        "aggregate_output_tok_s",
        "output_tok_s_per_gpu",
        "rows_s",
        "steady_state_output_tok_s",
        "steady_state_output_tok_s_per_gpu",
        "steady_state_rows_s",
        "mean_gpu_utilization",
    ]
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["gpu_count"]), []).append(row)

    summary_rows = []
    baseline = None
    if 1 in grouped:
        baseline = mean_std(row.get("aggregate_output_tok_s") for row in grouped[1])["mean"]

    for gpu_count in sorted(grouped):
        group = grouped[gpu_count]
        metrics = {metric: mean_std(row.get(metric) for row in group) for metric in metric_names}
        throughput_mean = metrics["aggregate_output_tok_s"]["mean"]
        speedup = round(throughput_mean / baseline, 6) if baseline and throughput_mean else None
        efficiency = round(speedup / gpu_count, 6) if speedup is not None else None
        summary_rows.append(
            {
                "gpu_count": gpu_count,
                "repetitions": len(group),
                "metrics": metrics,
                "speedup_vs_1gpu": speedup,
                "parallel_efficiency": efficiency,
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "single-node data-parallel strong scaling; do not infer multi-node scaling",
        "throughput_for_speedup": "aggregate_output_tok_s = total_output_tokens / end_to_end_wall_seconds",
        "metric_definitions": {
            "model_loading_seconds": "maximum shard model-loading time in a repetition, because shard workers load concurrently",
            "mean_gpu_utilization": "sample-weighted mean across shard-local nvidia-smi utilization pollers",
            "parallel_efficiency": "speedup_vs_1gpu / gpu_count",
        },
        "metadata": metadata,
        "raw_repetitions": rows,
        "summary": summary_rows,
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    fields = [
        "gpu_count",
        "repetitions",
        "aggregate_output_tok_s_mean",
        "aggregate_output_tok_s_std",
        "output_tok_s_per_gpu_mean",
        "output_tok_s_per_gpu_std",
        "rows_s_mean",
        "rows_s_std",
        "end_to_end_wall_seconds_mean",
        "end_to_end_wall_seconds_std",
        "model_loading_seconds_mean",
        "model_loading_seconds_std",
        "mean_gpu_utilization_mean",
        "mean_gpu_utilization_std",
        "speedup_vs_1gpu",
        "parallel_efficiency",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary["summary"]:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "gpu_count": item["gpu_count"],
                    "repetitions": item["repetitions"],
                    "aggregate_output_tok_s_mean": metrics["aggregate_output_tok_s"]["mean"],
                    "aggregate_output_tok_s_std": metrics["aggregate_output_tok_s"]["std"],
                    "output_tok_s_per_gpu_mean": metrics["output_tok_s_per_gpu"]["mean"],
                    "output_tok_s_per_gpu_std": metrics["output_tok_s_per_gpu"]["std"],
                    "rows_s_mean": metrics["rows_s"]["mean"],
                    "rows_s_std": metrics["rows_s"]["std"],
                    "end_to_end_wall_seconds_mean": metrics["end_to_end_wall_seconds"]["mean"],
                    "end_to_end_wall_seconds_std": metrics["end_to_end_wall_seconds"]["std"],
                    "model_loading_seconds_mean": metrics["model_loading_seconds"]["mean"],
                    "model_loading_seconds_std": metrics["model_loading_seconds"]["std"],
                    "mean_gpu_utilization_mean": metrics["mean_gpu_utilization"]["mean"],
                    "mean_gpu_utilization_std": metrics["mean_gpu_utilization"]["std"],
                    "speedup_vs_1gpu": item["speedup_vs_1gpu"],
                    "parallel_efficiency": item["parallel_efficiency"],
                }
            )


def latex_cell(metric: Dict[str, Optional[float]]) -> str:
    if metric["mean"] is None:
        return "--"
    return f"{metric['mean']:.2f} $\\pm$ {metric['std']:.2f}"


def write_latex_table(path: Path, summary: Dict[str, Any]) -> None:
    line_end = "\\\\"
    lines = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        f"GPUs & Output tok/s & Output tok/s/GPU & Rows/s & Speedup & Efficiency {line_end}",
        "\\midrule",
    ]
    for item in summary["summary"]:
        metrics = item["metrics"]
        speedup = "--" if item["speedup_vs_1gpu"] is None else f"{item['speedup_vs_1gpu']:.2f}"
        efficiency = "--" if item["parallel_efficiency"] is None else f"{item['parallel_efficiency']:.2f}"
        lines.append(
            f"{item['gpu_count']} & {latex_cell(metrics['aggregate_output_tok_s'])} & "
            f"{latex_cell(metrics['output_tok_s_per_gpu'])} & {latex_cell(metrics['rows_s'])} & "
            f"{speedup} & {efficiency} {line_end}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.append("% Single-node multi-GPU strong scaling only; do not infer multi-node scaling.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_outputs(output_root: Path, metadata: Dict[str, Any]) -> None:
    rows = collect_rows(output_root)
    write_raw_results(output_root / "raw_results.csv", rows)
    summary = build_summary(rows, metadata)
    write_json(output_root / "summary.json", summary)
    write_summary_csv(output_root / "summary.csv", summary)
    write_latex_table(output_root / "latex_table.txt", summary)
    plot_script = SCRIPTS_DIR / "plot.py"
    if plot_script.exists() and rows:
        subprocess.run(
            [sys.executable, str(plot_script), "--summary-csv", str(output_root / "summary.csv"), "--output-dir", str(output_root)],
            check=False,
        )


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args)
    output_root = resolve_path(cfg.get("output_root", str(EXPERIMENT_DIR)))
    metadata_path = output_root / "experiment_metadata.json"
    metadata = environment_metadata()

    if args.aggregate_only:
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else metadata
        write_all_outputs(output_root, previous_metadata)
        return

    gpu_count = int(cfg["gpu_count"])
    repetitions = int(cfg.get("repetitions", 3))
    batch_size = int(cfg.get("batch_size", 256))
    visible_gpus = [int(item) for item in cfg.get("visible_gpus", list(range(gpu_count)))]
    if gpu_count not in (1, 2, 4):
        raise ValueError("This experiment is restricted to gpu_count in {1, 2, 4}")
    if len(visible_gpus) < gpu_count:
        raise ValueError(f"Need at least {gpu_count} visible_gpus, got {visible_gpus}")
    gpu_ids = visible_gpus[:gpu_count]

    semantic_recipe = resolve_path(cfg["semantic_recipe"])
    workload_jsonl = resolve_path(cfg["workload_jsonl"])
    if not semantic_recipe.exists():
        raise FileNotFoundError(semantic_recipe)
    if not workload_jsonl.exists() and not args.dry_run:
        raise FileNotFoundError(workload_jsonl)

    extra_env = {str(k): str(v) for k, v in (cfg.get("environment") or {}).items()}
    workload_metadata_path = workload_jsonl.parent / "metadata.json"
    workload_metadata = json.loads(workload_metadata_path.read_text(encoding="utf-8")) if workload_metadata_path.exists() else {}
    metadata.update(
        {
            "execution_config": cfg.get("execution_config"),
            "semantic_recipe": str(semantic_recipe),
            "workload_jsonl": str(workload_jsonl),
            "workload_metadata": workload_metadata,
            "gpu_count_requested": gpu_count,
            "visible_gpus_used": gpu_ids,
            "batch_size": batch_size,
            "repetitions": repetitions,
        }
    )
    write_json(metadata_path, metadata)

    if args.dry_run:
        dry_run_dir = output_root / "runs" / f"gpu_{gpu_count}" / "dry_run"
        payload = launch_shards(
            run_dir=dry_run_dir,
            semantic_recipe=semantic_recipe,
            workload_jsonl=workload_jsonl,
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            extra_env=extra_env,
            dry_run=True,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        write_all_outputs(output_root, metadata)
        return

    for repetition in range(args.repetition_start, args.repetition_start + repetitions):
        run_dir = output_root / "runs" / f"gpu_{gpu_count}" / f"rep_{repetition}"
        if run_dir.exists():
            if not args.overwrite:
                raise RuntimeError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run_manifest.json", metadata)
        started = time.monotonic()
        result = launch_shards(
            run_dir=run_dir,
            semantic_recipe=semantic_recipe,
            workload_jsonl=workload_jsonl,
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            extra_env=extra_env,
            dry_run=False,
        )
        wall_seconds = time.monotonic() - started
        if result["failures"]:
            write_json(run_dir / "failed_launch.json", {"wall_seconds": wall_seconds, **result})
            raise RuntimeError(f"One or more shards failed. See logs under {run_dir / 'logs'}")
        aggregate_repetition(run_dir, gpu_count, repetition, wall_seconds)
        write_all_outputs(output_root, metadata)

    write_all_outputs(output_root, metadata)
    print(json.dumps(json.loads((output_root / "summary.json").read_text(encoding="utf-8")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
