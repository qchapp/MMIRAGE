#!/usr/bin/env python3
"""Build and execute the publication experiment command plans."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
SCALING = EXPERIMENTS / "scaling"
RECOVERY = EXPERIMENTS / "recovery"
TEXT = EXPERIMENTS / "text_shortening"
VLM = EXPERIMENTS / "vlm_enrichment"

MODEL = "Qwen/Qwen3-4B"
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SCALING_FRAMEWORKS = ["mmirage", "raw_sglang", "datatrove", "nemo_curator"]
RECOVERY_FRAMEWORKS = ["raw_sglang", "datatrove", "nemo_curator", "distilabel", "ray_data_llm"]
RECOVERY_MMIRAGE_CONDITIONS = ["baseline", "fail_1", "fail_4"]
RECOVERY_NATIVE_CONDITIONS = ["fail_1", "fail_4"]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_template(text: str) -> str:
    import re

    return re.sub(r"\{\{\s*[a-zA-Z0-9_]+\s*\}\}|\{\s*[a-zA-Z0-9_]+\s*\}", "{}", text.strip())


def _recipe_prompt(path: Path) -> str:
    for item in _read_yaml(path).get("processing_params", {}).get("outputs", []):
        if isinstance(item.get("prompt"), str):
            return item["prompt"]
    raise RuntimeError(f"No output prompt in {path}")


def verify_templates(stages: Iterable[str]) -> None:
    from experiments._shared import native_frameworks
    from experiments._shared import vlm_runners

    selected = set(stages)
    checks: list[tuple[str, bool]] = []
    if selected & {"scaling", "recovery"}:
        rewrite = _normalize_template(native_frameworks.REWRITE_PROMPT_TEMPLATE)
        checks.append(("scaling recipe == native rewrite template", _normalize_template(_recipe_prompt(SCALING / "configs" / "semantic_recipe.yaml")) == rewrite))
        checks.append(("recovery recipe == native rewrite template", _normalize_template(_recipe_prompt(RECOVERY / "configs" / "mmirage_recovery.yaml")) == rewrite))
    if "text" in selected:
        checks.append(("text recipe == native summarize template", _normalize_template(_recipe_prompt(TEXT / "configs" / "semantic_recipe.yaml")) == _normalize_template(native_frameworks.SUMMARIZE_PROMPT_TEMPLATE)))
    if "vlm" in selected:
        checks.append(("VLM recipe == native VLM template", _normalize_template(_recipe_prompt(VLM / "configs" / "mmirage_vlm.yaml")) == _normalize_template(vlm_runners.VLM_REFORMAT_TEMPLATE)))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"template-check: {'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit("Template equivalence failed: " + ", ".join(failed))


def _python(env_name: str | None = None) -> str:
    if env_name:
        candidate = os.environ.get(env_name)
        if candidate:
            return candidate
    return sys.executable


def _native_python(framework: str) -> str:
    return _python({
        "datatrove": "MMIRAGE_DATATROVE_PYTHON",
        "nemo_curator": "MMIRAGE_NEMO_CURATOR_PYTHON",
        "distilabel": "MMIRAGE_DISTILABEL_PYTHON",
        "ray_data_llm": "MMIRAGE_RAY_DATA_LLM_PYTHON",
    }.get(framework))


def _recovery_root() -> str:
    return os.environ.get("MMIRAGE_RECOVERY_ROOT", str(RECOVERY / "workdir"))


def scaling_plan(hardware: str, repetitions: int, overwrite: bool) -> list[dict[str, Any]]:
    points = [1, 2, 4] if hardware == "h100" else [4]
    hardware_dir = "h100" if hardware == "h100" else "a100"
    units: list[dict[str, Any]] = []
    for gpu_count in points:
        gpus = [str(i) for i in range(gpu_count)]
        for framework in SCALING_FRAMEWORKS:
            output_root = SCALING / "results" / hardware_dir / framework
            if framework == "mmirage":
                command = [
                    sys.executable,
                    str(SCALING / "scripts" / "run.py"),
                    "--execution-config", str(SCALING / "configs" / "execution.yaml"),
                    "--semantic-recipe", str(SCALING / "configs" / "semantic_recipe.yaml"),
                    "--workload-jsonl", str(SCALING / "workload" / "workload.jsonl"),
                    "--output-root", str(output_root),
                    "--gpu-count", str(gpu_count),
                    "--visible-gpus", ",".join(gpus),
                    "--repetitions", str(repetitions),
                ]
            else:
                command = [
                    _native_python(framework),
                    str(SCALING / "scripts" / f"run_{framework}_scaling.py"),
                    "--workload-jsonl", str(SCALING / "workload" / "workload.jsonl"),
                    "--output-root", str(output_root),
                    "--gpu-count", str(gpu_count),
                    "--visible-gpus", ",".join(gpus),
                    "--repetitions", str(repetitions),
                    "--model", MODEL,
                    "--concurrency", "64",
                    "--temperature", "0.0",
                    "--max-new-tokens", "256",
                    "--prompt-style", "rewrite",
                ]
            if overwrite:
                command.append("--overwrite")
            units.append({"stage": "scaling", "hardware": hardware, "framework": framework, "gpu_count": gpu_count, "physical_gpus": gpus, "commands": [command]})
    return units


def recovery_plan(overwrite: bool, repetitions: int) -> list[dict[str, Any]]:
    if repetitions != 3:
        raise SystemExit("Publication recovery requires exactly 3 repetitions")
    shared = _recovery_root()
    gpus = ["0", "1", "2", "3"]
    units: list[dict[str, Any]] = []
    for condition in RECOVERY_MMIRAGE_CONDITIONS:
        commands: list[list[str]] = []
        for rep in range(1, 4):
            run_cmd = [
                sys.executable, str(RECOVERY / "scripts" / "run_local.py"), "run-condition",
                "--condition", condition, "--rep", str(rep), "--shared-root", shared,
                "--max-active-shards", "4", "--gpu-ids", ",".join(gpus),
                "--config", str(RECOVERY / "configs" / "mmirage_recovery.yaml"),
            ]
            if condition != "baseline":
                run_cmd += ["--kill-after-seconds", "30"]
            if overwrite:
                run_cmd.append("--overwrite")
            commands.append(run_cmd)
            if condition != "baseline":
                commands.append([
                    sys.executable, str(RECOVERY / "scripts" / "run_local.py"), "retry",
                    "--condition", condition, "--rep", str(rep), "--shared-root", shared,
                    "--config", str(RECOVERY / "configs" / "mmirage_recovery.yaml"),
                ])
            rep_label = f"rep_{rep:02d}"
            commands.append([
                "mmirage", "merge-dir",
                "--input-dir", f"{shared}/runs/{condition}/{rep_label}/output",
                "--output-dir", f"{shared}/runs/{condition}/{rep_label}/merged",
            ])
        units.append({"stage": "recovery", "framework": "mmirage", "condition": condition, "gpu_count": 4, "physical_gpus": gpus, "commands": commands})

    for framework in RECOVERY_FRAMEWORKS:
        for condition in RECOVERY_NATIVE_CONDITIONS:
            commands = []
            for rep in range(1, 4):
                command = [
                    _native_python(framework), str(RECOVERY / "scripts" / "run_native_recovery_publication.py"),
                    "--framework", framework, "--condition", condition, "--rep", str(rep),
                    "--shared-root", shared, "--gpu-ids", ",".join(gpus),
                    "--max-active-shards", "4", "--model", MODEL,
                    "--concurrency", "64", "--max-new-tokens", "256",
                    "--kill-after-seconds", "30",
                ]
                if overwrite:
                    command.append("--overwrite")
                commands.append(command)
            units.append({"stage": "recovery", "framework": framework, "condition": condition, "gpu_count": 4, "physical_gpus": gpus, "commands": commands})
    return units


def text_plan(overwrite: bool, repetitions: int) -> list[dict[str, Any]]:
    gpus = ["0", "1", "2", "3"]
    mm = [
        sys.executable, str(SCALING / "scripts" / "run.py"),
        "--execution-config", str(TEXT / "configs" / "execution_4gpu.yaml"),
        "--visible-gpus", ",".join(gpus),
    ]
    if overwrite:
        mm.append("--overwrite")
    units = [{"stage": "text", "framework": "mmirage", "gpu_count": 4, "physical_gpus": gpus, "commands": [mm]}]
    for framework in ["datatrove", "nemo_curator"]:
        command = [
            _native_python(framework), str(SCALING / "scripts" / f"run_{framework}_scaling.py"),
            "--workload-jsonl", str(TEXT / "workload" / "workload.jsonl"),
            "--output-root", str(TEXT / "results" / "native_competitors" / framework),
            "--gpu-count", "4", "--visible-gpus", ",".join(gpus),
            "--repetitions", str(repetitions), "--model", MODEL,
            "--prompt-style", "summarize", "--concurrency", "64",
            "--temperature", "0.0", "--max-new-tokens", "128",
        ]
        if overwrite:
            command.append("--overwrite")
        units.append({"stage": "text", "framework": framework, "gpu_count": 4, "physical_gpus": gpus, "commands": [command]})
    return units


def vlm_plan(overwrite: bool, repetitions: int) -> list[dict[str, Any]]:
    gpus = ["0", "1", "2", "3"]
    mm = [
        sys.executable, str(VLM / "scripts" / "run_mmirage_vlm.py"),
        "--execution-config", str(VLM / "configs" / "execution_4gpu.yaml"),
        "--visible-gpus", ",".join(gpus),
    ]
    if overwrite:
        mm.append("--overwrite")
    units = [{"stage": "vlm", "framework": "mmirage", "gpu_count": 4, "physical_gpus": gpus, "commands": [mm]}]
    for framework in ["sglang", "datatrove", "nemo_curator"]:
        command = [
            sys.executable, str(VLM / "scripts" / "run_native_vlm_competitor.py"),
            "--framework", framework,
            "--workload-jsonl", str(VLM / "workload" / "rows.jsonl"),
            "--image-base-path", str(VLM / "workload"),
            "--output-root", str(VLM / "results" / "native_competitors" / framework),
            "--gpu-count", "4", "--visible-gpus", ",".join(gpus),
            "--repetitions", str(repetitions), "--model", VLM_MODEL,
            "--concurrency", "64", "--temperature", "0.1", "--top-p", "0.9",
            "--max-new-tokens", "1024",
        ]
        if framework in {"datatrove", "nemo_curator"}:
            command += ["--worker-python", _native_python(framework)]
        if overwrite:
            command.append("--overwrite")
        units.append({"stage": "vlm", "framework": framework, "gpu_count": 4, "physical_gpus": gpus, "commands": [command]})
    return units


def extraction_plan(repetitions: int) -> list[dict[str, Any]]:
    reps = ",".join(str(i) for i in range(1, repetitions + 1))
    command = [
        sys.executable, str(RECOVERY / "scripts" / "extract_results.py"),
        "--shared-root", _recovery_root(), "--conditions", "baseline,fail_1,fail_4",
        "--reps", reps, "--config", str(RECOVERY / "configs" / "mmirage_recovery.yaml"),
    ]
    return [{"stage": "recovery_extract", "framework": "all", "commands": [command]}]


def build_plan(stage: str, hardware: str, repetitions: int, overwrite: bool) -> list[dict[str, Any]]:
    if stage == "scaling":
        return scaling_plan(hardware, repetitions, overwrite)
    if stage == "recovery":
        return recovery_plan(overwrite, repetitions)
    if stage == "recovery_extract":
        return extraction_plan(repetitions)
    if stage == "text":
        return text_plan(overwrite, repetitions)
    if stage == "vlm":
        return vlm_plan(overwrite, repetitions)
    raise ValueError(stage)


def execute(plan: list[dict[str, Any]], dry_run: bool) -> None:
    for unit in plan:
        label = "/".join(str(unit[k]) for k in ("stage", "framework") if k in unit)
        if "condition" in unit:
            label += "/" + str(unit["condition"])
        if "gpu_count" in unit:
            label += f"/{unit['gpu_count']}gpu"
        print(f"publication: {label}")
        for command in unit["commands"]:
            print("  " + " ".join(command))
            if not dry_run:
                subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["scaling", "recovery", "recovery_extract", "text", "vlm"], required=True)
    parser.add_argument("--hardware", choices=["h100", "a100"], default="h100")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable plan and exit.")
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    template_stage = {"recovery_extract": "recovery"}.get(args.stage, args.stage)
    verify_templates([template_stage])
    plan = build_plan(args.stage, args.hardware, args.repetitions, args.overwrite)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    execute(plan, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
