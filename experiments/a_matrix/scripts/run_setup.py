#!/usr/bin/env python3
"""Orchestrate the A matrix and the Comparison B extras on one 4-GPU node.

This is a thin scheduler, not a runner: every unit of work delegates to an
existing, tested runner script via subprocess. A unit is one independent
command (or, for MMIRAGE shard recovery, the run-condition -> retry -> merge
sequence). Each unit declares how many GPUs it needs; the scheduler pins it to
free physical GPUs via the existing ``--visible-gpus`` / ``--gpu-ids`` flags.

The default entry point is ``bash experiments/run_all.sh``, which runs the
whole matrix (scaling 1/2/4, recovery, text, vlm) on one 4-GPU node. With an
explicit ``--setup``, run_setup.py runs just that stage.

Concurrency rules (kept deliberately simple to avoid correctness bugs):
  * at most one scaling unit per framework at a time, because the scaling
    runners write ``experiment_metadata.json`` / ``summary.json`` at a
    framework-scoped ``output_root``;
  * a unit starts only when enough GPU slots are free;
  * 4-GPU units (recovery, text, vlm) therefore run one at a time.

Usage:
  python experiments/a_matrix/scripts/run_setup.py --setup gpu_scaling --dry-run
  python experiments/a_matrix/scripts/run_setup.py --setup recovery
  python experiments/a_matrix/scripts/run_setup.py --setup a100_4gpu --dry-run
  python experiments/a_matrix/scripts/run_setup.py --setup gpu_scaling --reuse-fastruns
  python experiments/a_matrix/scripts/run_setup.py --setup recovery --extract

The pod assignment lives in experiments/a_matrix/schedule.yaml. ``--prepare``
only runs the workload preparation scripts the pod needs; ``--extract`` only
re-runs extraction/aggregation. Templates are verified byte-identical against
the shared prompt definitions before anything launches. ``--reuse-fastruns``
skips the units already satisfied by the 2026-08-15 fast-runs archive
(configs/reused_units.yaml); run_all.sh passes it by default (see ``--rerun-reused``).

``SETUPTOOLS_USE_DISTUTILS`` is forced to ``local`` at startup: the Python 3.12
venvs have no stdlib ``distutils``, and distilabel/ray_data_llm workers import
vllm (via setuptools) which needs the setuptools-provided shim. A shell
exporting ``=stdlib`` (or unsetting it before launch) breaks those workers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

A_MATRIX_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = A_MATRIX_DIR.parents[1]
SCRIPTS = A_MATRIX_DIR / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SINGLE_NODE_DIR = REPO_ROOT / "experiments" / "single_node_h100_scaling"
SHARD_RECOVERY_DIR = REPO_ROOT / "experiments" / "shard_recovery"
TEXT_DIR = REPO_ROOT / "experiments" / "task_comparison" / "text_shortening"
VLM_DIR = REPO_ROOT / "experiments" / "task_comparison" / "vlm_enrichment"

MODEL = "Qwen/Qwen3-4B"
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
GPU_IDS_DEFAULT = "0,1,2,3"


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding="utf-8")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", choices=["pod_a", "pod_b"], default=None, help="Pod whose scheduled work runs (see schedule.yaml).")
    parser.add_argument(
        "--setup",
        default=None,
        help="Restrict to one setup: gpu_scaling, a100_4gpu, recovery, text_shortening (or text), vlm_enrichment (or vlm).",
    )
    parser.add_argument("--gpus", default=GPU_IDS_DEFAULT, help="Comma-separated physical GPU ids on this pod.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and verify templates, then exit.")
    parser.add_argument("--prepare", action="store_true", help="Only run the workload preparation steps for the selected work.")
    parser.add_argument("--extract", action="store_true", help="Only re-run extraction/aggregation for the selected work.")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to the underlying runners.")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetitions (default 3).")
    parser.add_argument(
        "--serial", action="store_true", help="Run at most one unit at a time (no concurrent GPU scaling units)."
    )
    parser.add_argument(
        "--reuse-fastruns",
        action="store_true",
        help="Skip units satisfied by the 2026-08-15 fast-runs archive (see configs/reused_units.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Template drift guards
# ---------------------------------------------------------------------------


def _normalize_template(text: str) -> str:
    """Collapse any ``{{ x }}`` / ``{x}`` placeholder to a single marker."""
    import re

    return re.sub(r"\{\{\s*[a-zA-Z0-9_]+\s*\}\}|\{\s*[a-zA-Z0-9_]+\s*\}", "{}", text.strip())


def _recipe_output_prompt(recipe_path: Path) -> str:
    recipe = read_yaml(recipe_path)
    outputs = recipe.get("processing_params", {}).get("outputs", [])
    for item in outputs:
        prompt = item.get("prompt")
        if isinstance(prompt, str):
            return prompt
    raise RuntimeError(f"No prompt found in recipe outputs: {recipe_path}")


def verify_a_templates() -> Dict[str, bool]:
    import experiments._shared.native_frameworks as native

    task_prompt = read_yaml(A_MATRIX_DIR / "task.yaml").get("prompt_template", "")
    recipe_prompt = _recipe_output_prompt(A_MATRIX_DIR / "configs" / "semantic_recipe.yaml")
    recovery_prompt = _recipe_output_prompt(A_MATRIX_DIR / "configs" / "mmirage_recovery.yaml")
    checks = {
        "task.yaml == semantic_recipe.yaml": _normalize_template(task_prompt) == _normalize_template(recipe_prompt),
        "task.yaml == REWRITE_PROMPT_TEMPLATE": _normalize_template(task_prompt) == _normalize_template(native.REWRITE_PROMPT_TEMPLATE),
        "semantic_recipe.yaml == mmirage_recovery.yaml": _normalize_template(recipe_prompt) == _normalize_template(recovery_prompt),
    }
    return checks


def verify_text_templates() -> Dict[str, bool]:
    import experiments._shared.native_frameworks as native

    recipe_prompt = _recipe_output_prompt(TEXT_DIR / "configs" / "semantic_recipe.yaml")
    return {
        "text recipe == SUMMARIZE_PROMPT_TEMPLATE": _normalize_template(recipe_prompt) == _normalize_template(native.SUMMARIZE_PROMPT_TEMPLATE)
    }


def verify_vlm_templates() -> Dict[str, bool]:
    import experiments._shared.vlm_runners as vlm

    recipe_prompt = _recipe_output_prompt(VLM_DIR / "configs" / "mmirage_vlm.yaml")
    return {
        "vlm recipe == VLM_REFORMAT_TEMPLATE": _normalize_template(recipe_prompt) == _normalize_template(vlm.VLM_REFORMAT_TEMPLATE)
    }


def verify_templates(selected: Sequence[str]) -> bool:
    checks: Dict[str, bool] = {}
    if "gpu_scaling" in selected or "a100_4gpu" in selected or "recovery" in selected:
        checks.update(verify_a_templates())
    if "text_shortening" in selected:
        checks.update(verify_text_templates())
    if "vlm_enrichment" in selected:
        checks.update(verify_vlm_templates())
    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"template-check: {'PASS' if passed else 'FAIL'}  {name}")
    if not ok:
        print("template verification failed; refusing to launch.", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Unit model
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    label: str
    setup: str
    framework: str
    gpus_needed: int
    commands: List[List[str]]  # run sequentially; unit done when all rc==0
    log_path: Path


def _python(venv_env: str) -> str:
    value = os.environ.get(venv_env)
    if value and Path(value).is_file() and os.access(value, os.X_OK):
        return value
    return sys.executable


# ---------------------------------------------------------------------------
# Command builders (thin wrappers over existing runners; no logic duplication)
# ---------------------------------------------------------------------------


def _scaling_mmirage_cmds(setup: str, gpu_count: int, gpus: Sequence[str], repetitions: int, overwrite: bool) -> List[List[str]]:
    results_dir = A_MATRIX_DIR / "results" / setup / "mmirage"
    return [
        [
            sys.executable,
            str(SINGLE_NODE_DIR / "scripts" / "run.py"),
            "--execution-config",
            str(A_MATRIX_DIR / "configs" / "execution.yaml"),
            "--semantic-recipe",
            str(A_MATRIX_DIR / "configs" / "semantic_recipe.yaml"),
            "--workload-jsonl",
            str(A_MATRIX_DIR / "workload" / "workload.jsonl"),
            "--output-root",
            str(results_dir),
            "--gpu-count",
            str(gpu_count),
            "--visible-gpus",
            ",".join(gpus),
            "--repetitions",
            str(repetitions),
        ]
        + (["--overwrite"] if overwrite else [])
    ]


def _scaling_native_cmds(setup: str, framework: str, gpu_count: int, gpus: Sequence[str], repetitions: int, overwrite: bool) -> List[List[str]]:
    venv_env = {"datatrove": "MMIRAGE_DATATROVE_PYTHON", "nemo_curator": "MMIRAGE_NEMO_CURATOR_PYTHON", "distilabel": "MMIRAGE_DISTILABEL_PYTHON", "ray_data_llm": "MMIRAGE_RAY_DATA_LLM_PYTHON"}.get(framework)
    python = _python(venv_env) if venv_env else sys.executable
    return [
        [
            python,
            str(SINGLE_NODE_DIR / "scripts" / f"run_{framework}_scaling.py"),
            "--workload-jsonl",
            str(A_MATRIX_DIR / "workload" / "workload.jsonl"),
            "--output-root",
            str(A_MATRIX_DIR / "results" / setup / framework),
            "--gpu-count",
            str(gpu_count),
            "--visible-gpus",
            ",".join(gpus),
            "--repetitions",
            str(repetitions),
            "--model",
            MODEL,
            "--concurrency",
            "64",
            "--temperature",
            "0.0",
            "--max-new-tokens",
            "256",
            "--prompt-style",
            "rewrite",
        ]
        + (["--overwrite"] if overwrite else [])
    ]


def _recovery_shared_root() -> str:
    return os.environ.get("MMIRAGE_RECOVERY_ROOT", "/workspace/mmirage-recovery")


def _recovery_mmirage_cmds(condition: str, gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    shared_root = _recovery_shared_root()
    commands: List[List[str]] = []
    for rep in range(1, 4):
        rep_label = f"rep_{rep:02d}"
        run_cond_cmd = [
            sys.executable,
            str(SHARD_RECOVERY_DIR / "scripts" / "run_local.py"),
            "run-condition",
            "--condition",
            condition,
            "--rep",
            str(rep),
            "--shared-root",
            shared_root,
            "--max-active-shards",
            "4",
            "--gpu-ids",
            ",".join(gpus),
            "--config",
            str(A_MATRIX_DIR / "configs" / "mmirage_recovery.yaml"),
        ]
        if condition != "baseline":
            run_cond_cmd += ["--kill-after-seconds", "30"]
        if overwrite:
            run_cond_cmd.append("--overwrite")
        commands.append(run_cond_cmd)
        if condition != "baseline":
            retry_cmd = [
                sys.executable,
                str(SHARD_RECOVERY_DIR / "scripts" / "run_local.py"),
                "retry",
                "--condition",
                condition,
                "--rep",
                str(rep),
                "--shared-root",
                shared_root,
                "--config",
                str(A_MATRIX_DIR / "configs" / "mmirage_recovery.yaml"),
            ]
            commands.append(retry_cmd)
        commands.append(
            [
                "mmirage",
                "merge-dir",
                "--input-dir",
                f"{shared_root}/runs/{condition}/{rep_label}/output",
                "--output-dir",
                f"{shared_root}/runs/{condition}/{rep_label}/merged",
            ]
        )
    return commands


def _recovery_native_cmds(framework: str, condition: str, gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    venv_env = {"datatrove": "MMIRAGE_DATATROVE_PYTHON", "nemo_curator": "MMIRAGE_NEMO_CURATOR_PYTHON", "distilabel": "MMIRAGE_DISTILABEL_PYTHON", "ray_data_llm": "MMIRAGE_RAY_DATA_LLM_PYTHON"}.get(framework)
    python = _python(venv_env) if venv_env else sys.executable
    commands: List[List[str]] = []
    for rep in range(1, 4):
        commands.append(
            [
                python,
                str(SHARD_RECOVERY_DIR / "scripts" / "run_native_recovery_competitor.py"),
                "--framework",
                framework,
                "--condition",
                condition,
                "--rep",
                str(rep),
                "--shared-root",
                _recovery_shared_root(),
                "--gpu-ids",
                ",".join(gpus),
                "--max-active-shards",
                "4",
                "--model",
                MODEL,
                "--concurrency",
                "64",
                "--max-new-tokens",
                "256",
                "--kill-after-seconds",
                "30",
            ]
            + (["--overwrite"] if overwrite else [])
        )
    return commands


def _text_mmirage_cmds(gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    return [
        [
            sys.executable,
            str(SINGLE_NODE_DIR / "scripts" / "run.py"),
            "--execution-config",
            str(TEXT_DIR / "configs" / "execution_4gpu.yaml"),
            "--visible-gpus",
            ",".join(gpus),
        ]
        + (["--overwrite"] if overwrite else [])
    ]


def _text_native_cmds(framework: str, repetitions: int, gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    venv_env = {"datatrove": "MMIRAGE_DATATROVE_PYTHON", "nemo_curator": "MMIRAGE_NEMO_CURATOR_PYTHON", "distilabel": "MMIRAGE_DISTILABEL_PYTHON", "ray_data_llm": "MMIRAGE_RAY_DATA_LLM_PYTHON"}[framework]
    return [
        [
            _python(venv_env),
            str(SINGLE_NODE_DIR / "scripts" / f"run_{framework}_scaling.py"),
            "--workload-jsonl",
            str(TEXT_DIR / "workload" / "workload.jsonl"),
            "--output-root",
            str(TEXT_DIR / "results" / "native_competitors" / framework),
            "--gpu-count",
            "4",
            "--visible-gpus",
            ",".join(gpus),
            "--repetitions",
            str(repetitions),
            "--model",
            MODEL,
            "--prompt-style",
            "summarize",
            "--concurrency",
            "64",
            "--temperature",
            "0.0",
            "--max-new-tokens",
            "128",
        ]
        + (["--overwrite"] if overwrite else [])
    ]


def _vlm_mmirage_cmds(gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    return [
        [
            sys.executable,
            str(VLM_DIR / "scripts" / "run_mmirage_vlm.py"),
            "--execution-config",
            str(VLM_DIR / "configs" / "execution_4gpu.yaml"),
            "--visible-gpus",
            ",".join(gpus),
        ]
        + (["--overwrite"] if overwrite else [])
    ]


def _vlm_native_cmds(framework: str, repetitions: int, gpus: Sequence[str], overwrite: bool) -> List[List[str]]:
    venv_env = {"datatrove": "MMIRAGE_DATATROVE_PYTHON", "nemo_curator": "MMIRAGE_NEMO_CURATOR_PYTHON", "distilabel": "MMIRAGE_DISTILABEL_PYTHON", "ray_data_llm": "MMIRAGE_RAY_DATA_LLM_PYTHON"}.get(framework)
    command = [
        sys.executable,
        str(VLM_DIR / "scripts" / "run_native_vlm_competitor.py"),
        "--framework",
        framework,
        "--workload-jsonl",
        str(VLM_DIR / "workload" / "rows.jsonl"),
        "--image-base-path",
        str(VLM_DIR / "workload"),
        "--output-root",
        str(VLM_DIR / "results" / "native_competitors" / framework),
        "--gpu-count",
        "4",
        "--visible-gpus",
        ",".join(gpus),
        "--repetitions",
        str(repetitions),
        "--model",
        VLM_MODEL,
        "--concurrency",
        "64",
        "--temperature",
        "0.1",
        "--top-p",
        "0.9",
        "--max-new-tokens",
        "1024",
    ]
    if venv_env:
        command += ["--worker-python", _python(venv_env)]
    if overwrite:
        command.append("--overwrite")
    return [command]


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

SCALING_FRAMEWORKS = ["mmirage", "raw_sglang", "datatrove", "nemo_curator"]
RECOVERY_FRAMEWORKS = ["raw_sglang", "datatrove", "nemo_curator", "distilabel", "ray_data_llm"]
RECOVERY_NATIVE_CONDITIONS = ["fail_1", "fail_4"]
RECOVERY_MMIRAGE_CONDITIONS = ["baseline", "fail_1", "fail_4"]


def build_plan(args: argparse.Namespace, selected: Sequence[str]) -> List[Unit]:
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    log_dir = REPO_ROOT / "experiments" / "run_all_logs" / "a_matrix" / (args.pod or args.setup or "standalone")
    log_dir.mkdir(parents=True, exist_ok=True)
    repetitions = args.repetitions or 3

    units: List[Unit] = []

    def add(setup: str, framework: str, label: str, gpus: int, commands: List[List[str]]) -> None:
        units.append(
            Unit(
                label=label,
                setup=setup,
                framework=framework,
                gpus_needed=gpus,
                commands=commands,
                log_path=log_dir / f"{label}.log",
            )
        )

    if "gpu_scaling" in selected:
        gpu_points = [1, 2, 4]
        if args.pod == "pod_a":
            gpu_points = [1, 2]
        elif args.pod == "pod_b":
            gpu_points = [4]
        for gpu_count in gpu_points:
            for framework in SCALING_FRAMEWORKS:
                add(
                    "gpu_scaling",
                    framework,
                    f"gpu_scaling/{framework}/gpu_{gpu_count}",
                    gpu_count,
                    _scaling_mmirage_cmds("gpu_scaling", gpu_count, gpu_ids[:gpu_count], repetitions, args.overwrite)
                    if framework == "mmirage"
                    else _scaling_native_cmds("gpu_scaling", framework, gpu_count, gpu_ids[:gpu_count], repetitions, args.overwrite),
                )

    if "a100_4gpu" in selected:
        for framework in SCALING_FRAMEWORKS:
            add(
                "a100_4gpu",
                framework,
                f"a100_4gpu/{framework}/gpu_4",
                4,
                _scaling_mmirage_cmds("a100_4gpu", 4, gpu_ids[:4], repetitions, args.overwrite)
                if framework == "mmirage"
                else _scaling_native_cmds("a100_4gpu", framework, 4, gpu_ids[:4], repetitions, args.overwrite),
            )

    if "recovery" in selected:
        pod_gpus = gpu_ids[:4]
        for condition in RECOVERY_MMIRAGE_CONDITIONS:
            add("recovery", "mmirage", f"recovery/mmirage/{condition}", 4, _recovery_mmirage_cmds(condition, pod_gpus, args.overwrite))
        for framework in RECOVERY_FRAMEWORKS:
            for condition in RECOVERY_NATIVE_CONDITIONS:
                add(
                    "recovery",
                    framework,
                    f"recovery/{framework}/{condition}",
                    4,
                    _recovery_native_cmds(framework, condition, pod_gpus, args.overwrite),
                )

    if "text_shortening" in selected:
        pod_gpus = gpu_ids[:4]
        add("text_shortening", "mmirage", "text_shortening/mmirage", 4, _text_mmirage_cmds(pod_gpus, args.overwrite))
        for framework in ["datatrove", "nemo_curator"]:
            add("text_shortening", framework, f"text_shortening/{framework}", 4, _text_native_cmds(framework, repetitions, pod_gpus, args.overwrite))

    if "vlm_enrichment" in selected:
        pod_gpus = gpu_ids[:4]
        add("vlm_enrichment", "mmirage", "vlm_enrichment/mmirage", 4, _vlm_mmirage_cmds(pod_gpus, args.overwrite))
        for framework in ["sglang", "datatrove", "nemo_curator"]:
            add("vlm_enrichment", framework, f"vlm_enrichment/{framework}", 4, _vlm_native_cmds(framework, repetitions, pod_gpus, args.overwrite))

    return units


# ---------------------------------------------------------------------------
# Reuse of the 2026-08-15 fast-runs archive
# ---------------------------------------------------------------------------


def _load_reused_units() -> List[Dict[str, Any]]:
    path = A_MATRIX_DIR / "configs" / "reused_units.yaml"
    if not path.exists():
        return []
    return list((read_yaml(path) or {}).get("reused_units", []))


def _filter_reused(units: List[Unit], reuse_fastruns: bool) -> List[Unit]:
    if not reuse_fastruns:
        return units
    entries = {item["label"]: item for item in _load_reused_units()}
    kept: List[Unit] = []
    for unit in units:
        entry = entries.get(unit.label)
        if entry is None:
            kept.append(unit)
            continue
        target = REPO_ROOT / entry.get("target", "")
        if not target.exists():
            print(
                f"  WARNING {unit.label}: previous results not found at {target} - "
                "restore the fast-runs archive before analysis (README 'Reusing the 2026-08-15 fast-runs').",
                file=sys.stderr,
            )
        print(f"  SKIP {unit.label}  <- reused from fast-runs archive: {entry.get('src', '?')}")
    return kept


def _verify_reused_sizes() -> bool:
    """Reused results are only valid while the committed sizes match the runs."""
    size_files = {
        "gpu_scaling/mmirage/gpu_1": A_MATRIX_DIR / "configs" / "workload_size.yaml",
        "gpu_scaling/mmirage/gpu_2": A_MATRIX_DIR / "configs" / "workload_size.yaml",
        "gpu_scaling/mmirage/gpu_4": A_MATRIX_DIR / "configs" / "workload_size.yaml",
        "text_shortening/mmirage": TEXT_DIR / "configs" / "workload_size.yaml",
        "vlm_enrichment/mmirage": VLM_DIR / "configs" / "workload_size.yaml",
    }
    ok = True
    for entry in _load_reused_units():
        path = size_files.get(entry["label"])
        if path is None or not path.exists():
            continue
        size = int((read_yaml(path) or {}).get("num_rows", -1))
        expected = int(entry.get("num_rows", -1))
        if size != expected:
            ok = False
            print(
                f"  ERROR {entry['label']}: {path.relative_to(REPO_ROOT)} num_rows={size} != {expected} "
                f"(the 2026-08-15 run was {expected} rows). The reused results would not match the new workload. "
                "Fix the size or rerun with --rerun-reused / without --reuse-fastruns.",
                file=sys.stderr,
            )
    return ok


# ---------------------------------------------------------------------------
# Preparation steps (workloads, shared recovery root)
# ---------------------------------------------------------------------------


def run_prepare(selected: Sequence[str], args: argparse.Namespace) -> int:
    steps: List[List[str]] = []
    if "gpu_scaling" in selected or "a100_4gpu" in selected or "recovery" in selected:
        cmd = [
            sys.executable,
            str(SCRIPTS / "prepare_workload.py"),
            "--output-dir",
            str(A_MATRIX_DIR / "workload"),
        ]
        if "recovery" in selected:
            cmd += ["--shared-root", _recovery_shared_root()]
        steps.append(cmd)
    if "text_shortening" in selected:
        steps.append(
            [
                sys.executable,
                str(TEXT_DIR / "scripts" / "prepare_workload.py"),
                "--output-dir",
                str(TEXT_DIR / "workload"),
            ]
        )
    if "vlm_enrichment" in selected:
        steps.append(
            [
                sys.executable,
                str(VLM_DIR / "scripts" / "prepare_workload.py"),
                "--output-dir",
                str(VLM_DIR / "workload"),
            ]
        )
    for command in steps:
        print(f"prepare: {' '.join(command)}")
        result = subprocess.run(command, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            return result.returncode
    return 0


# ---------------------------------------------------------------------------
# Extraction / aggregation
# ---------------------------------------------------------------------------


def run_extract(selected: Sequence[str], args: argparse.Namespace) -> int:
    failed = 0
    if "recovery" in selected:
        command = [
            sys.executable,
            str(SHARD_RECOVERY_DIR / "scripts" / "extract_results.py"),
            "--shared-root",
            _recovery_shared_root(),
            "--conditions",
            "baseline,fail_1,fail_4",
            "--reps",
            "1",
            "--config",
            str(A_MATRIX_DIR / "configs" / "mmirage_recovery.yaml"),
        ]
        print(f"extract: {' '.join(command)}")
        result = subprocess.run(command, cwd=str(REPO_ROOT))
        failed += 1 if result.returncode != 0 else 0
    scaling_dirs = []
    if "gpu_scaling" in selected:
        scaling_dirs.append(A_MATRIX_DIR / "results" / "gpu_scaling")
    if "a100_4gpu" in selected:
        scaling_dirs.append(A_MATRIX_DIR / "results" / "a100_4gpu")
    for results_dir in scaling_dirs:
        if not results_dir.exists():
            print(f"extract: no results yet at {results_dir}")
            continue
        for framework_dir in sorted(results_dir.iterdir()):
            if not framework_dir.is_dir():
                continue
            summary = framework_dir / "summary.json"
            raw = framework_dir / "raw_results.csv"
            present = summary.exists() and raw.exists()
            print(f"extract: {framework_dir.name}: summary={'present' if summary.exists() else 'MISSING'} raw={'present' if raw.exists() else 'MISSING'}")
            if not present:
                failed += 1
    return failed


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass
class _RunningUnit:
    unit: Unit
    gpus: List[str]
    proc: Optional[subprocess.Popen[str]]
    pending_commands: List[List[str]]
    started_at: float
    log_handle: Any = None
    steps: List[Dict[str, Any]] = field(default_factory=list)


def _start_command(running: _RunningUnit, log_dir: Path) -> None:
    command = running.pending_commands.pop(0)
    running.unit.log_path.parent.mkdir(parents=True, exist_ok=True)
    running.log_handle = running.unit.log_path.open("ab")
    running.log_handle.write(f"\n[run_setup] {utc_now()} step: {' '.join(command)}\n".encode("utf-8"))
    running.log_handle.flush()
    running.steps.append({"command": " ".join(command), "returncode": None})
    # GPU isolation is achieved by passing disjoint *physical* GPU ids via the
    # runners' own --visible-gpus/--gpu-ids flags; every runner pins each shard
    # worker with CUDA_VISIBLE_DEVICES=<physical id> itself. The parent must NOT
    # set a CUDA_VISIBLE_DEVICES mask here: that would renumber the physical ids
    # the runners hand to their workers and break the mapping.
    running.proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        stdout=running.log_handle,
        stderr=subprocess.STDOUT,
    )


def _finish_command(running: _RunningUnit) -> None:
    proc = running.proc
    running.proc = None
    rc = proc.poll() if proc is not None else -1
    if running.log_handle is not None:
        running.log_handle.write(f"[run_setup] {utc_now()} step finished rc={rc}\n".encode("utf-8"))
        running.log_handle.close()
        running.log_handle = None
    if running.steps:
        running.steps[-1]["returncode"] = rc
    if rc != 0:
        running.pending_commands.clear()


def run_scheduler(units: List[Unit], gpu_ids: List[str], dry_run: bool, serial: bool = False) -> int:
    if dry_run:
        return 0
    pending = list(units)
    free_gpus = list(gpu_ids)
    running: Dict[str, _RunningUnit] = {}
    active_frameworks: Dict[str, str] = {}
    status: Dict[str, Dict[str, Any]] = {}
    overall_failed = 0

    try:
        while pending or running:
            launched_any = False
            for unit in list(pending):
                if serial and running:
                    break
                if unit.gpus_needed > len(free_gpus):
                    continue
                if unit.setup == "gpu_scaling" and unit.framework in active_frameworks:
                    continue
                gpus = free_gpus[: unit.gpus_needed]
                free_gpus = free_gpus[unit.gpus_needed:]
                for cmd in unit.commands:
                    for i, arg in enumerate(cmd):
                        if arg == "--visible-gpus" and i + 1 < len(cmd):
                            cmd[i + 1] = ",".join(gpus)
                item = _RunningUnit(unit=unit, gpus=gpus, proc=None, pending_commands=list(unit.commands), started_at=time.monotonic())
                if unit.setup == "gpu_scaling":
                    active_frameworks[unit.framework] = unit.label
                pending.remove(unit)
                running[unit.label] = item
                print(f"launch: {unit.label} gpus={','.join(gpus)} free={len(free_gpus)}")
                _start_command(item, REPO_ROOT / "experiments" / "run_all_logs")
                launched_any = True

            if not running:
                if pending:
                    raise RuntimeError(f"no runnable unit with {len(free_gpus)} free GPUs; remaining={[u.label for u in pending]}")
                break
            if not launched_any and running:
                time.sleep(2)

            finished = []
            for label, item in running.items():
                if item.proc is not None and item.proc.poll() is not None:
                    _finish_command(item)
                if item.proc is None:
                    if item.pending_commands:
                        _start_command(item, REPO_ROOT / "experiments" / "run_all_logs")
                    else:
                        finished.append(label)
            for label in finished:
                item = running.pop(label)
                free_gpus.extend(item.gpus)
                if item.unit.setup == "gpu_scaling":
                    active_frameworks.pop(item.unit.framework, None)
                ok = all(step["returncode"] == 0 for step in item.steps)
                if not ok:
                    overall_failed += 1
                wall = round(time.monotonic() - item.started_at, 3)
                status[label] = {
                    "status": "ok" if ok else "FAILED",
                    "wall_seconds": wall,
                    "steps": item.steps,
                    "gpus": item.gpus,
                }
                print(f"done: {label} {'ok' if ok else 'FAILED'} after {wall}s")
    except KeyboardInterrupt:
        print("run_setup: interrupted; terminating running units.", file=sys.stderr)
        for item in running.values():
            if item.proc is not None:
                item.proc.terminate()
        raise

    write_json(REPO_ROOT / "experiments" / "run_all_logs" / "a_matrix" / "status.json", status)
    return overall_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def selected_setups(args: argparse.Namespace) -> List[str]:
    if args.setup:
        aliases = {"text": "text_shortening", "vlm": "vlm_enrichment"}
        name = aliases.get(args.setup, args.setup)
        valid = {"gpu_scaling", "a100_4gpu", "recovery", "text_shortening", "vlm_enrichment"}
        if name not in valid:
            raise SystemExit(f"unknown setup {args.setup!r}; choose from {sorted(valid)}")
        return [name]
    schedule = read_yaml(A_MATRIX_DIR / "schedule.yaml")
    if not args.pod:
        raise SystemExit("need --pod (pod_a|pod_b) or an explicit --setup")
    entry = schedule.get(args.pod)
    if not entry:
        raise SystemExit(f"no schedule entry for {args.pod}")
    selected = [entry["setup"]]
    selected.extend(entry.get("extra", []))
    return selected


def main() -> int:
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must contain at least one GPU id")

    selected = selected_setups(args)
    print(f"run_setup: pod={args.pod} setups={selected} gpus={gpu_ids} mode={'dry-run' if args.dry_run else 'run'}")

    if not verify_templates(selected):
        return 1

    if args.prepare:
        return run_prepare(selected, args)
    if args.extract:
        return run_extract(selected, args)

    units = build_plan(args, selected)
    if args.reuse_fastruns:
        print("run_setup: --reuse-fastruns: using results from the 2026-08-15 fast-runs archive:")
        if not _verify_reused_sizes():
            return 1
        units = _filter_reused(units, True)
    print(f"run_setup: plan has {len(units)} unit(s)")
    for unit in units:
        print(f"  {unit.label}  gpus={unit.gpus_needed}  cmds={len(unit.commands)}")
        if args.dry_run:
            for command in unit.commands:
                print(f"      {' '.join(command)}")
    if args.dry_run:
        return 0

    missing = []
    for unit in units:
        for command in unit.commands:
            candidate = command[0]
            if candidate == "mmirage" and not shutil.which("mmirage"):
                missing.append(candidate)
            elif candidate != "mmirage" and not shutil.which(candidate) and not Path(candidate).exists():
                missing.append(candidate)
    if missing:
        print(f"run_setup: missing executables: {sorted(set(missing))}", file=sys.stderr)
        return 1

    if args.overwrite:
        for unit in units:
            unit.log_path.parent.mkdir(parents=True, exist_ok=True)
            unit.log_path.open("wb").close()
            print(f"  CLEAR {unit.label} log")

    return run_scheduler(units, gpu_ids, dry_run=False, serial=args.serial)


if __name__ == "__main__":
    raise SystemExit(main())
