#!/usr/bin/env python3
"""Measure fresh-environment setup time for a comparison framework.

Creates a brand-new virtual environment and installs the framework's pinned
requirements while timing each phase. Results are written as JSON so that
``analyze_results.py`` can include setup effort alongside runtime metrics.

The installed artifacts are removed afterwards unless ``--keep-venv`` is given;
only the timing record is persisted. uv's wheel cache is reused (report the
numbers as warm-cache setup times).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FRAMEWORKS: dict[str, dict[str, object]] = {
    "anonlib": {
        "python": "3.12",
        "requirements": "experiments/nemo_curator_comparison/environment/anonlib_uv_requirements.txt",
        "extra_index": None,
        "pin_setuptools": None,
        "prerelease": True,
    },
    "nemo": {
        "python": "3.12",
        "requirements": "experiments/nemo_curator_comparison/environment/nemo_curator_uv_requirements.txt",
        "extra_index": "https://pypi.nvidia.com",
        "pin_setuptools": "75.8.0",
        "prerelease": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=sorted(FRAMEWORKS))
    parser.add_argument("--venv-dir", default=None, help="Where to create the throwaway venv (default: .venv-{framework}-setup-measure)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: setup_times/{framework}.json under the experiment)")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing venv-dir before measuring")
    parser.add_argument("--keep-venv", action="store_true", help="Keep the created venv instead of deleting it after measuring")
    return parser.parse_args()


def run_timed(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(command)}\n{result.stdout[-4000:]}")
    return elapsed


def main() -> None:
    args = parse_args()
    spec = FRAMEWORKS[args.framework]
    venv_dir = Path(args.venv_dir) if args.venv_dir else Path(f".venv-{args.framework}-setup-measure")
    output_path = Path(args.output) if args.output else Path("experiments/nemo_curator_comparison/setup_times") / f"{args.framework}.json"
    output_path = output_path if output_path.is_absolute() else REPO_ROOT / output_path

    if venv_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to reuse existing venv {venv_dir}. Use --overwrite to measure a fresh install.")
        shutil.rmtree(venv_dir)

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to measure setup time but was not found on PATH")

    phases: dict[str, float] = {}
    try:
        phases["venv_creation_seconds"] = run_timed(["uv", "venv", "--python", str(spec["python"]), str(venv_dir)], cwd=REPO_ROOT)

        venv_python = venv_dir / "bin" / "python"
        installs = 0.0
        if spec["pin_setuptools"]:
            pin = str(spec["pin_setuptools"])
            installs += run_timed(["uv", "pip", "install", "--python", str(venv_python), f"setuptools=={pin}"], cwd=REPO_ROOT)

        install_command = ["uv", "pip", "install", "--python", str(venv_python)]
        if spec["extra_index"]:
            install_command += ["--extra-index-url", str(spec["extra_index"])]
        if spec["pin_setuptools"]:
            install_command += ["--no-build-isolation"]
        if spec["prerelease"]:
            install_command += ["--prerelease=allow"]
        install_command += ["-r", str(spec["requirements"])]
        install_env = dict(os.environ)
        if spec["pin_setuptools"]:
            install_env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        installs += run_timed(install_command, cwd=REPO_ROOT, env=install_env)
        phases["dependency_install_seconds"] = installs

        python_version = subprocess.run([str(venv_python), "-V"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        uv_version = subprocess.run(["uv", "--version"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    finally:
        if not args.keep_venv and venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

    total = sum(phases.values())
    now = datetime.now(timezone.utc)
    record = {
        "framework": args.framework,
        "requirements_file": str(spec["requirements"]),
        "uv_version": uv_version,
        "python_version": python_version,
        "hostname": socket.gethostname(),
        "started_at": now.isoformat(),
        "warm_cache": True,
        "cache_note": "uv wheel cache reused; repeated measurements on the same machine do not re-download cached artifacts.",
        "phases": phases,
        "total_setup_seconds": round(total, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
