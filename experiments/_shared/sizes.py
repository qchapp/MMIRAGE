"""Workload sizes committed by experiments/smoke/calibrate.py.

Each experiment can carry a ``configs/workload_size.yaml`` file whose keys name
a ``prepare_workload`` flag (for example ``num_rows`` or ``num_records``). The
calibrator writes this file after smoke-testing one GPU pod; ``prepare_workload``
scripts read it as the default for that flag so humans can reproduce a run with
no arguments. When the file is missing, scripts fall back to a built-in size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def experiment_dir_of(path: Path) -> Path:
    """Return the first ancestor that contains a ``configs/`` directory."""
    for candidate in [path, *path.parents]:
        if (candidate / "configs").is_dir():
            return candidate
    return path


def workload_size_path(experiment_dir: Path) -> Path:
    return experiment_dir / "configs" / "workload_size.yaml"


def load_workload_size(experiment_dir: Path) -> Dict[str, Any]:
    path = workload_size_path(experiment_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_size(experiment_dir: Path, key: str, fallback: int) -> int:
    """Read ``key`` from the committed size file, else return ``fallback``."""
    value = load_workload_size(experiment_dir).get(key)
    if value is None:
        return fallback
    return int(value)
