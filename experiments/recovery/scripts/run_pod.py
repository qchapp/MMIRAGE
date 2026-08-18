#!/usr/bin/env python3
"""Pod entrypoint for one externally orchestrated MMIRAGE shard."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CONTAINER_REPO = Path("/workspace/MMIRAGE")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


sys.path.insert(0, str(repo_root() / "src"))

from mmirage.config.utils import load_mmirage_config  # noqa: E402
from mmirage.shard_utils import (  # noqa: E402
    _mark_failure,
    read_status,
    shard_state_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.config).exists():
        raise RuntimeError(
            f"MMIRAGE config not found inside the shard pod at {args.config}. "
            f"The pod image should contain the repository at {DEFAULT_CONTAINER_REPO}."
        )
    source_path = str(repo_root() / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_path
    )
    os.environ["SLURM_ARRAY_TASK_ID"] = str(args.shard_id)
    os.environ.setdefault("MMIRAGE_COLLECT_STATS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    cfg = load_mmirage_config(args.config)
    state_dir = shard_state_dir(args.shard_id, cfg.loading_params.get_state_root())
    command = [sys.executable, "-m", "mmirage.shard_process", "--config", args.config]
    child = subprocess.Popen(command, env=os.environ.copy())
    terminated_by_wrapper = False

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal terminated_by_wrapper
        terminated_by_wrapper = True
        print(
            f"Wrapper for shard {args.shard_id} received signal {signum}; terminating MMIRAGE shard process {child.pid}.",
            flush=True,
        )
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    try:
        while child.poll() is None:
            time.sleep(1)
        return_code = int(child.returncode or 0)
    finally:
        if child.poll() is None:
            child.kill()
            return_code = 128 + signal.SIGKILL

    if terminated_by_wrapper:
        _mark_failure(
            state_dir,
            "Kubernetes pod was deliberately terminated by the shard-recovery experiment controller.",
        )
        sys.exit(1)

    if return_code != 0:
        status = read_status(state_dir)
        if status.status not in {"failed", "success"}:
            _mark_failure(state_dir, f"Shard process exited with code {return_code}.")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
