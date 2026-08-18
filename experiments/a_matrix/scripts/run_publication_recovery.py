#!/usr/bin/env python3
"""Run the corrected publication recovery matrix sequentially on four GPUs."""

from __future__ import annotations

import argparse
import subprocess

import run_setup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def execute(commands, dry_run: bool) -> None:
    wrapper = run_setup.SHARD_RECOVERY_DIR / "scripts" / "run_native_recovery_publication.py"
    original = str(run_setup.SHARD_RECOVERY_DIR / "scripts" / "run_native_recovery_competitor.py")
    for command in commands:
        command = list(command)
        if len(command) > 1 and command[1] == original:
            command[1] = str(wrapper)
        print("  " + " ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=str(run_setup.REPO_ROOT), check=True)


def main() -> int:
    args = parse_args()
    # The underlying legacy command builders currently encode the publication
    # recovery matrix as exactly reps 1,2,3. Refuse silent mismatch.
    if args.repetitions != 3:
        raise SystemExit("publication recovery currently requires --repetitions 3")
    if not run_setup.verify_templates(["recovery"]):
        return 1

    gpus = ["0", "1", "2", "3"]
    for condition in run_setup.RECOVERY_MMIRAGE_CONDITIONS:
        print(f"publication-recovery: mmirage/{condition}")
        execute(run_setup._recovery_mmirage_cmds(condition, gpus, args.overwrite), args.dry_run)

    for framework in run_setup.RECOVERY_FRAMEWORKS:
        for condition in run_setup.RECOVERY_NATIVE_CONDITIONS:
            print(f"publication-recovery: {framework}/{condition}")
            execute(
                run_setup._recovery_native_cmds(framework, condition, gpus, args.overwrite),
                args.dry_run,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
