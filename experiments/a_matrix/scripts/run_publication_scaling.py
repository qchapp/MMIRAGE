#!/usr/bin/env python3
"""Publication scaling runner with deterministic physical GPU placement."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import run_setup


FRAMEWORKS = ["mmirage", "raw_sglang", "datatrove", "nemo_curator"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", choices=["gpu_scaling", "a100_4gpu"], required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if not run_setup.verify_templates([args.setup]):
        return 1

    points = [1, 2, 4] if args.setup == "gpu_scaling" else [4]
    status = {}
    for gpu_count in points:
        # Publication invariant: every framework at the same GPU count uses the
        # same physical cards, eliminating card-to-card placement as a confound.
        gpus = [str(i) for i in range(gpu_count)]
        for framework in FRAMEWORKS:
            if framework == "mmirage":
                commands = run_setup._scaling_mmirage_cmds(
                    args.setup, gpu_count, gpus, args.repetitions, args.overwrite
                )
            else:
                commands = run_setup._scaling_native_cmds(
                    args.setup, framework, gpu_count, gpus, args.repetitions, args.overwrite
                )
            label = f"{args.setup}/{framework}/gpu_{gpu_count}"
            print(f"publication-scaling: {label} physical_gpus={','.join(gpus)}")
            status[label] = {"physical_gpus": gpus, "commands": []}
            for command in commands:
                print("  " + " ".join(command))
                status[label]["commands"].append(" ".join(command))
                if not args.dry_run:
                    subprocess.run(command, cwd=str(run_setup.REPO_ROOT), check=True)
            status[label]["status"] = "planned" if args.dry_run else "ok"

    if not args.dry_run:
        out = run_setup.REPO_ROOT / "experiments" / "run_all_logs" / "a_matrix" / f"{args.setup}_publication_status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"created_at": datetime.now(timezone.utc).isoformat(), "units": status},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
