#!/usr/bin/env python3
"""Publication-safe native recovery wrapper.

It preserves the existing native recovery implementation but fixes the
post-failure orphan-engine cleanup wait so a subsequent wave cannot start while
GPU memory from a killed vLLM engine is still draining.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

import run_native_recovery_competitor as target


def safe_sweep_orphaned_engine_cores(args, max_wait_seconds: int = 90):
    orphans = target.orphaned_engine_cores()
    if not orphans:
        return []

    print(f"recovery-cleanup: terminating orphaned vLLM engine cores {orphans}", flush=True)
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    term_deadline = time.monotonic() + 12.0
    remaining = set(orphans)
    while remaining and time.monotonic() < term_deadline:
        for pid in list(remaining):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                remaining.discard(pid)
        if remaining:
            time.sleep(0.5)

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    drain_deadline = time.monotonic() + max_wait_seconds
    last = None
    while time.monotonic() < drain_deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            values = [int(v) for v in result.stdout.split()]
            last = values
            if values and all(v <= 500 for v in values):
                print(f"recovery-cleanup: GPU memory drained: {values} MiB", flush=True)
                return orphans
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as exc:
            last = repr(exc)
        time.sleep(5)

    raise RuntimeError(
        f"GPU memory did not drain within {max_wait_seconds}s after killing "
        f"orphaned vLLM engines {orphans}; last observation={last}"
    )


target.sweep_orphaned_engine_cores = safe_sweep_orphaned_engine_cores

if __name__ == "__main__":
    raise SystemExit(target.main())
