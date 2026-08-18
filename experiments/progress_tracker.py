#!/usr/bin/env python3
"""Read-only progress dashboard for the publication H100 and A100 suites.

The tracker can be started after a run has already begun. It inspects process
state and durable result artifacts only; it never writes experiment state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCALING = HERE / "scaling"
RECOVERY = HERE / "recovery"
TEXT = HERE / "text_shortening"
VLM = HERE / "vlm_enrichment"
OVERHEAD = HERE / "sglang_overhead"
DEFAULT_RECOVERY_ROOT = Path(os.environ.get("MMIRAGE_RECOVERY_ROOT", "/workspace/mmirage-recovery"))

H100_PRIORS = {
    "prepare": 10 * 60,
    "prefetch": 5 * 60,
    "scaling": 100 * 60,
    "recovery": 220 * 60,
    "extract": 5 * 60,
    "text": 25 * 60,
    "vlm": 25 * 60,
    "overhead": 20 * 60,
}
A100_PRIORS = {"verify": 5 * 60, "prefetch": 10 * 60, "scaling": 40 * 60, "overhead": 25 * 60}


@dataclass
class Stage:
    key: str
    label: str
    done: int
    total: int
    status: str = "queued"

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total) if self.total else 0.0


def run_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=8, check=False).stdout.strip()
    except Exception:
        return ""


def processes() -> list[str]:
    text = run_output(["ps", "-eo", "pid=,etimes=,args="])
    return [line.strip() for line in text.splitlines() if line.strip()]


def active_line(lines: Iterable[str], needles: Iterable[str]) -> str | None:
    for line in lines:
        if any(needle in line for needle in needles):
            return line
    return None


def count_rep_summaries(root: Path) -> int:
    return sum(1 for p in root.rglob("rep_summary.json") if p.is_file()) if root.exists() else 0


def nonempty_dir(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def recovery_count(shared: Path) -> int:
    done = 0
    for condition in ("baseline", "fail_1", "fail_4"):
        for rep in range(1, 4):
            if nonempty_dir(shared / "runs" / condition / f"rep_{rep:02d}" / "merged"):
                done += 1
    for framework in ("raw_sglang", "datatrove", "nemo_curator", "distilabel", "ray_data_llm"):
        for condition in ("fail_1", "fail_4"):
            for rep in range(1, 4):
                summary = shared / "native_competitors" / framework / condition / f"rep_{rep:02d}" / "summary.json"
                validation = summary.parent / "validation.json"
                if summary.is_file() and validation.is_file():
                    try:
                        if json.loads(validation.read_text(encoding="utf-8")).get("valid") is True:
                            done += 1
                    except Exception:
                        pass
    return done


def csv_pairs(path: Path) -> int:
    if not path.is_file():
        return 0
    pairs = set()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                framework = row.get("framework") or row.get("path") or row.get("mode")
                rep = row.get("repetition") or row.get("rep")
                if framework and rep:
                    pairs.add((framework, rep))
    except Exception:
        return 0
    return len(pairs)


def h100_stages(shared: Path) -> list[Stage]:
    prep_files = [
        SCALING / "workload" / "metadata.json",
        TEXT / "workload" / "metadata.json",
        VLM / "workload" / "metadata.json",
        OVERHEAD / "workload" / "metadata.json",
    ]
    prep = sum(p.is_file() for p in prep_files)
    models = SCALING / "workload" / "model_revisions.json"
    prefetch = 0
    if models.is_file():
        try:
            payload = json.loads(models.read_text(encoding="utf-8"))
            prefetch = sum(model in payload for model in ("Qwen/Qwen3-4B", "Qwen/Qwen3-VL-4B-Instruct"))
        except Exception:
            pass
    scaling = count_rep_summaries(SCALING / "results" / "h100")
    recovery = recovery_count(shared)
    extract = int((shared / "results" / "recovery_results.json").is_file() or (RECOVERY / "results" / "recovery_results.json").is_file())
    text = count_rep_summaries(TEXT / "results")
    vlm = count_rep_summaries(VLM / "results")
    overhead = csv_pairs(OVERHEAD / "results" / "h100" / "raw_results.csv")
    return [
        Stage("prepare", "workload preparation", prep, 4),
        Stage("prefetch", "model prefetch/revision lock", prefetch, 2),
        Stage("scaling", "H100 strong scaling", scaling, 36),
        Stage("recovery", "recovery matrix", recovery, 39),
        Stage("extract", "recovery extraction/persist", extract, 1),
        Stage("text", "text shortening", text, 9),
        Stage("vlm", "VLM enrichment", vlm, 12),
        Stage("overhead", "endpoint-matched overhead", overhead, 6),
    ]


def a100_stages() -> list[Stage]:
    manifest = SCALING / "workload" / "publication_manifest.json"
    prefetched = SCALING / "workload" / "a100_model_prefetch.json"
    scaling = count_rep_summaries(SCALING / "results" / "a100")
    overhead = csv_pairs(OVERHEAD / "results" / "a100" / "raw_results.csv")
    return [
        Stage("verify", "H100 artifact verification", int(manifest.is_file()), 1),
        Stage("prefetch", "A100 exact model prefetch", int(prefetched.is_file()), 1),
        Stage("scaling", "A100 4-GPU transfer", scaling, 12),
        Stage("overhead", "endpoint-matched overhead", overhead, 6),
    ]


def infer_suite(lines: list[str]) -> str:
    if active_line(lines, ["publication/run_a100.sh"]):
        return "a100"
    if active_line(lines, ["publication/run_h100.sh"]):
        return "h100"
    gpu = run_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if "A100" in gpu and "H100" not in gpu:
        return "a100"
    return "h100"


def assign_status(stages: list[Stage], lines: list[str], suite: str) -> None:
    needles = {
        "prepare": ["prepare_workload.py"],
        # A100 artifact verification happens inline in the shell driver rather
        # than in a durable child process. Its manifest presence is therefore
        # used as the durable indicator; do not match the long-lived driver.
        "verify": [],
        "prefetch": ["prefetch_models.py"],
        # The generic scaling runner is reused by text shortening, so the
        # orchestrator stage is the unambiguous scaling-process marker.
        "scaling": ["orchestrate.py --stage scaling"],
        # Include the next argument so recovery does not also match the
        # recovery_extract orchestrator command by substring.
        "recovery": ["orchestrate.py --stage recovery --repetitions", "run_local.py", "run_native_recovery_publication.py", "run_native_recovery_competitor.py"],
        "extract": ["extract_results.py"],
        "text": ["orchestrate.py --stage text"],
        "vlm": ["orchestrate.py --stage vlm", "run_mmirage_vlm.py", "run_native_vlm_competitor.py"],
        "overhead": ["sglang_overhead/scripts/run.py", "raw_sglang_client.py", "run_mmirage_with_sglang_endpoint.py"],
    }
    active_index = None
    for i, stage in enumerate(stages):
        if active_line(lines, needles.get(stage.key, [])):
            active_index = i
            break
    for i, stage in enumerate(stages):
        if stage.done >= stage.total:
            stage.status = "ok"
        elif active_index == i:
            stage.status = "running"
        elif active_index is not None and i < active_index:
            stage.status = "ok"
            stage.done = stage.total
        else:
            stage.status = "queued"


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def estimate_remaining(stages: list[Stage], suite: str) -> float:
    priors = H100_PRIORS if suite == "h100" else A100_PRIORS
    total = 0.0
    for stage in stages:
        prior = priors.get(stage.key, 0)
        total += prior * (1.0 - stage.fraction)
    return total


def gpu_lines() -> list[str]:
    text = run_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
    return text.splitlines() if text else []


def bar(fraction: float, width: int = 12) -> str:
    fill = round(width * min(1.0, max(0.0, fraction)))
    return "█" * fill + "░" * (width - fill)


def snapshot(suite: str, shared: Path, no_gpu: bool) -> dict:
    lines = processes()
    if suite == "auto":
        suite = infer_suite(lines)
    stages = h100_stages(shared) if suite == "h100" else a100_stages()
    assign_status(stages, lines, suite)
    driver = active_line(lines, [f"publication/run_{suite}.sh"])
    nested = active_line(lines, ["orchestrate.py", "prepare_workload.py", "prefetch_models.py", "run_native_recovery", "run_mmirage_vlm.py", "run_native_vlm_competitor.py", "sglang_overhead/scripts/run.py"])
    return {
        "suite": suite,
        "stages": [{"key": s.key, "label": s.label, "status": s.status, "done": s.done, "total": s.total, "fraction": s.fraction} for s in stages],
        "estimated_remaining_seconds": round(estimate_remaining(stages, suite)),
        "driver_process": driver,
        "active_process": nested,
        "gpus": [] if no_gpu else gpu_lines(),
    }


def render(payload: dict) -> str:
    out = [f"MMIRAGE publication progress · {payload['suite'].upper()}", "", "stage                            status     progress     bar", "----------------------------------------------------------------------"]
    for s in payload["stages"]:
        out.append(f"{s['label']:<32} {s['status']:<10} {s['done']}/{s['total']:<8} {bar(s['fraction'])}")
    out += ["", f"estimated remaining (prior-based): {fmt_duration(payload['estimated_remaining_seconds'])}"]
    if payload.get("active_process"):
        out.append("now: " + payload["active_process"])
    elif payload.get("driver_process"):
        out.append("driver: " + payload["driver_process"])
    if payload["gpus"]:
        out += ["", "[GPUs]"] + ["  " + line for line in payload["gpus"]]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["auto", "h100", "a100"], default="auto")
    p.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()
    while True:
        payload = snapshot(args.suite, args.recovery_root, args.no_gpu)
        text = json.dumps(payload, indent=2) if args.json else render(payload)
        if args.once or not sys.stdout.isatty():
            print(text)
            return 0
        print("\033[2J\033[H" + text, end="", flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
