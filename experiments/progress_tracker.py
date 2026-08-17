#!/usr/bin/env python3
"""Live progress tracker for experiments/a_matrix/scripts/run_setup.py.

Reads only: /proc process trees, experiments/run_all_logs/a_matrix/**/*.log,
experiments/run_all_logs/a_matrix/status.json, the captured run_setup output
(run_all.out, run_all_logs/<stage>.log, or --out), smoke/calibration.json,
and nvidia-smi. Never signals or writes into the running experiment.

The tracker mirrors experiments/run_all.sh/../progress_tracker.py but for the
A-matrix scheduler: the run_setup.py "plan" takes the place of the run_all.sh
stages, so the unit table below is the plan from run_setup.py build_plan()
(plus any units discovered in logs / status.json). Units are skipped
(reused) exactly when run_setup.py is launched with --reuse-fastruns.

Usage:
  python experiments/progress_tracker.py            # live dashboard (TTY)
  python experiments/progress_tracker.py --once     # single snapshot
  python experiments/progress_tracker.py --json     # machine-readable snapshot
  python experiments/progress_tracker.py --setup recovery
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
A_MATRIX_DIR = HERE / "a_matrix"
LOG_ROOT = HERE / "run_all_logs" / "a_matrix"
STATUS_FILE = LOG_ROOT / "status.json"
CALIBRATION = HERE / "smoke" / "calibration.json"
TEXT_DIR = HERE / "task_comparison" / "text_shortening"
VLM_DIR = HERE / "task_comparison" / "vlm_enrichment"
DEFAULT_OUT = REPO_ROOT / "run_all.out"
DEFAULT_RECOVERY_ROOT = os.environ.get("MMIRAGE_RECOVERY_ROOT", "/workspace/mmirage-recovery")

SCALING_FRAMEWORKS = ["mmirage", "raw_sglang", "datatrove", "nemo_curator"]
RECOVERY_FRAMEWORKS = ["raw_sglang", "datatrove", "nemo_curator", "distilabel", "ray_data_llm"]
RECOVERY_MMIRAGE_CONDITIONS = ["baseline", "fail_1", "fail_4"]
RECOVERY_NATIVE_CONDITIONS = ["fail_1", "fail_4"]
TEXT_FRAMEWORKS = ["mmirage", "datatrove", "nemo_curator"]
VLM_FRAMEWORKS = ["mmirage", "sglang", "datatrove", "nemo_curator"]
SETUP_ALIASES = {"text": "text_shortening", "vlm": "vlm_enrichment"}
EXPECTED_KEYS = {
    "gpu_scaling": "single_node_h100_scaling",
    "a100_4gpu": "single_node_h100_scaling",
    "recovery": "shard_recovery",
    "text_shortening": "task_comparison/text_shortening",
    "vlm_enrichment": "task_comparison/vlm_enrichment",
}

CLEAR = "\x1b[2J\x1b[H"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"

STALL_AFTER = 600
HARD_STALL_AFTER = 1800

STEP_RE = re.compile(r"^\[run_setup\] (\S+) step: (.*)$")
FINISH_RE = re.compile(r"^\[run_setup\] (\S+) step finished rc=(\d+)$")
HEADER_RE = re.compile(r"run_setup: pod=(\S+) setups=\[(.*?)\] gpus=\[(.*?)\] mode=(\S+)")


@dataclass
class Unit:
    label: str
    setup: str
    framework: str
    tag: str
    gpus: int
    cmds: int


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(max(0.0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_eta(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(max(0.0, sec))
    if sec >= 3600:
        return f"≈ {sec // 3600}h {(sec % 3600) // 60}m"
    return f"≈ {sec // 60}m {sec % 60:02d}s"


def opt(joined: str, flag: str) -> str | None:
    match = re.search(re.escape(flag) + r"(?:=|\s+)(\S+)", joined)
    return match.group(1) if match else None


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        parts = [p for p in raw.split(b"\x00") if p]
        return [p.decode(errors="replace") for p in parts]
    except OSError:
        return []


def proc_meta(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    idx = stat.rfind(")")
    rest = stat[idx + 2:].split()
    try:
        start_ticks = int(rest[19])
        cpu = int(rest[11]) + int(rest[12])
    except (IndexError, ValueError):
        return None
    return {"start": start_ticks, "cpu": cpu}


def boot_time() -> int:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(time.time())


def descendants(root_pid: int) -> list[dict]:
    children: dict[int, list[int]] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = entry.joinpath("stat").read_text()
            except OSError:
                continue
            idx = stat.rfind(")")
            try:
                ppid = int(stat[idx + 2:].split()[1])
            except (IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry.name))
    except OSError:
        return []
    found: list[dict] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        meta = proc_meta(pid)
        if meta is None:
            continue
        found.append({"pid": pid, "cmd": read_cmdline(pid), **meta})
        stack.extend(children.get(pid, []))
    return found


def _is_run_setup(pid: int) -> bool:
    cmd = read_cmdline(pid)
    if not cmd:
        return False
    if Path(cmd[0]).name in ("bash", "sh"):
        return False
    return "run_setup.py" in " ".join(cmd)


def find_run_setup(args: argparse.Namespace) -> int | None:
    try:
        raw = subprocess.run(["pgrep", "-f", "run_setup.py"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return None
    candidates = [int(p) for p in raw.split()]
    candidates = [p for p in candidates if _is_run_setup(p)]
    if not candidates:
        return None
    if args.pod or args.setup:
        wanted = [c for c in candidates if _matches_filter(c, args)]
        if not wanted:
            return None
        candidates = wanted
    alive = [c for c in candidates if proc_meta(c) is not None]
    if not alive:
        return None
    return max(alive, key=lambda p: proc_meta(p)["start"])


def _matches_filter(pid: int, args: argparse.Namespace) -> bool:
    parsed = parse_rs_cmd(read_cmdline(pid))
    if not parsed:
        return False
    if args.pod and parsed["pod"] != args.pod:
        return False
    if args.setup:
        name = SETUP_ALIASES.get(args.setup, args.setup)
        if name not in (parsed["setups"] or []):
            return False
    return True


def parse_rs_cmd(cmd: list[str]) -> dict | None:
    joined = " ".join(cmd)
    if "run_setup.py" not in joined:
        return None
    setup = opt(joined, "--setup")
    pod = opt(joined, "--pod")
    gpus = opt(joined, "--gpus") or "0,1,2,3"
    reps_raw = opt(joined, "--repetitions")
    return {
        "pod": pod,
        "setup": setup,
        "gpus": [g.strip() for g in gpus.split(",") if g.strip()],
        "reps": int(reps_raw) if reps_raw else 3,
        "reuse": "--reuse-fastruns" in joined,
        "mode": "prepare" if "--prepare" in joined
        else ("extract" if "--extract" in joined
              else ("dry-run" if "--dry-run" in joined else "run")),
        "setups": selected_setups(pod, setup),
    }


def selected_setups(pod: str | None, setup: str | None) -> list[str]:
    if setup:
        name = SETUP_ALIASES.get(setup, setup)
        return [name] if name in EXPECTED_KEYS else []
    if pod:
        return read_schedule_setups(pod)
    return []


def read_schedule_setups(pod: str) -> list[str]:
    fallback = {
        "pod_a": ["gpu_scaling", "recovery"],
        "pod_b": ["gpu_scaling", "text_shortening", "vlm_enrichment"],
    }
    try:
        text = (A_MATRIX_DIR / "schedule.yaml").read_text()
    except OSError:
        return fallback.get(pod, [])
    block = re.search(rf"^{re.escape(pod)}:.*?(?=^\S|\Z)", text, re.M | re.S)
    if not block:
        return fallback.get(pod, [])
    body = block.group(0)
    m = re.search(r"^setup:\s*(\w+)", body, re.M)
    if not m:
        return fallback.get(pod, [])
    setups = [m.group(1)]
    em = re.search(r"^extra:\s*\[(.*?)\]", body, re.M | re.S)
    if em:
        setups += [s.strip() for s in em.group(1).split(",") if s.strip()]
    return setups


def build_units(inv: dict) -> list[Unit]:
    selected = inv["setups"]
    units: list[Unit] = []

    def add(setup: str, framework: str, tag: str, gpus: int, cmds: int) -> None:
        label = f"{setup}/{framework}/{tag}".rstrip("/")
        units.append(Unit(label=label, setup=setup, framework=framework, tag=tag, gpus=gpus, cmds=cmds))

    if "gpu_scaling" in selected:
        points = [1, 2, 4]
        if inv.get("pod") == "pod_a":
            points = [1, 2]
        elif inv.get("pod") == "pod_b":
            points = [4]
        for gpu_count in points:
            for framework in SCALING_FRAMEWORKS:
                add("gpu_scaling", framework, f"gpu_{gpu_count}", gpu_count, 1)

    if "a100_4gpu" in selected:
        for framework in SCALING_FRAMEWORKS:
            add("a100_4gpu", framework, "gpu_4", 4, 1)

    if "recovery" in selected:
        for condition in RECOVERY_MMIRAGE_CONDITIONS:
            add("recovery", "mmirage", condition, 4, 2 if condition == "baseline" else 3)
        for framework in RECOVERY_FRAMEWORKS:
            for condition in RECOVERY_NATIVE_CONDITIONS:
                add("recovery", framework, condition, 4, 1)

    if "text_shortening" in selected:
        for framework in TEXT_FRAMEWORKS:
            add("text_shortening", framework, "", 4, 1)

    if "vlm_enrichment" in selected:
        for framework in VLM_FRAMEWORKS:
            add("vlm_enrichment", framework, "", 4, 1)

    return units


def unit_from_label(label: str) -> Unit:
    parts = label.split("/")
    setup = parts[0]
    if setup in ("gpu_scaling", "a100_4gpu", "recovery"):
        framework = parts[1] if len(parts) > 1 else "?"
        tag = parts[2] if len(parts) > 2 else ""
        gpus = 4
        if setup in ("gpu_scaling", "a100_4gpu"):
            m = re.fullmatch(r"gpu_(\d+)", tag)
            gpus = int(m.group(1)) if m else 4
        cmds = 1
        if setup == "recovery" and framework == "mmirage":
            cmds = 2 if tag == "baseline" else 3
    else:
        framework = parts[1] if len(parts) > 1 else "?"
        tag = ""
        gpus, cmds = 4, 1
    return Unit(label=label, setup=setup, framework=framework, tag=tag, gpus=gpus, cmds=cmds)


def short_label(unit: Unit) -> str:
    if unit.setup == "gpu_scaling":
        return f"{unit.framework}/{unit.tag}"
    if unit.setup == "a100_4gpu":
        return unit.framework
    if unit.setup == "recovery":
        return f"{unit.framework}/{unit.tag}"
    return unit.framework


def load_status_json() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def reused_labels() -> set[str]:
    path = A_MATRIX_DIR / "configs" / "reused_units.yaml"
    try:
        text = path.read_text()
    except OSError:
        return set()
    return set(re.findall(r"^\s*-\s+label:\s*(\S+)", text, re.M))


def scan_known_labels() -> set[str]:
    labels = set()
    for log in LOG_ROOT.rglob("*.log"):
        rel = log.relative_to(LOG_ROOT).with_suffix("").as_posix()
        parts = rel.split("/", 1)
        if len(parts) == 2:
            labels.add(parts[1])
    labels.update(load_status_json())
    return labels


def unit_log_path(label: str) -> Path | None:
    for base in sorted(LOG_ROOT.glob("*")):
        if not base.is_dir():
            continue
        path = base / f"{label}.log"
        if path.exists():
            return path
    return None


def _parse_ts(iso: str) -> float | None:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def read_steps(path: Path | None, since_epoch: float | None = None) -> list[dict] | None:
    if path is None:
        return None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    steps: list[dict] = []
    for line in lines:
        m = STEP_RE.match(line)
        if m:
            steps.append({"cmd": m.group(2), "start": _parse_ts(m.group(1)), "finish": None, "rc": None})
            continue
        m = FINISH_RE.match(line)
        if m and steps and steps[-1]["finish"] is None:
            steps[-1]["finish"] = _parse_ts(m.group(1))
            steps[-1]["rc"] = int(m.group(2))
    if since_epoch is not None:
        kept = [s for s in steps if s["start"] is None or s["start"] >= since_epoch - 1.0]
        if not kept:
            return None
        return kept
    return steps


def unit_elapsed(steps: list[dict], now: float) -> float | None:
    total = 0.0
    seen = False
    for step in steps:
        start = step["start"]
        if start is None:
            continue
        end = step["finish"] if step["finish"] is not None else now
        total += max(0.0, end - start)
        seen = True
    return total if seen else None


def derive_statuses(units: list[Unit], inv: dict, alive: bool, since_epoch: float | None, now: float) -> tuple[dict, dict]:
    status_json = load_status_json() if not alive else {}
    status_json_mtime = STATUS_FILE.stat().st_mtime if STATUS_FILE.exists() else 0.0
    reused = reused_labels()
    status: dict[str, str] = {}
    elapsed: dict[str, float | None] = {}

    if inv["mode"] in ("prepare", "extract"):
        for unit in units:
            status[unit.label] = "waiting"
            elapsed[unit.label] = None
        return status, elapsed

    for unit in units:
        steps = read_steps(unit_log_path(unit.label), since_epoch)
        elapsed[unit.label] = unit_elapsed(steps, now) if steps else None
        if steps is None:
            if not alive:
                entry = status_json.get(unit.label)
                status[unit.label] = "ok" if (entry or {}).get("status") == "ok" else (
                    "failed" if entry else "skipped")
            else:
                status[unit.label] = "pending"
            continue
        if not alive:
            entry = status_json.get(unit.label)
            newer_run = any(s["start"] is not None and s["start"] > status_json_mtime for s in steps)
            if entry and entry.get("status") in ("ok", "failed") and not newer_run:
                status[unit.label] = entry["status"]
                continue
            if not steps:
                status[unit.label] = "ok" if (entry or {}).get("status") == "ok" else (
                    "failed" if entry else "skipped")
                continue
            recent = steps[-unit.cmds:]
            if any(s["rc"] is not None and s["rc"] != 0 for s in recent):
                status[unit.label] = "failed"
                continue
            open_step = steps[-1]["rc"] is None
            all_done = len(steps) >= unit.cmds
            if open_step:
                last_start = steps[-1]["start"]
                stale = last_start is None or (now - last_start) > 10800
                status[unit.label] = "aborted" if stale else "running"
            else:
                status[unit.label] = "aborted" if not all_done else "ok"
            continue
        if any(s["rc"] is not None and s["rc"] != 0 for s in steps):
            status[unit.label] = "failed"
            continue
        if not steps:
            status[unit.label] = "pending"
            continue
        open_step = steps[-1]["rc"] is None
        all_done = len(steps) >= unit.cmds
        if open_step or not all_done:
            status[unit.label] = "running"
        else:
            status[unit.label] = "ok"

    if alive:
        running = [i for i, u in enumerate(units) if status[u.label] == "running"]
        if running:
            first = running[0]
            for i in range(first + 1, len(units)):
                if status[units[i].label] == "pending":
                    status[units[i].label] = "queued"

    if inv["reuse"]:
        for unit in units:
            if unit.label in reused:
                status[unit.label] = "reused"
    return status, elapsed


def expected_seconds() -> dict:
    cal: dict = {}
    try:
        cal = json.loads(CALIBRATION.read_text()).get("calibrations", {})
    except (OSError, ValueError):
        pass

    def per_cell(name: str) -> float | None:
        node = cal.get(name) or {}
        wall = node.get("expected_wall_seconds")
        if not wall:
            return None
        return float(wall)

    out: dict[str, float | None] = {}
    for setup, key in EXPECTED_KEYS.items():
        out[setup] = per_cell(key)
    return out


def unit_expected(unit: Unit, per_cell: dict) -> float | None:
    base = per_cell.get(unit.setup)
    if base is None:
        return None
    if unit.setup == "recovery" and unit.framework == "mmirage" and unit.tag != "baseline":
        return base * 2.2
    return base


def sub_progress(unit: Unit, recovery_root: str, reps: int) -> str | None:
    if unit.setup in ("gpu_scaling", "a100_4gpu"):
        base = A_MATRIX_DIR / "results" / unit.setup
        if unit.framework == "mmirage":
            runs = base / "mmirage" / "runs" / unit.tag
        else:
            runs = base / unit.framework / "runs" / unit.tag
        done = sum(1 for _ in runs.glob("rep_*/rep_summary.json"))
        return f"{done}/{reps} reps"
    if unit.setup == "recovery":
        if unit.framework == "mmirage":
            rd = Path(recovery_root) / "runs" / unit.tag / "rep_01"
            if (rd / "merged").is_dir():
                return "merged"
        else:
            rd = Path(recovery_root) / "native_competitors" / unit.framework / unit.tag / "rep_01"
            if (rd / "merged" / "merged.jsonl").exists():
                return "merged"
        files = sorted(rd.glob("controller/completed_shards_*.json"))
        if files:
            try:
                data = json.loads(files[-1].read_text())
                shards = data.get("shards") if unit.framework == "mmirage" else list(data.get("snapshot", {}))
                return f"{len(shards)}/16 shards"
            except (OSError, ValueError):
                pass
        return "starting"
    if unit.setup == "text_shortening":
        if unit.framework == "mmirage":
            runs = TEXT_DIR / "results" / "runs" / "gpu_4"
        else:
            runs = TEXT_DIR / "results" / "native_competitors" / unit.framework / "runs" / "gpu_4"
        done = sum(1 for _ in runs.glob("rep_*/rep_summary.json"))
        return f"{done}/{reps} reps"
    if unit.setup == "vlm_enrichment":
        if unit.framework == "mmirage":
            runs = VLM_DIR / "results" / "runs" / "gpu_4"
        else:
            runs = VLM_DIR / "results" / "native_competitors" / unit.framework / "runs" / "gpu_4"
        done = sum(1 for _ in runs.glob("rep_*/rep_summary.json"))
        return f"{done}/{reps} reps"
    return None


def step_label(cmd: list[str]) -> str | None:
    joined = " ".join(cmd)
    if "run_local.py" in joined:
        cond = opt(joined, "--condition") or "?"
        if re.search(r"\brun-condition\b", joined):
            return f"recovery · mmirage {cond} (run)"
        if re.search(r"\bretry\b", joined):
            return f"recovery · mmirage {cond} (retry)"
        return "recovery · mmirage"
    if re.search(r"\bmerge-dir\b", joined):
        return "recovery · mmirage (merge)"
    if "run_native_recovery_competitor.py" in joined:
        return f"recovery · {opt(joined, '--framework') or '?'} {opt(joined, '--condition') or '?'}"
    if "extract_results.py" in joined:
        return "recovery · extract"
    if "run_mmirage_vlm.py" in joined:
        return "vlm · mmirage 4-GPU"
    if "run_native_vlm_competitor.py" in joined:
        return f"vlm · native {opt(joined, '--framework') or '?'}"
    if "run_datatrove_scaling.py" in joined or "run_nemo_curator_scaling.py" in joined or "run_raw_sglang_scaling.py" in joined:
        if "run_raw_sglang_scaling.py" in joined:
            fw = "raw_sglang"
        else:
            fw = "datatrove" if "run_datatrove_scaling.py" in joined else "nemo_curator"
        n = opt(joined, "--gpu-count") or "?"
        if "summarize" in (opt(joined, "--prompt-style") or ""):
            return f"text · native {fw} 4-GPU"
        return f"scaling · {fw} {n}-GPU"
    if "run.py" in joined:
        cfg = opt(joined, "--execution-config") or ""
        if "text_shortening" in cfg:
            return "text · mmirage 4-GPU"
        if "vlm_enrichment" in cfg:
            return "vlm · mmirage 4-GPU"
        return f"scaling · mmirage {opt(joined, '--gpu-count') or '?'}-GPU"
    if "prepare_workload.py" in joined:
        return "preparing workload"
    return None


def gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                rows.append({
                    "index": int(parts[0]),
                    "util": int(parts[1]),
                    "used_gb": round(int(parts[2]) / 1024, 1),
                    "total_gb": round(int(parts[3]) / 1024, 1),
                })
            except ValueError:
                continue
    return rows


def capture_sources(args: argparse.Namespace, pid: int | None) -> list[Path]:
    sources: list[Path] = []
    if args.out:
        sources.append(Path(args.out))
    if pid:
        try:
            target = os.readlink(f"/proc/{pid}/fd/1")
            if os.path.isfile(target):
                sources.append(Path(target))
        except OSError:
            pass
    if DEFAULT_OUT.exists():
        sources.append(DEFAULT_OUT)
    sources.extend(sorted(REPO_ROOT.glob("run_setup*.out")))
    sources.extend(sorted(HERE.glob("run_all_logs/*.log")))
    seen: set[Path] = set()
    out: list[Path] = []
    for src in sources:
        resolved = src.resolve() if src.exists() else src
        if resolved not in seen:
            seen.add(resolved)
            out.append(src)
    return out


def invocation_from_captures(args: argparse.Namespace, pid: int | None) -> dict | None:
    found: list[dict] = []
    for source in capture_sources(args, pid):
        try:
            lines = source.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            m = HEADER_RE.search(line)
            if not m:
                continue
            pod = None if m.group(1) == "None" else m.group(1)
            setups = re.findall(r"'([^']*)'", m.group(2))
            gpus = re.findall(r"\d+", m.group(3))
            mode = m.group(4)
            reuse = "--reuse-fastruns" in line
            if not reuse:
                for nxt in lines[i + 1:i + 60]:
                    if HEADER_RE.search(nxt):
                        break
                    if "--reuse-fastruns" in nxt:
                        reuse = True
                        break
            found.append({"pod": pod, "setup": None, "setups": setups, "gpus": gpus,
                          "reps": 3, "reuse": reuse,
                          "mode": mode, "out_file": str(source)})
    if not found:
        return None
    if args.pod or args.setup:
        wanted_name = SETUP_ALIASES.get(args.setup, args.setup) if args.setup else None
        for entry in reversed(found):
            if args.pod and entry["pod"] != args.pod:
                continue
            if wanted_name and wanted_name not in entry["setups"]:
                continue
            return entry
        return None
    return found[-1]


def invocation_from_schedule(args: argparse.Namespace) -> dict | None:
    """Synthesize an invocation from the schedule file when no run_setup header
    is visible (e.g. monitoring a remote node sharing this filesystem)."""
    pod = args.pod
    if not pod:
        return None
    if args.setup:
        name = SETUP_ALIASES.get(args.setup, args.setup)
        if name not in EXPECTED_KEYS:
            return None
        setups = [name]
    else:
        setups = read_schedule_setups(pod)
    return {"pod": pod, "setup": None, "setups": setups, "gpus": ["0", "1", "2", "3"],
            "reps": 3, "reuse": False, "mode": "run", "out_file": None}


def snapshot(args: argparse.Namespace, last_pid: int | None = None, last_cpu: int | None = None) -> dict:
    pid = args.pid or find_run_setup(args)
    btime = boot_time()
    hz = float(os.sysconf("SC_CLK_TCK"))
    now = time.time()

    alive = pid is not None
    rs_elapsed = None
    since_epoch = None
    invocation: dict | None = None
    youngest: dict | None = None
    if pid:
        meta = proc_meta(pid)
        if meta is not None:
            rs_elapsed = now - (btime + meta["start"] / hz)
            since_epoch = btime + meta["start"] / hz
        parsed = parse_rs_cmd(read_cmdline(pid))
        if parsed:
            invocation = {"pod": parsed["pod"], "setup": parsed["setup"], "setups": parsed["setups"],
                          "gpus": parsed["gpus"], "reps": parsed["reps"], "reuse": parsed["reuse"],
                          "mode": parsed["mode"], "out_file": None}
        proc_list = [p for p in descendants(pid) if p["cmd"]]
        non_bash = [p for p in proc_list if p["cmd"] and Path(p["cmd"][0]).name not in ("bash", "sh")]
        known = [p for p in proc_list if step_label(p["cmd"]) is not None]
        pythonish = [p for p in non_bash if re.search(r"(python|mmirage|sglang)", p["cmd"][0])]
        pool = known or pythonish or non_bash
        youngest = max(pool, key=lambda p: p["start"]) if pool else None
    else:
        invocation = invocation_from_captures(args, None)

    if invocation is None:
        invocation = invocation_from_schedule(args)
    if invocation is None:
        invocation = {"pod": None, "setup": None, "setups": [], "gpus": ["0", "1", "2", "3"],
                      "reps": 3, "reuse": False, "mode": "run", "out_file": None}

    units = build_units(invocation)
    allowed = set(invocation["setups"]) if invocation["setups"] else None
    pod_points = None
    if invocation.get("pod") == "pod_a":
        pod_points = {1, 2}
    elif invocation.get("pod") == "pod_b":
        pod_points = {4}
    known_labels = {u.label for u in units}
    for label in sorted(scan_known_labels() - known_labels):
        known = unit_from_label(label)
        if allowed and known.setup not in allowed:
            continue
        if pod_points and known.setup == "gpu_scaling":
            m = re.fullmatch(r"gpu_(\d+)", known.tag)
            if m and int(m.group(1)) not in pod_points:
                continue
        units.append(known)
    status, elapsed = derive_statuses(units, invocation, alive, since_epoch, now)

    per_cell = expected_seconds()
    expected: dict[str, float | None] = {u.label: unit_expected(u, per_cell) for u in units}
    approx = True

    active_unit = None
    if alive and invocation["mode"] == "run":
        running = [u for u in units if status[u.label] == "running"]
        if running:
            best, best_mtime = None, None
            for u in running:
                log = unit_log_path(u.label)
                if log is None:
                    continue
                try:
                    mtime = log.stat().st_mtime
                except OSError:
                    continue
                if best_mtime is None or mtime > best_mtime:
                    best, best_mtime = u, mtime
            active_unit = best

    units_out: dict[str, dict] = {}
    for unit in units:
        st = status[unit.label]
        exp = expected[unit.label]
        frac = None
        if exp:
            frac = 1.0 if st in ("ok", "failed") else min(1.0, (elapsed[unit.label] or 0.0) / exp)
        sub = None
        if st == "reused":
            sub = "reused from fast-runs archive"
        elif st in ("running", "ok", "failed"):
            sub = sub_progress(unit, args.recovery_root, invocation["reps"])
        units_out[unit.label] = {
            "label": short_label(unit),
            "setup": unit.setup,
            "status": st,
            "elapsed_seconds": elapsed[unit.label],
            "expected_seconds": exp,
            "approximate_expected": approx,
            "bar": frac,
            "sub_progress": sub,
        }

    eta = None
    if alive:
        remaining = 0.0
        known = False
        for unit in units:
            st = status[unit.label]
            exp = expected[unit.label]
            if st == "queued" and exp is not None:
                remaining += exp
                known = True
            elif st == "running" and exp is not None:
                remaining += max(0.0, exp - (elapsed[unit.label] or 0.0))
                known = True
        if known:
            eta = remaining

    current = None
    if alive:
        log_age = None
        stall = False
        cpu_advancing: bool | None = None
        if youngest:
            if last_pid == youngest["pid"] and last_cpu is not None:
                cpu_advancing = youngest["cpu"] > last_cpu
        active_log = unit_log_path(active_unit.label) if active_unit else None
        if active_log is not None:
            try:
                log_age = now - active_log.stat().st_mtime
            except OSError:
                pass
        if active_log is not None and log_age is not None:
            if log_age > HARD_STALL_AFTER:
                stall = True
            elif log_age > STALL_AFTER and cpu_advancing is False:
                stall = True
        step = None
        step_elapsed = None
        if youngest:
            step = step_label(youngest["cmd"])
            if step is None:
                step = f"{Path(youngest['cmd'][0]).name} …"
            step_elapsed = now - (btime + youngest["start"] / hz)
        current = {
            "pid": youngest["pid"] if youngest else None,
            "unit": short_label(active_unit) if active_unit else None,
            "step": step,
            "step_elapsed_seconds": step_elapsed,
            "log_age_seconds": log_age,
            "stall": stall,
            "cpu": youngest["cpu"] if youngest else None,
        }

    return {
        "run_setup": {
            "pid": pid,
            "alive": alive,
            "elapsed_seconds": rs_elapsed,
            "pod": invocation["pod"],
            "setups": invocation["setups"],
            "gpus": invocation["gpus"],
            "mode": invocation["mode"],
            "reuse_fastruns": invocation["reuse"],
            "out_file": invocation["out_file"],
        },
        "units": units_out,
        "unit_order": [u.label for u in units],
        "setup_order": list(dict.fromkeys(invocation["setups"] + [u.setup for u in units])),
        "active_unit": active_unit.label if active_unit else None,
        "current": current,
        "eta_remaining_seconds": eta,
        "eta_approximate": approx,
        "gpus": gpu_stats() if not args.no_gpu else [],
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }


STATUS_COLOR = {"ok": GREEN, "running": CYAN, "failed": RED, "skipped": DIM, "queued": DIM,
                "pending": DIM, "aborted": RED, "reused": YELLOW, "waiting": DIM}
STATUS_TEXT = {"waiting": "—"}


def render(info: dict, tty: bool, interval: float = 5.0) -> str:
    lines: list[str] = []
    rs = info["run_setup"]
    if not rs["alive"] and rs["pid"] is None and not rs["out_file"] and not info["units"]:
        lines.append("No run_setup.py process found and no run_setup output to read.")
        lines.append("Start it with: nohup python experiments/a_matrix/scripts/run_setup.py --setup recovery > run_setup.out 2>&1 &")
        return "\n".join(lines)

    pid_txt = str(rs["pid"]) if rs["pid"] else "—"
    out_txt = rs["out_file"] or "—"
    pod_txt = rs["pod"] or "—"
    setups_txt = ", ".join(rs["setups"]) if rs["setups"] else "—"
    gpus_txt = ", ".join(rs["gpus"]) if rs["gpus"] else "—"
    header = (f"{BOLD}run_setup{RESET}: up {fmt_dur(rs['elapsed_seconds'])} · PID {pid_txt} "
              f"· pod {pod_txt} · setups [{setups_txt}] · mode {rs['mode']} · gpus {gpus_txt}")
    if not tty:
        header = header.replace(BOLD, "").replace(RESET, "")
    lines.append(header)
    if rs["out_file"]:
        lines.append(f"{DIM}out {out_txt}{RESET}" if tty else f"out {out_txt}")
    if rs["mode"] == "dry-run":
        lines.append(f"{DIM}dry-run: plan only, nothing was executed{RESET}" if tty else "dry-run: plan only, nothing was executed")
    if rs["mode"] in ("prepare", "extract"):
        lines.append(f"{DIM}mode {rs['mode']}: units below are part of the setup but are not scheduled{RESET}"
                     if tty else f"mode {rs['mode']}: units below are part of the setup but are not scheduled")
    lines.append("")

    if not info["units"]:
        lines.append("(no units known yet — run_setup just started, or use --out to point at its captured output)")
        return "\n".join(lines)

    unit_row = " " * 20
    header_row = f"{'unit':<20} {'status':<8} {'elapsed':<10} {'expected':<10} bar"
    for setup in info["setup_order"]:
        group = [label for label in info["unit_order"] if info["units"][label]["setup"] == setup]
        if not group:
            continue
        lines.append(f"{BOLD}[{setup}]{RESET}" if tty else f"[{setup}]")
        lines.append(header_row if not tty else f"{BOLD}{header_row}{RESET}")
        lines.append("-" * 56)
        for label in group:
            s = info["units"][label]
            st = s["status"]
            st_txt = STATUS_TEXT.get(st, st)
            exp_txt = fmt_dur(s["expected_seconds"])
            bar_txt = ""
            if s["bar"] is not None:
                filled = round(s["bar"] * 10)
                bar_txt = "█" * filled + "░" * (10 - filled)
            elif st in ("skipped", "queued", "pending", "waiting"):
                bar_txt = "·" * 10
            over = st == "running" and s["expected_seconds"] and (s["elapsed_seconds"] or 0) > s["expected_seconds"] * 1.05
            if tty:
                color = STATUS_COLOR.get(st, "")
                elapsed_txt = fmt_dur(s["elapsed_seconds"])
                if over:
                    elapsed_txt = f"{YELLOW}{elapsed_txt} ⏱ over{RESET}"
                cell = f"{color}{s['label']:<20}{RESET} {color}{st_txt:<8}{RESET} {elapsed_txt:<10} {exp_txt:<10} {bar_txt}"
            else:
                elapsed_txt = fmt_dur(s["elapsed_seconds"]) + (" ⏱" if over else "")
                cell = f"{s['label']:<20} {st_txt:<8} {elapsed_txt:<12} {exp_txt:<10} {bar_txt}"
            lines.append(cell)
            if s["sub_progress"]:
                lines.append(f"{' ' * 21}{DIM}{s['sub_progress']}{RESET}" if tty else f"{' ' * 21}{s['sub_progress']}")
        lines.append("")

    eta = fmt_eta(info["eta_remaining_seconds"])
    lines.append(f"{BOLD}ETA{RESET}: remaining {eta} (rough)" if tty else f"ETA: remaining {eta} (rough)")

    cur = info["current"]
    if cur and cur["step"]:
        step_txt = cur["step"]
        if cur["unit"]:
            step_txt = f"{cur['unit']} → {step_txt}"
        step_txt += f" · {fmt_dur(cur['step_elapsed_seconds'])}"
        if cur["log_age_seconds"] is not None:
            step_txt += f" · log written {fmt_dur(cur['log_age_seconds'])} ago"
        lines.append("")
        lines.append(f"{CYAN}now{RESET}: {step_txt}" if tty else f"now: {step_txt}")
        if cur["stall"]:
            lines.append(f"{RED}⚠ possible stall: no log output recently{RESET}" if tty else "WARNING: possible stall: no log output recently")
    elif cur is not None and info["active_unit"]:
        lines.append("")
        lines.append("now: starting next unit…")

    if info["gpus"]:
        lines.append("")
        cells = []
        for g in info["gpus"]:
            color = GREEN if g["util"] >= 50 else (YELLOW if g["util"] >= 10 else DIM)
            label = f"[{g['index']}] {g['util']}% {g['used_gb']}/{g['total_gb']}G"
            cells.append(f"{color}{label}{RESET}" if tty else label)
        lines.append("GPU  " + "   ".join(cells))

    if tty:
        lines.append("")
        lines.append(f"{DIM}Ctrl-C to stop · refresh every {interval}s{RESET}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON snapshot and exit.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh seconds (default 5).")
    parser.add_argument("--pid", type=int, default=None, help="Override run_setup.py PID.")
    parser.add_argument("--setup", default=None, help="Track only this setup (gpu_scaling, recovery, text, vlm, ...).")
    parser.add_argument("--pod", default=None, help="Track only this pod (pod_a | pod_b).")
    parser.add_argument("--out", default=None, help="Path to captured run_setup output (default run_all.out / stage logs).")
    parser.add_argument("--recovery-root", default=DEFAULT_RECOVERY_ROOT, help="MMIRAGE recovery shared root.")
    parser.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi polling.")
    args = parser.parse_args()

    tty = sys.stdout.isatty()

    def run_once() -> int:
        info = snapshot(args)
        if args.json:
            print(json.dumps(info, indent=2, sort_keys=True))
            return 0
        print(render(info, False))
        return 0

    if args.once or args.json or not tty:
        return run_once()

    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()

    def cleanup():
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

    atexit.register(cleanup)

    def fingerprint(info: dict) -> tuple:
        rs = info.get("run_setup") or {}
        units = tuple(
            (label, u.get("status"), u.get("sub_progress"), round(u.get("bar") or 0.0, 2))
            for label, u in sorted((info.get("units") or {}).items())
        )
        cur = info.get("current") or {}
        return (
            rs.get("alive"),
            rs.get("pid"),
            rs.get("pod"),
            tuple(rs.get("setups") or []),
            rs.get("mode"),
            info.get("active_unit"),
            units,
            cur.get("unit"),
            cur.get("step"),
            cur.get("stall"),
        )

    last_cpu: int | None = None
    last_pid: int | None = None
    last_fp: tuple | None = None
    try:
        while True:
            info = snapshot(args, last_pid=last_pid, last_cpu=last_cpu)
            cur = info.get("current") or {}
            last_pid = cur.get("pid")
            last_cpu = cur.get("cpu")
            fp = fingerprint(info)
            if fp != last_fp:
                sys.stdout.write(CLEAR + render(info, True, interval=args.interval))
                sys.stdout.flush()
                last_fp = fp
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
