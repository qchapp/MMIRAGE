#!/usr/bin/env python3
"""Analyze AnonLib vs NeMo Curator comparison outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

FRAMEWORK_FILES = {
    "anonlib": [
        "experiments/nemo_curator_comparison/anonlib/chartqa_anonlib.yaml",
        "experiments/nemo_curator_comparison/anonlib/run_anonlib_with_openai_vision_endpoint.py",
    ],
    "nemo": [
        "experiments/nemo_curator_comparison/nemo_curator/chartqa_pipeline.py",
        "experiments/nemo_curator_comparison/nemo_curator/data_designer_config.yaml",
    ],
}

PIPELINE_COMPONENTS = {"anonlib": 4, "nemo": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/nemo_curator_comparison/results")
    parser.add_argument("--expected-input-jsonl", default="experiments/nemo_curator_comparison/workload/chartqa/chartqa_subset.jsonl")
    parser.add_argument("--output-dir", default="experiments/nemo_curator_comparison/results/analysis")
    parser.add_argument("--setup-times-dir", default="experiments/nemo_curator_comparison/setup_times")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl")) if path.exists() else []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def arrow_rows(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.rglob("data-*.arrow")) if path.exists() else []
    if not files:
        return []
    try:
        from datasets import Dataset
    except ImportError:
        return []
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(Dataset.from_file(str(file)).to_list())
    return rows


def find_output_rows(run_dir: Path) -> list[dict[str, Any]]:
    output_dir = run_dir / "output"
    if (output_dir / "merged").exists():
        return jsonl_rows(output_dir / "merged")
    rows = jsonl_rows(output_dir)
    if rows:
        return rows
    return arrow_rows(output_dir)


def validate_row(row: dict[str, Any]) -> bool:
    try:
        messages = row.get("messages")
        metadata = row.get("metadata")
        if not isinstance(row.get("id"), str) or not row["id"]:
            return False
        if not isinstance(messages, list) or len(messages) != 2:
            return False
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            return False
        if not isinstance(messages[0].get("content"), list) or len(messages[0]["content"]) < 2:
            return False
        return isinstance(metadata, dict) and {"reference_answer", "generated_answer_normalized", "source"}.issubset(metadata)
    except Exception:
        return False


def collapse_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def message_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def normalization_checks(row: dict[str, Any], expected_row: dict[str, Any]) -> tuple[bool, bool]:
    messages = row["messages"]
    user_text = message_text(messages[0]["content"])
    answer = message_text(messages[1]["content"]).split("\n\nRationale:", 1)[0]
    query_ok = user_text == collapse_whitespace(expected_row["query"])
    answer_ok = row["metadata"]["generated_answer_normalized"] == collapse_whitespace(answer).lower()
    return query_ok, answer_ok


def sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "total_rows",
        "successfully_materialized_rows",
        "schema_valid_rows",
        "query_normalization_consistent_rows",
        "answer_normalization_consistent_rows",
        "total_generated_tokens",
        "generation_wall_seconds",
        "full_end_to_end_wall_seconds",
        "tok_s_gpu",
        "rows_s",
        "mean_gpu_utilization",
        "setup_seconds",
    ]
    out: list[dict[str, Any]] = []
    for framework in sorted({row["framework"] for row in raw_rows}):
        subset = [row for row in raw_rows if row["framework"] == framework]
        summary: dict[str, Any] = {"framework": framework, "repetitions": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "") and not (isinstance(row.get(metric), float) and math.isnan(row[metric]))]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_sd"] = sample_stdev(values) if values else None
        out.append(summary)
    return out


def nonblank_noncomment_lines(path: Path) -> int:
    count = 0
    in_triple = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if path.suffix == ".py":
            if line.startswith(('"""', "'''")):
                if line.count('"""') == 1 or line.count("'''") == 1:
                    in_triple = not in_triple
                continue
            if in_triple:
                if '"""' in line or "'''" in line:
                    in_triple = False
                continue
        count += 1
    return count


def implementation_footprint() -> list[dict[str, Any]]:
    rows = []
    for framework, files in FRAMEWORK_FILES.items():
        declarative = 0
        python = 0
        for file_name in files:
            path = Path(file_name)
            lines = nonblank_noncomment_lines(path)
            if path.suffix in {".yaml", ".yml", ".json"}:
                declarative += lines
            elif path.suffix == ".py":
                python += lines
        rows.append(
            {
                "framework": framework,
                "declarative_loc": declarative,
                "glue_python_loc": python,
                "user_authored_files": len(files),
                "pipeline_components": PIPELINE_COMPONENTS[framework],
                "counted_files": ";".join(files),
            }
        )
    return rows


def extract_anonlib_tokens(run_dir: Path) -> tuple[int | None, float | None, float | None]:
    status_files = sorted((run_dir / "state").rglob("status.json"))
    tokens = 0
    model_load = 0.0
    found_tokens = False
    for path in status_files:
        data = read_json(path)
        stats = data.get("stats") or {}
        if stats.get("output_tokens") is not None:
            tokens += int(stats["output_tokens"])
            found_tokens = True
        if stats.get("model_load_seconds") is not None:
            model_load += float(stats["model_load_seconds"])
    summary = read_json(run_dir / "run_summary.json")
    generation_wall = None
    if summary.get("full_end_to_end_wall_seconds") is not None and model_load:
        generation_wall = max(0.0, float(summary["full_end_to_end_wall_seconds"]) - model_load)
    return (tokens if found_tokens else None), generation_wall, model_load or None


def load_setup_times(setup_times_dir: Path) -> dict[str, float]:
    times: dict[str, float] = {}
    if not setup_times_dir.is_dir():
        return times
    for file in sorted(setup_times_dir.glob("*.json")):
        data = read_json(file)
        framework = data.get("framework") or file.stem
        total = data.get("total_setup_seconds")
        if total is None:
            phases = data.get("phases") or {}
            values = [float(value) for value in phases.values() if value is not None]
            total = sum(values)
        if total is not None:
            times[framework] = float(total)
    return times


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_rows = jsonl_rows(Path(args.expected_input_jsonl))
    expected_ids = [row["id"] for row in expected_rows]
    expected_by_id = {row["id"]: row for row in expected_rows}
    setup_times = load_setup_times(Path(args.setup_times_dir))
    raw_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for run_dir in sorted(Path(args.results_root).glob("*_rep*")):
        if not run_dir.is_dir():
            continue
        framework = run_dir.name.split("_rep", 1)[0]
        rep = int(run_dir.name.rsplit("rep", 1)[1])
        rows = find_output_rows(run_dir)
        ids = [row.get("id") for row in rows]
        valid_rows = sum(1 for row in rows if validate_row(row))
        query_normalization_consistent = 0
        answer_normalization_consistent = 0
        for row in rows:
            expected_row = expected_by_id.get(row.get("id"))
            if expected_row is None or not validate_row(row):
                continue
            query_ok, answer_ok = normalization_checks(row, expected_row)
            query_normalization_consistent += int(query_ok)
            answer_normalization_consistent += int(answer_ok)
        missing = len(set(expected_ids) - set(ids))
        duplicates = len(ids) - len(set(ids))
        order_correct = ids == expected_ids[: len(ids)]
        launcher = read_json(run_dir / "launcher_summary.json")
        summary = read_json(run_dir / "run_summary.json")
        tokens, generation_wall, model_load = extract_anonlib_tokens(run_dir)
        if framework == "nemo":
            tokens = None
            generation_wall = None
            model_load = None
        wall = summary.get("full_end_to_end_wall_seconds") or launcher.get("launcher_wall_seconds")
        gpu = (launcher.get("gpu_utilization") or {}).get("mean")
        raw_rows.append(
            {
                "framework": framework,
                "rep": rep,
                "total_rows": len(expected_rows),
                "successfully_materialized_rows": len(rows),
                "missing_rows": missing,
                "duplicate_ids": duplicates,
                "schema_valid_rows": valid_rows,
                "query_normalization_consistent_rows": query_normalization_consistent,
                "answer_normalization_consistent_rows": answer_normalization_consistent,
                "row_order_correct": order_correct,
                "total_generated_tokens": tokens,
                "generation_wall_seconds": generation_wall,
                "full_end_to_end_wall_seconds": wall,
                "model_server_startup_seconds": model_load,
                "tok_s_gpu": (tokens / generation_wall) if tokens and generation_wall else None,
                "rows_s": (len(rows) / wall) if wall else None,
                "mean_gpu_utilization": gpu,
                "setup_seconds": setup_times.get(framework),
                "returncode": launcher.get("returncode", summary.get("returncode")),
            }
        )
        validation_rows.append(
            {
                "framework": framework,
                "rep": rep,
                "schema_valid_rows": valid_rows,
                "query_normalization_consistent_rows": query_normalization_consistent,
                "answer_normalization_consistent_rows": answer_normalization_consistent,
                "invalid_rows": len(rows) - valid_rows,
                "missing_rows": missing,
                "duplicate_ids": duplicates,
                "row_order_correct": order_correct,
            }
        )
    return raw_rows, validation_rows


def latex_table(summary_rows: list[dict[str, Any]], footprint_rows: list[dict[str, Any]]) -> str:
    footprint = {row["framework"]: row for row in footprint_rows}
    names = {"anonlib": "AnonLib", "nemo": "NeMo Curator + Data Designer"}
    latex_break = " " + ("\\" * 2)
    lines = [
        "Framework & Valid outputs & tok/s/GPU & End-to-end time & Setup time & Declarative LOC & Glue Python LOC" + latex_break,
        "\\hline",
    ]
    for row in summary_rows:
        fw = row["framework"]
        valid = row.get("schema_valid_rows_mean")
        toks = row.get("tok_s_gpu_mean")
        wall = row.get("full_end_to_end_wall_seconds_mean")
        setup = row.get("setup_seconds_mean")
        tok_text = f"{toks:.2f}" if toks is not None else "N/A"
        wall_text = f"{wall:.2f} s" if wall is not None else "N/A"
        setup_text = f"{setup:.2f} s" if setup is not None else "N/A"
        valid_text = f"{valid:.0f}" if valid is not None else "N/A"
        lines.append(
            f"{names.get(fw, fw)} & {valid_text} & {tok_text} & {wall_text} & {setup_text} & {footprint[fw]['declarative_loc']} & {footprint[fw]['glue_python_loc']}" + latex_break
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, validation_rows = collect(args)
    summary_rows = summarize(raw_rows)
    footprint_rows = implementation_footprint()
    setup_times = load_setup_times(Path(args.setup_times_dir))
    write_csv(out_dir / "raw_results.csv", raw_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "implementation_footprint.csv", footprint_rows)
    write_csv(out_dir / "output_validation.csv", validation_rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary_rows, "footprint": footprint_rows, "setup_times": setup_times}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "latex_table.tex").write_text(latex_table(summary_rows, footprint_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "runs": len(raw_rows)}, indent=2))


if __name__ == "__main__":
    main()
