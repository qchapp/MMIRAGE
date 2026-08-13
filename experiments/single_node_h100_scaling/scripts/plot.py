#!/usr/bin/env python3
"""Plot aggregate single-node throughput and parallel efficiency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default="summary.csv")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def as_float(value: str):
    if value in ("", "None", None):
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows = [row for row in rows if row.get("gpu_count")]
    if not rows:
        return

    gpu_counts = [int(row["gpu_count"]) for row in rows]
    throughput = [as_float(row["aggregate_output_tok_s_mean"]) for row in rows]
    throughput_std = [as_float(row["aggregate_output_tok_s_std"]) or 0.0 for row in rows]
    efficiency = [as_float(row["parallel_efficiency"]) for row in rows]

    plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(gpu_counts, throughput, yerr=throughput_std, marker="o", capsize=4)
    plt.xticks(gpu_counts)
    plt.xlabel("H100 GPUs on one node")
    plt.ylabel("Aggregate output tok/s")
    plt.title("ANONLIB Single-Node Strong Scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_throughput_vs_gpu.png", dpi=200)
    plt.close()

    if any(value is not None for value in efficiency):
        plt.figure(figsize=(5.2, 3.4))
        plt.plot(gpu_counts, efficiency, marker="o")
        plt.xticks(gpu_counts)
        plt.ylim(0, 1.05)
        plt.xlabel("H100 GPUs on one node")
        plt.ylabel("Parallel efficiency")
        plt.title("ANONLIB Single-Node Efficiency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "parallel_efficiency_vs_gpu.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
