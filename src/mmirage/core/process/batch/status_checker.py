"""Receiver-side helper to check provider batch status from metadata receipts.

Designed for CLI use against JSONL receipt files. Skips malformed lines and
missing keys to keep status checks resilient to partial metadata corruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Mapping, Sequence, TextIO, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
from mmirage.core.process.batch.provider_resolution import resolve_provider_configs
from mmirage.core.process.batch.registry import BatchAdapterFactory


def _normalize_metadata_paths(metadata_paths: str | Sequence[str]) -> List[str]:
    if isinstance(metadata_paths, str):
        return [metadata_paths]
    return [str(path) for path in metadata_paths]


def _read_metadata_records(metadata_output_paths: str | Sequence[str]) -> List[Dict[str, str]]:
    """Load JSONL metadata records from one or more files.

    Lines that are empty or invalid JSON are ignored to allow best-effort
    status checks when receipt files are partially corrupted.
    """
    records: List[Dict[str, str]] = []
    for metadata_output_path in _normalize_metadata_paths(metadata_output_paths):
        with open(metadata_output_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    return records


def extract_unique_provider_batches(metadata_records: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """Return unique ``(provider, provider_batch_id)`` pairs.

    Normalizes provider names to lowercase and ignores records that do not
    provide both keys, preventing accidental calls with incomplete metadata.
    """
    unique_pairs: List[Tuple[str, str]] = []
    seen = set()

    for record in metadata_records:
        provider = str(record.get("provider", "")).strip().lower()
        provider_batch_id = str(record.get("provider_batch_id", "")).strip()

        if not provider or not provider_batch_id:
            continue

        pair = (provider, provider_batch_id)
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    return unique_pairs


def run_status_checker(
    metadata_records: Sequence[Mapping[str, Any]],
    provider_configs: Mapping[str, BatchProviderConfig],
    output: TextIO = sys.stdout,
) -> List[BatchSubmissionResult]:
    """Check batch status for each referenced provider batch.

    Prints a per-batch line and a per-provider summary. Providers missing
    from ``provider_configs`` are skipped rather than failing the run so
    partial configurations still yield useful status output.
    """
    results: List[BatchSubmissionResult] = []
    counter: Dict[str, Dict[str, int]] = {}

    for provider, provider_batch_id in extract_unique_provider_batches(metadata_records):
        if provider not in provider_configs:
            print(f"Skipping batch {provider_batch_id}: no config for provider '{provider}'.", file=output)
            provider_counts = counter.setdefault(provider, {})
            provider_counts["skipped"] = provider_counts.get("skipped", 0) + 1
            continue

        config = provider_configs[provider]
        adapter = BatchAdapterFactory.from_config(config)
        result = adapter.check_batch_status(provider_batch_id=provider_batch_id, config=config)
        results.append(result)

        print(f"Batch {provider_batch_id} ({provider}): {result.status}", file=output)
        provider_counts = counter.setdefault(provider, {})
        provider_counts[result.status] = provider_counts.get(result.status, 0) + 1

    print("\n------------ Batch status summary ------------", file=output)
    for provider, status_counts in counter.items():
        print(f"Total batches for provider '{provider}':", file=output)
        total = sum(status_counts.values())
        for status, count in status_counts.items():
            print(f"  {status}: {count}/{total}", file=output)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the status-check entry point."""
    parser = argparse.ArgumentParser(description="Check provider batch statuses from metadata receipts.")
    parser.add_argument(
        "--metadata-path",
        nargs="+",
        required=True,
        help="Path(s) to metadata JSONL receipt file(s). Supports multiple files.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that returns a process-style status code.

    Returns 0 on success or no batches found, and 1 on configuration or
    provider resolution failures.
    """
    args = _build_arg_parser().parse_args(argv)
    from mmirage.config.utils import load_mmirage_config

    records = _read_metadata_records(args.metadata_path)
    pairs = extract_unique_provider_batches(records)
    if not pairs:
        print(f"No provider batch IDs found in metadata file: {args.metadata_path}")
        return 0

    try:
        cfg = load_mmirage_config(args.config)
        provider_configs = resolve_provider_configs(records, cfg)
        if not provider_configs:
            print("No supported provider configurations could be built from metadata.")
            return 1
        run_status_checker(
            metadata_records=records,
            provider_configs=provider_configs,
        )
    except Exception as exc:
        print(f"Status checker failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
