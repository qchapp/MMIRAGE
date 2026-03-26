"""Receiver-side utility for polling provider batch statuses from metadata receipts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence, TextIO, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
from mmirage.core.process.batch.registry import BatchAdapterFactory


def extract_unique_provider_batches(metadata_output_path: str) -> List[Tuple[str, str]]:
    """Parse metadata JSONL and return unique ``(provider, provider_batch_id)`` pairs.

    Malformed lines and records missing required keys are skipped safely.
    """
    unique_pairs: List[Tuple[str, str]] = []
    seen = set()

    with open(metadata_output_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

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
    metadata_output_path: str,
    provider_configs: Mapping[str, BatchProviderConfig],
    output: TextIO = sys.stdout,
) -> List[BatchSubmissionResult]:
    """Check and print statuses for batches referenced in a metadata receipt file."""
    results: List[BatchSubmissionResult] = []
    counter: Dict[str, Dict[str, int]] = {}

    for provider, provider_batch_id in extract_unique_provider_batches(metadata_output_path):
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


def _build_provider_configs_from_metadata(
    metadata_output_path: str,
) -> Dict[str, BatchProviderConfig]:
    provider_names = {provider for provider, _ in extract_unique_provider_batches(metadata_output_path)}
    configs: Dict[str, BatchProviderConfig] = {}

    if "openai" in provider_names:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to check statuses for provider 'openai'."
            )
        configs["openai"] = OpenAIBatchConfig(credentials={"api_key": api_key})

    return configs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check provider batch statuses from metadata receipts.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to metadata JSONL receipt file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pairs = extract_unique_provider_batches(args.metadata_path)
    if not pairs:
        print(f"No provider batch IDs found in metadata file: {args.metadata_path}")
        return 0

    try:
        provider_configs = _build_provider_configs_from_metadata(args.metadata_path)
        if not provider_configs:
            print("No supported provider configurations could be built from metadata.")
            return 1
        run_status_checker(
            metadata_output_path=args.metadata_path,
            provider_configs=provider_configs,
        )
    except Exception as exc:
        print(f"Status checker failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
