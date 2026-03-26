"""Receiver-side utility for collecting provider results and merging by source row index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.registry import BatchAdapterFactory


def _read_metadata_records(metadata_output_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(metadata_output_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _aggregate_batch_mappings(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    aggregated: Dict[Tuple[str, str], Dict[str, int]] = {}

    for record in records:
        provider = str(record.get("provider", "")).strip().lower()
        provider_batch_id = str(record.get("provider_batch_id", "")).strip()
        mapping = record.get("custom_id_to_source_index", {})

        if not provider or not provider_batch_id or not isinstance(mapping, dict):
            continue

        key = (provider, provider_batch_id)
        if key not in aggregated:
            aggregated[key] = {}

        for custom_id, source_index in mapping.items():
            try:
                aggregated[key][str(custom_id)] = int(source_index)
            except (TypeError, ValueError):
                continue

    return aggregated


def collect_and_merge(
    metadata_output_path: str,
    provider_configs: Mapping[str, BatchProviderConfig],
    output_path: str,
) -> List[Dict[str, Any]]:
    """Collect completed results and reconstruct rows in source index order."""
    records = _read_metadata_records(metadata_output_path)
    pair_to_mapping = _aggregate_batch_mappings(records)

    adapters: Dict[str, Any] = {}
    pair_to_results: Dict[Tuple[str, str], Sequence[Dict[str, Any]]] = {}

    for provider, provider_batch_id in pair_to_mapping.keys():
        if provider not in provider_configs:
            raise ValueError(f"No provider config found for '{provider}'.")

        if provider not in adapters:
            adapters[provider] = BatchAdapterFactory.from_config(provider_configs[provider])

        pair = (provider, provider_batch_id)
        pair_to_results[pair] = adapters[provider].retrieve_results(
            provider_batch_id=provider_batch_id,
            config=provider_configs[provider],
        )

    indexed_rows: MutableMapping[str, Dict[str, Any]] = {}
    for pair, mapping in pair_to_mapping.items():
        results = pair_to_results.get(pair, [])
        for result_row in results:
            custom_id = str(result_row.get("custom_id", "")).strip()
            if not custom_id or custom_id not in mapping:
                continue
            row_payload = _build_output_payload(result_row)
            indexed_rows[custom_id] = {
                "source_index": 0,
                "custom_id": custom_id,
                **row_payload,
            }

    ordered_rows = list(indexed_rows.values())
    for idx, row in enumerate(ordered_rows):
        row["source_index"] = idx

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ordered_rows


def _build_output_payload(result_row: Mapping[str, Any]) -> Dict[str, Any]:
    raw_content = _extract_content_string(result_row)
    if not raw_content:
        return {"caption": ""}

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return {"caption": raw_content}

    if isinstance(parsed, dict) and ("question" in parsed or "answer" in parsed):
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": str(parsed.get("question", "")),
                },
                {
                    "role": "assistant",
                    "content": str(parsed.get("answer", "")),
                },
            ]
        }

    return {"caption": raw_content}


def _extract_content_string(result_row: Mapping[str, Any]) -> str:
    # Preferred OpenAI envelope path for Structured Outputs / plain responses.
    response = result_row.get("response")
    if isinstance(response, Mapping):
        body = response.get("body")
        if isinstance(body, Mapping):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, Mapping):
                    message = first_choice.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content

    # Fallback for normalized adapter payloads carrying generated_text directly.
    generated_text = result_row.get("generated_text")
    if isinstance(generated_text, str):
        return generated_text

    return ""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect provider batch outputs and merge rows by source index."
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to metadata JSONL receipt file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write merged JSONL output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for collector execution.")

    provider_configs: Dict[str, BatchProviderConfig] = {
        "openai": OpenAIBatchConfig(credentials={"api_key": api_key})
    }

    rows = collect_and_merge(args.metadata_path, provider_configs, args.output_path)
    print(f"Merged {len(rows)} rows and saved to {args.output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Collector failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
