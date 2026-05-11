"""Shared helpers for batch metadata receipt files."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


def _normalize_metadata_paths(metadata_paths: str | Sequence[str]) -> List[str]:
    """Return metadata paths as a concrete list."""
    if isinstance(metadata_paths, str):
        return [metadata_paths]
    return list(metadata_paths)


def _read_metadata_records(metadata_output_paths: str | Sequence[str]) -> List[Dict[str, Any]]:
    """Load valid JSON objects from one or more receipt files.

    Malformed lines are skipped with a warning so partially written or noisy
    receipt files do not stop collection. Only JSON objects are retained
    because downstream resolution depends on keyed metadata.
    """
    records: List[Dict[str, Any]] = []
    for metadata_output_path in _normalize_metadata_paths(metadata_output_paths):
        with open(metadata_output_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed metadata JSON line in %s: %s",
                        metadata_output_path,
                        exc,
                    )
                    continue
                #if isinstance(parsed, dict):
                records.append(parsed)
    return records