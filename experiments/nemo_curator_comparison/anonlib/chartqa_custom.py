"""AnonLib custom deterministic processing for the ChartQA comparison."""

from __future__ import annotations

import json
import re
from typing import Any

_SPACE_RE = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value)).strip()


def normalize_query(row: dict[str, Any]) -> dict[str, str]:
    """Normalize the source query while preserving row order."""
    normalized_query = _normalize_text(row.get("query", ""))
    return {"normalized_query": normalized_query}


def normalize_generated_answer(row: dict[str, Any]) -> dict[str, str]:
    """Normalize the structured generated answer after the VLM column exists."""
    generated = row.get("vlm_result", {})
    if isinstance(generated, str):
        try:
            generated = json.loads(generated)
        except json.JSONDecodeError:
            generated = {"answer": generated, "rationale": ""}
    if not isinstance(generated, dict):
        generated = {"answer": str(generated), "rationale": ""}
    answer = _normalize_text(generated.get("answer", "")).lower()
    return {"generated_answer_normalized": answer}
