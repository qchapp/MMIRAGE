"""Custom Data Designer columns for the Curator/Data Designer comparison."""

from __future__ import annotations

import json
import re
from typing import Any

import data_designer.config as dd


_SPACE_RE = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value)).strip()


@dd.custom_column_generator(required_columns=["query"])
def normalize_query(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["normalized_query"] = _normalize_text(row.get("query", ""))
    return row


@dd.custom_column_generator(required_columns=["vlm_result"])
def normalize_generated_answer(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    generated = row.get("vlm_result", {})
    if isinstance(generated, str):
        try:
            generated = json.loads(generated)
        except json.JSONDecodeError:
            generated = {"answer": generated, "rationale": ""}
    if not isinstance(generated, dict):
        generated = {"answer": str(generated), "rationale": ""}
    row["generated_answer_normalized"] = _normalize_text(generated.get("answer", "")).lower()
    return row
