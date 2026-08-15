#!/usr/bin/env python3
"""Prepare the public text workload for the MMIRAGE recovery experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402
from experiments._shared.sizes import default_size  # noqa: E402

DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NUM_RECORDS = default_size(EXPERIMENT_DIR, "num_records", 40_000)
DEFAULT_SEED = 20260813
DEFAULT_CONTAINER_ROOT = "/workspace/mmirage-recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"Shared experiment root inside the container. Defaults to $MMIRAGE_RECOVERY_ROOT or {DEFAULT_CONTAINER_ROOT}.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--max-source-chars", type=int, default=4096)
    return parser.parse_args()


def default_output_root() -> Path:
    import os

    env_root = os.environ.get("MMIRAGE_RECOVERY_ROOT")
    if env_root:
        return Path(env_root)
    return Path(DEFAULT_CONTAINER_ROOT)


def require_container_terminal(output_root: Path) -> None:
    if not output_root.is_absolute():
        raise RuntimeError(
            f"Use an absolute in-container --output-root path, got {output_root}. "
            f"Recommended: {DEFAULT_CONTAINER_ROOT}"
        )


def resolved_revision(repo_id: str, repo_type: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return revision


def repo_license(repo_id: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
        card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
        if isinstance(card_data, dict):
            value = card_data.get("license")
            if isinstance(value, list):
                return ",".join(str(item) for item in value)
            return str(value) if value is not None else None
    except Exception:
        return None
    return None


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            digest.update(line)
            handle.write(line)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prompt(row: Dict[str, Any], source_index: int) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                return content.strip()
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content:
                    return content

    for key in ("prompt", "question", "text", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Row {source_index} is incompatible with this workload: expected messages, prompt, question, text, or instruction."
    )


def build_prompt(source_prompt: str, max_source_chars: int) -> str:
    clipped = source_prompt[:max_source_chars]
    return (
        "Answer the public user request below in exactly three numbered paragraphs.\n"
        "Each paragraph should contain three to four complete sentences.\n\n"
        f"User request:\n{clipped}"
    )


def main() -> None:
    args = parse_args()
    if args.num_records < 16:
        raise ValueError("--num-records must be at least 16 for the fixed 16-shard workload")
    if args.max_source_chars < 1:
        raise ValueError("--max-source-chars must be positive")

    output_root = Path(args.output_root) if args.output_root else default_output_root()
    require_container_terminal(output_root)
    data_dir = output_root / "data" / "ultrachat_200k"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_revision = resolved_revision(args.dataset, "dataset", args.revision)
    model_revision = resolved_revision(args.model_path, "model", args.model_revision)

    ds = load_dataset(args.dataset, split=args.split, revision=args.revision)
    ds = ds.add_column("__source_index", list(range(len(ds))))
    selected = ds.shuffle(seed=args.seed).select(range(args.num_records))

    rows = []
    order_rows = []
    for order_index, row in enumerate(selected):
        source_index = int(row["__source_index"])
        source_prompt = extract_prompt(dict(row), source_index)
        mmirage_id = f"{args.dataset}:{args.split}:{dataset_revision or args.revision or 'unresolved'}:{source_index}"
        record = {
            "mmirage_id": mmirage_id,
            "order_index": order_index,
            "source_index": source_index,
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_revision": dataset_revision or args.revision or "unresolved",
            "source_prompt": source_prompt,
            "prompt_text": build_prompt(source_prompt, args.max_source_chars),
        }
        rows.append(record)
        order_rows.append(
            {
                "order_index": order_index,
                "mmirage_id": mmirage_id,
                "source_index": source_index,
            }
        )

    subset_path = data_dir / "subset.jsonl"
    order_path = data_dir / "id_order.jsonl"
    subset_checksum = write_jsonl(subset_path, rows)
    order_checksum = write_jsonl(order_path, order_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "dataset_revision_requested": args.revision,
        "dataset_revision_resolved": dataset_revision,
        "dataset_license": repo_license(args.dataset, args.revision),
        "model_path": args.model_path,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": model_revision,
        "num_records": args.num_records,
        "num_shards": 16,
        "selection": "datasets.Dataset.shuffle(seed).select(range(num_records)) after adding __source_index",
        "seed": args.seed,
        "max_source_chars": args.max_source_chars,
        "subset_jsonl": str(subset_path),
        "subset_sha256": subset_checksum,
        "id_order_jsonl": str(order_path),
        "id_order_sha256": order_checksum,
        "fallback_dataset_policy": "If HuggingFaceH4/ultrachat_200k is unavailable or its schema no longer exposes text/messages, choose another public text dataset and record the reason in this manifest before running MMIRAGE.",
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
