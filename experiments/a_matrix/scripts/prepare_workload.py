#!/usr/bin/env python3
"""Prepare one deterministic UltraChat workload for the whole A matrix.

Writes the same stable_id/source_index/prompt_text/prompt_sha256 rows as the
single-node scaling prep (same dataset, seed, selection and normalization), so
every A setup (gpu_scaling, a100_4gpu, recovery) consumes byte-identical model
inputs.

With ``--shared-root`` the recovery-compatible files are also written under
``<shared-root>/data/ultrachat_200k/`` in the same schema (subset.jsonl plus an
id_order.jsonl keyed on stable_id) so the shard-recovery controllers
(MMIRAGE and the native competitor harness) can consume the shared workload.
"""

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
DEFAULT_NUM_ROWS = default_size(EXPERIMENT_DIR, "num_rows", 20_000)


def default_recovery_rows() -> int:
    import yaml

    path = EXPERIMENT_DIR / "configs" / "recovery_size.yaml"
    if not path.exists():
        return 4_000
    with path.open("r", encoding="utf-8") as handle:
        return int((yaml.safe_load(handle) or {}).get("recovery_num_rows", 4_000))


DEFAULT_RECOVERY_ROWS = default_recovery_rows()
DEFAULT_SEED = 20260813


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shared-root", default=None, help="If set, also write recovery-compat files under data/ultrachat_200k/.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--recovery-rows", type=int, default=DEFAULT_RECOVERY_ROWS, help="Rows in the shared-root subset when --shared-root is set (independent of the scaling workload size).")
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--id-field", default="prompt_id")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--max-source-rows", type=int, default=None)
    return parser.parse_args()


def resolved_revision(repo_id: str, repo_type: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return revision


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            digest.update(line)
            handle.write(line)
    return digest.hexdigest()


def normalize_prompt(prompt: str) -> str:
    return "\n".join(line.rstrip() for line in prompt.strip().splitlines())


def extract_prompt(row: Dict[str, Any], prompt_field: str, messages_field: str) -> str:
    value = row.get(prompt_field)
    if isinstance(value, str) and value.strip():
        return value.strip()

    messages = row.get(messages_field)
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

    raise ValueError("Could not extract a non-empty prompt from row")


def main() -> None:
    args = parse_args()
    if args.num_rows < 1:
        raise ValueError("--num-rows must be at least 1")
    if args.recovery_rows < 1:
        raise ValueError("--recovery-rows must be at least 1")
    if args.warmup_rows < 0:
        raise ValueError("--warmup-rows must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_revision = resolved_revision(args.dataset, "dataset", args.dataset_revision)
    model_revision = resolved_revision(args.model_path, "model", args.model_revision)

    ds = load_dataset(args.dataset, split=args.split, revision=dataset_revision)
    if args.seed >= 0:
        ds = ds.shuffle(seed=args.seed)

    target_rows = max(args.num_rows + args.warmup_rows, args.recovery_rows)
    rows = []
    seen_prompt_hashes: set[str] = set()
    duplicate_prompts_skipped = 0
    inspected = 0

    for source_index, row in enumerate(ds):
        if args.max_source_rows is not None and inspected >= args.max_source_rows:
            break
        inspected += 1
        row_dict = dict(row)
        try:
            prompt_text = normalize_prompt(extract_prompt(row_dict, args.prompt_field, args.messages_field))
        except ValueError:
            continue

        prompt_sha = sha256_text(prompt_text)
        if prompt_sha in seen_prompt_hashes:
            duplicate_prompts_skipped += 1
            continue
        seen_prompt_hashes.add(prompt_sha)

        source_id = row_dict.get(args.id_field)
        stable_id = (
            f"{args.dataset}@{dataset_revision or args.dataset_revision or 'unknown'}:"
            f"{args.split}:{source_index}:{prompt_sha[:16]}"
        )
        rows.append(
            {
                "stable_id": stable_id,
                "source_index": source_index,
                "source_id": source_id if source_id is not None else "",
                "prompt_text": prompt_text,
                "prompt_sha256": prompt_sha,
                "dataset": args.dataset,
                "split": args.split,
                "dataset_revision": dataset_revision or args.dataset_revision or "",
            }
        )
        if len(rows) >= target_rows:
            break

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Requested {target_rows} unique rows, but only prepared {len(rows)} "
            f"after inspecting {inspected} source rows."
        )

    workload_rows = rows[: args.num_rows]
    warmup_rows = rows[args.num_rows : target_rows]
    workload_path = output_dir / "workload.jsonl"
    warmup_path = output_dir / "warmup.jsonl"
    workload_checksum = write_jsonl(workload_path, workload_rows)
    warmup_checksum = write_jsonl(warmup_path, warmup_rows)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_resolved": dataset_revision,
        "model_path": args.model_path,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": model_revision,
        "selection_seed": args.seed,
        "selection": "deterministic shuffle followed by first unique prompt hashes",
        "num_rows": args.num_rows,
        "warmup_rows": args.warmup_rows,
        "source_rows_inspected": inspected,
        "duplicate_prompts_skipped": duplicate_prompts_skipped,
        "prompt_field": args.prompt_field,
        "messages_field": args.messages_field,
        "id_field": args.id_field,
        "workload_jsonl": str(workload_path),
        "workload_sha256": workload_checksum,
        "warmup_jsonl": str(warmup_path),
        "warmup_sha256": warmup_checksum,
        "no_prompt_duplication_policy": "Duplicate prompt hashes are skipped; rows are not duplicated to lengthen the benchmark.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    recovery_manifest = None
    if args.shared_root:
        recovery_rows = rows[: args.recovery_rows]
        data_dir = Path(args.shared_root) / "data" / "ultrachat_200k"
        subset_path = data_dir / "subset.jsonl"
        order_path = data_dir / "id_order.jsonl"
        subset_checksum = write_jsonl(subset_path, recovery_rows)
        order_rows = [
            {"order_index": order_index, "stable_id": row["stable_id"], "source_index": row["source_index"]}
            for order_index, row in enumerate(recovery_rows)
        ]
        order_checksum = write_jsonl(order_path, order_rows)
        recovery_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shared_root": str(Path(args.shared_root)),
            "dataset": args.dataset,
            "split": args.split,
            "dataset_revision_resolved": dataset_revision,
            "model_path": args.model_path,
            "model_revision_resolved": model_revision,
            "num_records": args.recovery_rows,
            "num_shards": 16,
            "selection": metadata["selection"],
            "seed": args.seed,
            "subset_jsonl": str(subset_path),
            "subset_sha256": subset_checksum,
            "id_order_jsonl": str(order_path),
            "id_order_sha256": order_checksum,
        }
        (data_dir / "manifest.json").write_text(json.dumps(recovery_manifest, indent=2, sort_keys=True), encoding="utf-8")
        metadata["recovery_manifest"] = str(data_dir / "manifest.json")
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
