#!/usr/bin/env python3
"""Prepare a deterministic cnn_dailymail subset for the text_shortening task.

Rows carry the shared native contract fields (``stable_id``, ``source_index``,
``prompt_sha256``, ``prompt_text``) so the single-node scaling runners and the
framework-native shard workers consume them unchanged. ``prompt_text`` asks the
model to summarize the article in 2 to 3 sentences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402
from experiments._shared.sizes import default_size  # noqa: E402

DEFAULT_DATASET = "cnn_dailymail"
DEFAULT_CONFIG = "3.0.0"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NUM_ROWS = default_size(EXPERIMENT_DIR, "num_rows", 40_000)
DEFAULT_SEED = 20260813

SUMMARY_TEMPLATE = (
    "Summarize the following news article in 2 to 3 sentences.\n"
    "Keep the summary faithful to the facts in the article.\n\n"
    "Article:\n{article}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-source-chars", type=int, default=4096)
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def main() -> None:
    args = parse_args()
    if args.num_rows < 1:
        raise ValueError("--num-rows must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_revision = resolved_revision(args.dataset, "dataset", args.dataset_revision)
    model_revision = resolved_revision(args.model_path, "model", args.model_revision)

    ds = load_dataset(
        args.dataset,
        args.config_name,
        split=args.split,
        revision=args.dataset_revision,
    )
    if args.seed >= 0:
        ds = ds.shuffle(seed=args.seed)

    target_rows = args.num_rows + args.warmup_rows
    rows = []
    seen_hashes: set[str] = set()
    for source_index, row in enumerate(ds):
        row_dict = dict(row)
        article = row_dict.get("article")
        if not isinstance(article, str) or not article.strip():
            continue
        article = normalize_text(article[: args.max_source_chars])
        prompt_text = SUMMARY_TEMPLATE.format(article=article)
        prompt_sha = sha256_text(prompt_text)
        if prompt_sha in seen_hashes:
            continue
        seen_hashes.add(prompt_sha)
        source_id = row_dict.get("id", "")
        stable_id = (
            f"{args.dataset}@{args.config_name}@{dataset_revision or args.dataset_revision or 'unknown'}:"
            f"{args.split}:{source_index}:{prompt_sha[:16]}"
        )
        rows.append(
            {
                "stable_id": stable_id,
                "source_index": source_index,
                "source_id": source_id if isinstance(source_id, str) else str(source_id),
                "prompt_text": prompt_text,
                "prompt_sha256": prompt_sha,
                "dataset": args.dataset,
                "config_name": args.config_name,
                "split": args.split,
                "dataset_revision": dataset_revision or args.dataset_revision or "",
            }
        )
        if len(rows) >= target_rows:
            break

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Requested {target_rows} rows, but only prepared {len(rows)} from {len(ds)} source rows."
        )

    workload_rows = rows[: args.num_rows]
    warmup_rows = rows[args.num_rows : target_rows]
    workload_path = output_dir / "workload.jsonl"
    warmup_path = output_dir / "warmup.jsonl"
    write_jsonl(workload_path, workload_rows)
    write_jsonl(warmup_path, warmup_rows)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config_name": args.config_name,
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
        "max_source_chars": args.max_source_chars,
        "prompt_template": SUMMARY_TEMPLATE,
        "workload_jsonl": str(workload_path),
        "workload_sha256": sha256_file(workload_path),
        "warmup_jsonl": str(warmup_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
