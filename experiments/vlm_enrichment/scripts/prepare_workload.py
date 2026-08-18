#!/usr/bin/env python3
"""Prepare a deterministic MedTrinity-25M subset for the vlm_enrichment task.

Loads ``UCSC-VLAA/MedTrinity-25M`` (config ``25M_demo``, split ``train``),
saves each image to ``images/`` and writes a JSONL with one row per sample:

- ``id``: dataset sample id
- ``caption``: original caption
- ``image_path``: relative path under the image base directory

Set ``HF_TOKEN`` in the environment for the demo config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402
from experiments._shared.sizes import default_size  # noqa: E402

DEFAULT_REPO_ID = "UCSC-VLAA/MedTrinity-25M"
DEFAULT_CONFIG_NAME = "25M_demo"
DEFAULT_SPLIT = "train"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NUM_ROWS = default_size(EXPERIMENT_DIR, "num_rows", 400)
DEFAULT_SEED = 20260813


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def save_image(image: Any, path: Path) -> None:
    image = image.convert("RGB")
    image.save(path, format="PNG")


def main() -> None:
    args = parse_args()
    if args.num_rows < 1:
        raise ValueError("--num-rows must be at least 1")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset_revision = resolved_revision(args.repo_id, "dataset", args.dataset_revision)

    ds = load_dataset(
        args.repo_id,
        args.config_name,
        split=args.split,
        revision=dataset_revision,
    )
    if args.seed >= 0:
        ds = ds.shuffle(seed=args.seed)

    rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_index, row in enumerate(ds):
        row_dict = dict(row)
        image = row_dict.get("image")
        caption = row_dict.get("caption")
        if image is None:
            continue
        if not isinstance(caption, str):
            caption = str(caption) if caption is not None else ""
        sample_id = str(row_dict.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)

        image_sha = sha256_text(f"{sample_id}:{source_index}:{caption[:256]}")
        image_name = f"{sample_id[:24]}_{image_sha[:16]}.png"
        save_image(image, images_dir / image_name)
        rows.append(
            {
                "id": sample_id,
                "source_index": source_index,
                "caption": caption,
                "image_path": f"images/{image_name}",
            }
        )
        if len(rows) >= args.num_rows:
            break

    if len(rows) < args.num_rows:
        raise RuntimeError(
            f"Requested {args.num_rows} rows, but only prepared {len(rows)} from {len(ds)} source rows."
        )

    rows_path = output_dir / "rows.jsonl"
    write_jsonl(rows_path, rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "split": args.split,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_resolved": dataset_revision,
        "selection_seed": args.seed,
        "selection": "deterministic shuffle followed by first unique sample ids",
        "num_rows": args.num_rows,
        "rows_jsonl": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "image_count": len(list(images_dir.glob("*.png"))),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
