#!/usr/bin/env python3
"""Download MedTrinity-25M demo and save it as a local Hugging Face dataset.

The output directory is meant to be consumed by MMIRAGE with:

    type: loadable
    path: <output-dir>

The demo config on Hugging Face is `25M_demo` and contains the `image`, `id`,
and `caption` fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


def _as_dataset(obj: Any, split: str) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise KeyError(f"Split {split!r} not found. Available splits: {list(obj.keys())}")
        return obj[split]
    raise TypeError(f"Unexpected dataset object: {type(obj)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the local save_to_disk dataset will be written.",
    )
    parser.add_argument(
        "--repo-id",
        default="UCSC-VLAA/MedTrinity-25M",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--config-name",
        default="25M_demo",
        help="Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging. Omit for the full demo dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {output_dir}\n"
                "Use --force to overwrite, or choose another --output-dir."
            )
        import shutil

        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.repo_id} / {args.config_name} / split={args.split}")
    ds_obj = load_dataset(
        args.repo_id,
        args.config_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    ds = _as_dataset(ds_obj, args.split)

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided")
        ds = ds.select(range(min(args.max_samples, len(ds))))

    required_columns = {"image", "id", "caption"}
    missing = required_columns - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {sorted(missing)}. "
            f"Available columns: {ds.column_names}"
        )

    print(ds)
    print(ds.features)
    print(f"Saving to: {output_dir}")
    ds.save_to_disk(str(output_dir))

    metadata = {
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "split": args.split,
        "num_rows": len(ds),
        "columns": ds.column_names,
        "output_dir": str(output_dir),
    }
    with (output_dir / "medtrinity_demo_download_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"MEDTRINITY_DEMO={output_dir}")


if __name__ == "__main__":
    main()