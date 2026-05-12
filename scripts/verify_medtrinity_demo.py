#!/usr/bin/env python3
"""Verify a local MedTrinity demo dataset for the MMIRAGE configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image


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
        "--dataset-path",
        required=True,
        help="Path produced by scripts/download_medtrinity_demo.py.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to inspect if load_from_disk returns a DatasetDict.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of examples to decode and inspect.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    ds_obj = load_from_disk(str(dataset_path))
    ds = _as_dataset(ds_obj, args.split)

    print(ds)
    print(ds.features)

    required_columns = {"image", "id", "caption"}
    missing = required_columns - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {ds.column_names}"
        )

    if len(ds) == 0:
        raise ValueError("Dataset is empty.")

    n = min(args.num_examples, len(ds))
    for i in range(n):
        row = ds[i]

        image = row["image"]
        caption = row["caption"]
        sample_id = row["id"]

        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Example {i}: expected PIL.Image.Image in column 'image', got {type(image)!r}"
            )
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Example {i}: caption is empty or not a string")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Example {i}: id is empty or not a string")

        # Force image decoding and basic validation.
        image.load()

        print(
            f"Example {i}: id={sample_id!r}, "
            f"image={image.width}x{image.height}, caption_chars={len(caption)}"
        )

    print("Verification passed.")
    print(f"Dataset path for MMIRAGE: {dataset_path}")


if __name__ == "__main__":
    main()