#!/usr/bin/env python3
"""Prepare a pinned deterministic ChartQA subset for the comparison."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ID = "HuggingFaceM4/ChartQA"
DEFAULT_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/nemo_curator_comparison/workload/chartqa")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--image-format", choices=["path", "base64"], default="base64")
    return parser.parse_args()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value)


def image_to_png_bytes(image: Any) -> bytes:
    if isinstance(image, dict) and image.get("bytes") is not None:
        return bytes(image["bytes"])
    if isinstance(image, dict) and image.get("path"):
        return Path(image["path"]).read_bytes()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def get_first_existing(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of the expected fields exist: {names}; row keys={sorted(row)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_id, split=args.split, revision=args.revision)
    if args.num_rows > len(dataset):
        raise ValueError(f"Requested {args.num_rows} rows but split has only {len(dataset)} rows")

    indices = list(range(len(dataset)))
    import random

    rng = random.Random(args.seed)
    rng.shuffle(indices)
    selected_indices = sorted(indices[: args.num_rows])

    records_path = out_dir / "chartqa_subset.jsonl"
    checksum = hashlib.sha256()
    image_checksums: dict[str, str] = {}

    with records_path.open("w", encoding="utf-8") as handle:
        for subset_index, source_index in enumerate(selected_indices):
            raw = dataset[int(source_index)]
            image = get_first_existing(raw, ["image", "img"])
            question = get_first_existing(raw, ["query", "question"])
            answer = get_first_existing(raw, ["label", "answer", "answers"])
            sample_id = f"chartqa-{args.split}-{source_index:08d}"

            image_bytes = image_to_png_bytes(image)
            image_sha256 = sha256_bytes(image_bytes)
            image_name = f"{sample_id}.png"
            image_path = images_dir / image_name
            image_path.write_bytes(image_bytes)
            image_checksums[image_name] = image_sha256

            if args.image_format == "base64":
                image_value = base64.b64encode(image_bytes).decode("ascii")
            else:
                image_value = str(Path("images") / image_name)

            record = {
                "id": sample_id,
                "source_index": int(source_index),
                "subset_index": subset_index,
                "image": image_value,
                "image_path": str(Path("images") / image_name),
                "image_sha256": image_sha256,
                "query": str(question),
                "reference_answer": normalize_answer(answer),
                "source": "ChartQA",
            }
            line = stable_json(record).encode("utf-8") + b"\n"
            checksum.update(line)
            handle.write(line.decode("utf-8"))

    manifest = {
        "dataset_id": args.dataset_id,
        "dataset_revision": args.revision,
        "split": args.split,
        "num_rows": args.num_rows,
        "subset_seed": args.seed,
        "selected_indices": selected_indices,
        "records_file": str(records_path),
        "records_sha256": checksum.hexdigest(),
        "image_format": args.image_format,
        "image_checksums": image_checksums,
        "license": {
            "dataset_card": f"https://huggingface.co/datasets/{args.dataset_id}",
            "note": "Use the dataset card and original ChartQA terms as authoritative. This script records the source revision and does not relicense the data.",
        },
        "environment": {
            "python": os.sys.version,
            "cwd": str(Path.cwd()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": str(records_path), "manifest": str(out_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
