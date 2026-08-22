#!/usr/bin/env python3
"""Prepare shared simplescaling/s1K-1.1 prompts for overhead benchmarking."""

from __future__ import annotations

import argparse
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
from transformers import AutoTokenizer  # noqa: E402

DEFAULT_DATASET = "simplescaling/s1K-1.1"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_TEXT_TEMPLATE = (
    "<|im_start|>user\n{question}\n<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--num-rows", type=int, default=default_size(EXPERIMENT_DIR, "num_rows", 1000))
    parser.add_argument("--warmup-rows", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def resolved_revision(
    repo_id: str, repo_type: str, revision: Optional[str]
) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision
        )
        return getattr(info, "sha", None)
    except Exception:
        return revision


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_rows < 1:
        raise ValueError("--num-rows must be at least 1")
    if args.warmup_rows < 0:
        raise ValueError("--warmup-rows must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_revision = resolved_revision(args.dataset, "dataset", args.dataset_revision)
    model_revision = resolved_revision(args.model_path, "model", args.model_revision)

    total_rows = args.num_rows + args.warmup_rows
    ds = load_dataset(
        args.dataset,
        split=f"{args.split}[{args.start_index}:{args.start_index + total_rows}]",
        revision=dataset_revision,
    )
    if len(ds) < args.num_rows:
        raise RuntimeError(
            f"Expected at least {args.num_rows} measured rows, loaded {len(ds)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=model_revision,
        trust_remote_code=True,
    )

    rows = []
    for offset, row in enumerate(ds):
        source_index = args.start_index + offset
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Row {source_index} does not contain a string question")
        prompt_text = DEFAULT_PROMPT_TEXT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "source_index": source_index,
                "question": question,
                "prompt_text": prompt_text,
                "prompt": prompt,
            }
        )

    measured_rows = rows[: args.num_rows]
    warmup_rows = rows[args.num_rows : args.num_rows + args.warmup_rows]
    warmup_reused_from_measured = False
    if args.warmup_rows and len(warmup_rows) < args.warmup_rows:
        warmup_rows = measured_rows[: args.warmup_rows]
        warmup_reused_from_measured = True
    write_jsonl(output_dir / "warmup_prompts.jsonl", warmup_rows)
    write_jsonl(output_dir / "prompts.jsonl", measured_rows)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_resolved": dataset_revision,
        "model_path": args.model_path,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": model_revision,
        "start_index": args.start_index,
        "num_rows": args.num_rows,
        "warmup_rows": args.warmup_rows,
        "warmup_rows_reused_from_measured": warmup_reused_from_measured,
        "prompt_text_template": DEFAULT_PROMPT_TEXT_TEMPLATE,
        "prompt_construction": "MMIRAGE LLMProcessor behavior: Jinja prompt_text, then tokenizer.apply_chat_template(..., add_generation_prompt=True). Raw client uses the resulting prompt field.",
        "measured_prompts_jsonl": str(output_dir / "prompts.jsonl"),
        "warmup_prompts_jsonl": str(output_dir / "warmup_prompts.jsonl"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
