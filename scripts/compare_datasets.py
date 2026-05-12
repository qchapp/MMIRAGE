#!/usr/bin/env python3
"""Compare original MedTrinity/PathVQA-style answers with MMIRAGE-processed answers.

This script is schema-flexible. It supports:

1. PathVQA-style original datasets:
       id, question, answer

2. MedTrinity / ShareGPT / OpenAI-style datasets:
       id optional
       conversations

Conversation formats supported:
    OpenAI style:
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "..."}

    ShareGPT style:
        {"from": "human", "value": "..."}
        {"from": "gpt", "value": "..."}

The comparison target is:
    original answer  = --answer-key if present, otherwise auto-detected from
                       caption/answer/text-style columns, otherwise last assistant message
    processed answer = --processed-answer-key if present, otherwise auto-detected from
                       answer/response/output-style columns, otherwise last assistant message
    question         = --question-key if present, otherwise first user/human message,
                       otherwise blank

Recommended for MedTrinity/MMIRAGE demo:
    python scripts/compare_medtrinity_processed.py \
      --original-path /path/to/original_medtrinity_demo \
      --processed-path /path/to/mmirage_processed_demo \
      --alignment order \
      --num-proc 8 \
      --map-batch-size 256 \
      --output-json /tmp/medtrinity_compare_metrics.json \
      --output-csv /tmp/medtrinity_compare_examples.csv

Recommended for PathVQA:
    python scripts/compare_medtrinity_processed.py \
      --original-path ${PATH_VQA} \
      --processed-path ${SCRATCH}/path_vqa_conversations_formatted_local \
      --original-split train \
      --processed-split train \
      --answer-key answer \
      --question-key question \
      --alignment order
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import string
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value, load_from_disk


# Globals used by datasets.map workers.
PROCESSED_DS: Dataset | None = None
GLOBAL_ARGS: dict[str, Any] = {}


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "from",
    "by", "with", "without", "for", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "there", "here",
    "it", "its", "as", "into", "than", "then", "also", "such", "shows",
    "show", "shown", "image", "picture", "figure", "answer", "question",
    "response", "final", "result", "yes", "no", "please", "provide",
    "describe", "description", "following", "medical", "clinical",
}

ASSISTANT_ROLES = {"assistant", "gpt", "model", "bot"}
USER_ROLES = {"user", "human", "patient", "question"}
ROLE_KEYS = ("role", "from", "speaker")
CONTENT_KEYS = ("content", "value", "text")


def as_dataset(obj: Any, split: str) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise KeyError(
                f"Split {split!r} not found. Available splits: {list(obj.keys())}"
            )
        return obj[split]
    raise TypeError(f"Unexpected dataset object: {type(obj)!r}")


def none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "na", "n/a"}:
        return None
    return value


def existing_columns(ds: Dataset, requested: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in requested:
        col = none_if_blank(col)
        if col and col in ds.column_names and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def resolve_column(
    ds: Dataset,
    preferred: str | None,
    fallbacks: list[str],
) -> str | None:
    """Return an existing column, preferring `preferred`, then fallbacks.

    This makes demo datasets like ['image', 'id', 'caption'] work without
    requiring explicit --answer-key caption.
    """

    preferred = none_if_blank(preferred)
    if preferred and preferred in ds.column_names:
        return preferred

    for key in fallbacks:
        if key in ds.column_names:
            return key

    return preferred


def parse_possible_json(value: Any) -> Any:
    """Handle datasets where conversations were saved as a JSON string."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    if not (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        return value

    try:
        return json.loads(text)
    except Exception:
        return value


def message_role(msg: Any) -> str | None:
    if not isinstance(msg, dict):
        return None
    for key in ROLE_KEYS:
        value = msg.get(key)
        if value is not None:
            return str(value).strip().lower()
    return None


def message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        for key in CONTENT_KEYS:
            if key in msg and msg[key] is not None:
                return str(msg[key])
        return str(msg)
    return str(msg)


def normalize_conversations(conversations: Any) -> list[Any]:
    conversations = parse_possible_json(conversations)

    if isinstance(conversations, list):
        return conversations

    if isinstance(conversations, dict):
        # Some datasets use {"conversations": [...]} or {"messages": [...]}.
        for key in ("conversations", "messages", "conversation"):
            if isinstance(conversations.get(key), list):
                return conversations[key]
        return [conversations]

    if conversations is None:
        return []

    return [conversations]


def extract_assistant_answer_from_conversations(conversations: Any) -> str:
    messages = normalize_conversations(conversations)

    # Prefer the last explicitly assistant/gpt message.
    for msg in reversed(messages):
        role = message_role(msg)
        if role in ASSISTANT_ROLES:
            return message_content(msg)

    # Fallback: use the last message.
    if messages:
        return message_content(messages[-1])

    return ""


def extract_question_from_conversations(conversations: Any) -> str:
    messages = normalize_conversations(conversations)

    # Prefer the first explicitly user/human message.
    for msg in messages:
        role = message_role(msg)
        if role in USER_ROLES:
            return message_content(msg)

    # Fallback: use the first message.
    if messages:
        return message_content(messages[0])

    return ""


def get_optional_batch_value(batch: dict[str, list[Any]], key: str | None, i: int) -> Any | None:
    key = none_if_blank(key)
    if key and key in batch:
        return batch[key][i]
    return None


def get_row_id(batch: dict[str, list[Any]], key: str | None, i: int, fallback_index: int) -> str:
    value = get_optional_batch_value(batch, key, i)
    if value is None:
        return str(fallback_index)
    return str(value)


def get_answer_from_batch(
    batch: dict[str, list[Any]],
    i: int,
    answer_key: str | None,
    conversations_key: str | None,
) -> str:
    value = get_optional_batch_value(batch, answer_key, i)
    if value is not None:
        return str(value)

    conversations = get_optional_batch_value(batch, conversations_key, i)
    if conversations is not None:
        return extract_assistant_answer_from_conversations(conversations)

    return ""


def get_question_from_batch(
    batch: dict[str, list[Any]],
    i: int,
    question_key: str | None,
    conversations_key: str | None,
) -> str:
    value = get_optional_batch_value(batch, question_key, i)
    if value is not None:
        return str(value)

    conversations = get_optional_batch_value(batch, conversations_key, i)
    if conversations is not None:
        return extract_question_from_conversations(conversations)

    return ""


def strip_markdown(text: str) -> str:
    text = str(text)

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    for marker in ("**", "__", "*", "_", "`"):
        text = text.replace(marker, " ")

    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)

    return text


def remove_answer_labels(text: str) -> str:
    text = str(text)

    text = re.sub(
        r"(?im)^\s*(answer|final answer|response|ground-truth answer|ground truth answer)\s*[:\-]\s*",
        "",
        text,
    )

    text = re.sub(
        r"(?im)^\s*(answer|final answer|response|explanation|question)\s*$",
        "",
        text,
    )

    return text


def normalize_for_text_metric(text: str) -> str:
    text = strip_markdown(text)
    text = remove_answer_labels(text)
    text = text.lower()

    table = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(table)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_for_text_metric(text)
    if not normalized:
        return []
    return normalized.split()


def exact_match(reference: str, prediction: str) -> bool:
    return normalize_for_text_metric(reference) == normalize_for_text_metric(prediction)


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}

    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    overlap = 0
    for tok, count in ref_counts.items():
        overlap += min(count, pred_counts.get(tok, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, pred_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def yes_no_label(text: str) -> str | None:
    normalized = normalize_for_text_metric(text)
    tokens = normalized.split()

    if not tokens:
        return None

    if tokens[0] in {"yes", "no"}:
        return tokens[0]

    if "yes" in tokens and "no" not in tokens:
        return "yes"
    if "no" in tokens and "yes" not in tokens:
        return "no"

    return None


def important_tokens(text: str) -> list[str]:
    tokens = tokenize(text)

    kept = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if len(tok) <= 1:
            continue
        kept.append(tok)

    return kept


def important_token_recall(reference: str, prediction: str) -> float:
    ref_tokens = important_tokens(reference)
    pred_tokens = set(important_tokens(prediction))

    if not ref_tokens:
        return 1.0

    preserved = sum(1 for tok in ref_tokens if tok in pred_tokens)
    return preserved / len(ref_tokens)


def added_important_token_rate(reference: str, prediction: str) -> float:
    ref_tokens = set(important_tokens(reference))
    pred_tokens = important_tokens(prediction)

    if not pred_tokens:
        return 0.0

    added = sum(1 for tok in pred_tokens if tok not in ref_tokens)
    return added / len(pred_tokens)


def extract_numbers(text: str) -> list[str]:
    normalized = normalize_for_text_metric(text)
    return re.findall(r"\b\d+(?:\.\d+)?\b", normalized)


def number_preservation(reference: str, prediction: str) -> bool:
    ref_numbers = extract_numbers(reference)
    pred_numbers = set(extract_numbers(prediction))

    return all(num in pred_numbers for num in ref_numbers)


def length_ratio(reference: str, prediction: str) -> float:
    ref_len = max(len(normalize_for_text_metric(reference)), 1)
    pred_len = len(normalize_for_text_metric(prediction))
    return pred_len / ref_len


def contains_normalized_answer(reference: str, prediction: str) -> bool:
    ref = normalize_for_text_metric(reference)
    pred = normalize_for_text_metric(prediction)

    if not ref or not pred:
        return ref == pred

    return ref in pred or pred in ref


def information_preservation_pass(
    reference: str,
    prediction: str,
    token_f1_value: float,
    important_recall_value: float,
    added_rate_value: float,
    yes_no_consistent: bool | None,
    numbers_preserved: bool,
) -> bool:
    if yes_no_consistent is False:
        return False

    if not numbers_preserved:
        return False

    if contains_normalized_answer(reference, prediction):
        return added_rate_value <= 0.50

    if important_recall_value < 0.80:
        return False

    if token_f1_value < 0.60:
        return False

    if added_rate_value > 0.60:
        return False

    return True


def compute_metrics_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
    """Parallel batch function.

    It compares original rows in `batch` with processed rows at the same
    dataset indices in the global processed dataset.
    """

    if PROCESSED_DS is None:
        raise RuntimeError("PROCESSED_DS is not initialized.")

    id_key = GLOBAL_ARGS["id_key"]
    question_key = GLOBAL_ARGS["question_key"]
    answer_key = GLOBAL_ARGS["answer_key"]
    original_conversations_key = GLOBAL_ARGS["original_conversations_key"]
    processed_answer_key = GLOBAL_ARGS["processed_answer_key"]
    processed_conversations_key = GLOBAL_ARGS["processed_conversations_key"]

    processed_batch = PROCESSED_DS[list(indices)]

    output: dict[str, list[Any]] = {
        "id": [],
        "processed_id": [],
        "id_match": [],
        "question": [],
        "original_answer": [],
        "processed_answer": [],
        "normalized_original": [],
        "normalized_processed": [],
        "exact_match": [],
        "token_f1": [],
        "rouge_l_f1": [],
        "length_ratio": [],
        "original_yes_no": [],
        "processed_yes_no": [],
        "yes_no_consistent": [],
        "important_token_recall": [],
        "added_important_token_rate": [],
        "numbers_preserved": [],
        "information_preservation_pass": [],
    }

    # Any present column has the same batch length.
    num_rows = len(next(iter(batch.values()))) if batch else 0

    for j in range(num_rows):
        global_index = int(indices[j])
        sample_id = get_row_id(batch, id_key, j, global_index)
        processed_id = get_row_id(processed_batch, id_key, j, global_index)

        question = get_question_from_batch(
            batch=batch,
            i=j,
            question_key=question_key,
            conversations_key=original_conversations_key,
        )
        original_answer = get_answer_from_batch(
            batch=batch,
            i=j,
            answer_key=answer_key,
            conversations_key=original_conversations_key,
        )
        processed_answer = get_answer_from_batch(
            batch=processed_batch,
            i=j,
            answer_key=processed_answer_key,
            conversations_key=processed_conversations_key,
        )

        em = exact_match(original_answer, processed_answer)
        f1 = token_f1(original_answer, processed_answer)
        rouge_l = rouge_l_f1(original_answer, processed_answer)
        ratio = length_ratio(original_answer, processed_answer)

        original_yes_no = yes_no_label(original_answer)
        processed_yes_no = yes_no_label(processed_answer)

        yes_no_consistent = None
        if original_yes_no is not None:
            yes_no_consistent = original_yes_no == processed_yes_no

        important_recall = important_token_recall(original_answer, processed_answer)
        added_rate = added_important_token_rate(original_answer, processed_answer)
        numbers_preserved = number_preservation(original_answer, processed_answer)

        info_pass = information_preservation_pass(
            reference=original_answer,
            prediction=processed_answer,
            token_f1_value=f1,
            important_recall_value=important_recall,
            added_rate_value=added_rate,
            yes_no_consistent=yes_no_consistent,
            numbers_preserved=numbers_preserved,
        )

        output["id"].append(sample_id)
        output["processed_id"].append(processed_id)
        output["id_match"].append(sample_id == processed_id)
        output["question"].append(question)
        output["original_answer"].append(original_answer)
        output["processed_answer"].append(processed_answer)
        output["normalized_original"].append(normalize_for_text_metric(original_answer))
        output["normalized_processed"].append(normalize_for_text_metric(processed_answer))
        output["exact_match"].append(em)
        output["token_f1"].append(float(f1))
        output["rouge_l_f1"].append(float(rouge_l))
        output["length_ratio"].append(float(ratio))
        output["original_yes_no"].append(original_yes_no)
        output["processed_yes_no"].append(processed_yes_no)
        output["yes_no_consistent"].append(yes_no_consistent)
        output["important_token_recall"].append(float(important_recall))
        output["added_important_token_rate"].append(float(added_rate))
        output["numbers_preserved"].append(numbers_preserved)
        output["information_preservation_pass"].append(info_pass)

    return output


def safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")

    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    floor = math.floor(k)
    ceil = math.ceil(k)

    if floor == ceil:
        return float(values_sorted[int(k)])

    lower = values_sorted[floor] * (ceil - k)
    upper = values_sorted[ceil] * (k - floor)
    return float(lower + upper)


def iter_dataset_batches(ds: Dataset, batch_size: int):
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        yield ds[start:end]


def row_score(row: dict[str, Any]) -> tuple:
    """Lower is worse."""

    return (
        1 if row["information_preservation_pass"] else 0,
        float(row["important_token_recall"]),
        float(row["token_f1"]),
        float(row["rouge_l_f1"]),
    )


def aggregate_and_write_outputs(
    metrics_ds: Dataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sums = {
        "normalized_exact_match": 0.0,
        "token_f1": 0.0,
        "rouge_l_f1": 0.0,
        "length_ratio": 0.0,
        "important_token_recall": 0.0,
        "added_important_token_rate": 0.0,
        "number_preservation": 0.0,
        "information_preservation_pass": 0.0,
        "id_match": 0.0,
    }

    counts = {
        "rows": 0,
        "yes_no": 0,
        "yes_no_consistent": 0,
        "empty_original_answer": 0,
        "empty_processed_answer": 0,
    }

    # Optional exact percentile computation. Disabled by default to avoid
    # holding giant float lists for huge datasets.
    f1_values: list[float] = []
    important_recall_values: list[float] = []
    length_ratio_values: list[float] = []

    lowest_rows: list[dict[str, Any]] = []

    csv_writer = None
    csv_file = None

    fieldnames = [
        "id",
        "processed_id",
        "id_match",
        "question",
        "original_answer",
        "processed_answer",
        "normalized_original",
        "normalized_processed",
        "exact_match",
        "token_f1",
        "rouge_l_f1",
        "length_ratio",
        "original_yes_no",
        "processed_yes_no",
        "yes_no_consistent",
        "important_token_recall",
        "added_important_token_rate",
        "numbers_preserved",
        "information_preservation_pass",
    ]

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    printed = 0

    try:
        for batch in iter_dataset_batches(metrics_ds, args.aggregate_batch_size):
            batch_size = len(batch["id"])

            for i in range(batch_size):
                row = {key: batch[key][i] for key in fieldnames}

                counts["rows"] += 1

                if not normalize_for_text_metric(row["original_answer"]):
                    counts["empty_original_answer"] += 1
                if not normalize_for_text_metric(row["processed_answer"]):
                    counts["empty_processed_answer"] += 1

                sums["id_match"] += 1.0 if row["id_match"] else 0.0
                sums["normalized_exact_match"] += 1.0 if row["exact_match"] else 0.0
                sums["token_f1"] += float(row["token_f1"])
                sums["rouge_l_f1"] += float(row["rouge_l_f1"])
                sums["length_ratio"] += float(row["length_ratio"])
                sums["important_token_recall"] += float(row["important_token_recall"])
                sums["added_important_token_rate"] += float(row["added_important_token_rate"])
                sums["number_preservation"] += 1.0 if row["numbers_preserved"] else 0.0
                sums["information_preservation_pass"] += (
                    1.0 if row["information_preservation_pass"] else 0.0
                )

                if row["original_yes_no"] is not None:
                    counts["yes_no"] += 1
                    if row["yes_no_consistent"] is True:
                        counts["yes_no_consistent"] += 1

                if args.compute_percentiles:
                    f1_values.append(float(row["token_f1"]))
                    important_recall_values.append(float(row["important_token_recall"]))
                    length_ratio_values.append(float(row["length_ratio"]))

                if csv_writer is not None:
                    csv_writer.writerow(row)

                if args.print_examples and printed < args.max_print_examples:
                    print("=" * 120)
                    print(f"id: {row['id']}")
                    print(f"processed_id: {row['processed_id']}")
                    print(f"id_match: {row['id_match']}")
                    print(f"question: {row['question']}")
                    print(f"original_answer: {row['original_answer']}")
                    print(f"processed_answer: {row['processed_answer']}")
                    print(f"exact_match: {row['exact_match']}")
                    print(f"token_f1: {float(row['token_f1']):.4f}")
                    print(f"rouge_l_f1: {float(row['rouge_l_f1']):.4f}")
                    print(f"important_token_recall: {float(row['important_token_recall']):.4f}")
                    print(
                        f"added_important_token_rate: "
                        f"{float(row['added_important_token_rate']):.4f}"
                    )
                    print(f"numbers_preserved: {row['numbers_preserved']}")
                    print(f"yes_no_consistent: {row['yes_no_consistent']}")
                    print(
                        f"information_preservation_pass: "
                        f"{row['information_preservation_pass']}"
                    )
                    printed += 1

                lowest_rows.append(row)
                if len(lowest_rows) > args.num_lowest * 4:
                    lowest_rows.sort(key=row_score)
                    lowest_rows = lowest_rows[: args.num_lowest]

    finally:
        if csv_file is not None:
            csv_file.close()

    n = max(counts["rows"], 1)

    summary = {
        "num_aligned": counts["rows"],
        "empty_original_answer": counts["empty_original_answer"],
        "empty_processed_answer": counts["empty_processed_answer"],
        "id_match_rate": sums["id_match"] / n,
        "normalized_exact_match": sums["normalized_exact_match"] / n,
        "token_f1_mean": sums["token_f1"] / n,
        "rouge_l_f1_mean": sums["rouge_l_f1"] / n,
        "length_ratio_mean": sums["length_ratio"] / n,
        "important_token_recall_mean": sums["important_token_recall"] / n,
        "added_important_token_rate_mean": sums["added_important_token_rate"] / n,
        "number_preservation_rate": sums["number_preservation"] / n,
        "information_preservation_pass_rate": sums["information_preservation_pass"] / n,
        "yes_no_num": counts["yes_no"],
        "yes_no_consistency": (
            counts["yes_no_consistent"] / counts["yes_no"]
            if counts["yes_no"] > 0
            else None
        ),
    }

    if args.compute_percentiles:
        summary.update(
            {
                "token_f1_median": safe_median(f1_values),
                "token_f1_p05": percentile(f1_values, 0.05),
                "important_token_recall_median": safe_median(important_recall_values),
                "important_token_recall_p05": percentile(important_recall_values, 0.05),
                "length_ratio_median": safe_median(length_ratio_values),
                "length_ratio_p05": percentile(length_ratio_values, 0.05),
                "length_ratio_p95": percentile(length_ratio_values, 0.95),
            }
        )

    lowest_rows.sort(key=row_score)
    lowest_rows = lowest_rows[: args.num_lowest]

    print("\nSummary metrics:")
    print(json.dumps(summary, indent=2))

    print("\nLowest-scoring examples:")
    for row in lowest_rows:
        print("-" * 80)
        print(f"id: {row['id']}")
        print(f"processed_id: {row['processed_id']}")
        print(f"id_match: {row['id_match']}")
        print(f"information_preservation_pass: {row['information_preservation_pass']}")
        print(f"important_token_recall: {float(row['important_token_recall']):.4f}")
        print(f"added_important_token_rate: {float(row['added_important_token_rate']):.4f}")
        print(f"token_f1: {float(row['token_f1']):.4f}")
        print(f"rouge_l_f1: {float(row['rouge_l_f1']):.4f}")
        print(f"yes_no_consistent: {row['yes_no_consistent']}")
        print(f"numbers_preserved: {row['numbers_preserved']}")
        print(f"question: {row['question']}")
        print(f"original: {row['original_answer']}")
        print(f"processed: {row['processed_answer']}")

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary metrics to: {output_json}")

    if args.output_csv:
        print(f"Wrote per-example metrics to: {Path(args.output_csv).expanduser().resolve()}")

    return summary


def print_schema(name: str, ds: Dataset) -> None:
    print(f"{name} columns: {ds.column_names}")
    print(f"{name} rows: {len(ds):,}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--original-path", required=True)
    parser.add_argument("--processed-path", required=True)

    parser.add_argument("--original-split", default="train")
    parser.add_argument("--processed-split", default="train")

    parser.add_argument(
        "--alignment",
        choices=["order", "id"],
        default="order",
        help=(
            "order: compare row i with row i, fastest. "
            "id: sort both datasets by id before comparing."
        ),
    )

    parser.add_argument(
        "--id-key",
        default="id",
        help="Optional ID column. If missing and alignment=order, row index is used.",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="Optional original question column. If missing, extracted from conversations.",
    )
    parser.add_argument(
        "--answer-key",
        default="answer",
        help="Optional original answer column. If missing, extracted from conversations.",
    )
    parser.add_argument(
        "--original-conversations-key",
        default="conversations",
        help="Original dataset conversation column used when question/answer columns are absent.",
    )
    parser.add_argument(
        "--processed-answer-key",
        default=None,
        help="Optional processed answer column. If missing, extracted from processed conversations.",
    )
    parser.add_argument(
        "--processed-conversations-key",
        default="conversations",
        help="Processed dataset conversation column.",
    )

    parser.add_argument("--max-examples", type=int, default=None)

    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of parallel map workers.",
    )
    parser.add_argument(
        "--map-batch-size",
        type=int,
        default=512,
        help="Batch size used by datasets.map.",
    )
    parser.add_argument(
        "--aggregate-batch-size",
        type=int,
        default=10000,
        help="Batch size used for streaming aggregation.",
    )

    parser.add_argument("--print-examples", action="store_true")
    parser.add_argument("--max-print-examples", type=int, default=50)

    parser.add_argument("--num-lowest", type=int, default=20)

    parser.add_argument(
        "--compute-percentiles",
        action="store_true",
        help=(
            "Compute exact medians/percentiles. This stores metric arrays in memory, "
            "so leave disabled for very large datasets."
        ),
    )

    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)

    args = parser.parse_args()

    args.id_key = none_if_blank(args.id_key)
    args.question_key = none_if_blank(args.question_key)
    args.answer_key = none_if_blank(args.answer_key)
    args.original_conversations_key = none_if_blank(args.original_conversations_key)
    args.processed_answer_key = none_if_blank(args.processed_answer_key)
    args.processed_conversations_key = none_if_blank(args.processed_conversations_key)

    original_path = Path(args.original_path).expanduser().resolve()
    processed_path = Path(args.processed_path).expanduser().resolve()

    print(f"Loading original dataset: {original_path}")
    original_ds = as_dataset(load_from_disk(str(original_path)), args.original_split)

    print(f"Loading processed dataset: {processed_path}")
    processed_ds = as_dataset(load_from_disk(str(processed_path)), args.processed_split)

    print_schema("Original", original_ds)
    print_schema("Processed", processed_ds)

    # Auto-detect common non-chat schemas. MedTrinity demo originals are often
    # image-caption datasets with columns like: image, id, caption.
    args.answer_key = resolve_column(
        original_ds,
        args.answer_key,
        ["caption", "answer", "answers", "text", "report", "description"],
    )
    args.question_key = resolve_column(
        original_ds,
        args.question_key,
        ["question", "prompt", "instruction", "query"],
    )
    args.processed_answer_key = resolve_column(
        processed_ds,
        args.processed_answer_key,
        ["answer", "response", "output", "caption", "text", "report"],
    )

    print("Resolved columns:")
    print(f"  id_key: {args.id_key!r}")
    print(f"  question_key: {args.question_key!r}")
    print(f"  answer_key: {args.answer_key!r}")
    print(f"  original_conversations_key: {args.original_conversations_key!r}")
    print(f"  processed_answer_key: {args.processed_answer_key!r}")
    print(f"  processed_conversations_key: {args.processed_conversations_key!r}")

    if args.alignment == "id":
        if not args.id_key:
            raise ValueError("--alignment id requires --id-key")
        if args.id_key not in original_ds.column_names:
            raise ValueError(
                f"Original dataset missing id column {args.id_key!r}. "
                f"Available columns: {original_ds.column_names}"
            )
        if args.id_key not in processed_ds.column_names:
            raise ValueError(
                f"Processed dataset missing id column {args.id_key!r}. "
                f"Available columns: {processed_ds.column_names}"
            )

    original_has_answer = bool(args.answer_key and args.answer_key in original_ds.column_names)
    original_has_conversations = bool(
        args.original_conversations_key
        and args.original_conversations_key in original_ds.column_names
    )
    processed_has_answer = bool(
        args.processed_answer_key and args.processed_answer_key in processed_ds.column_names
    )
    processed_has_conversations = bool(
        args.processed_conversations_key
        and args.processed_conversations_key in processed_ds.column_names
    )

    if not original_has_answer and not original_has_conversations:
        raise ValueError(
            "Could not find an original answer source. Provide --answer-key or "
            "--original-conversations-key. Available original columns: "
            f"{original_ds.column_names}"
        )

    if not processed_has_answer and not processed_has_conversations:
        raise ValueError(
            "Could not find a processed answer source. Provide --processed-answer-key or "
            "--processed-conversations-key. Available processed columns: "
            f"{processed_ds.column_names}"
        )

    original_keep = existing_columns(
        original_ds,
        [
            args.id_key,
            args.question_key,
            args.answer_key,
            args.original_conversations_key,
        ],
    )
    processed_keep = existing_columns(
        processed_ds,
        [
            args.id_key,
            args.processed_answer_key,
            args.processed_conversations_key,
        ],
    )

    # Drop images and all unnecessary columns to avoid image decoding.
    original_ds = original_ds.select_columns(original_keep)
    processed_ds = processed_ds.select_columns(processed_keep)

    if args.max_examples is not None:
        original_ds = original_ds.select(range(min(args.max_examples, len(original_ds))))
        processed_ds = processed_ds.select(range(min(args.max_examples, len(processed_ds))))

    if args.alignment == "id":
        print("Sorting original and processed datasets by id")
        original_ds = original_ds.sort(args.id_key)
        processed_ds = processed_ds.sort(args.id_key)

    n = min(len(original_ds), len(processed_ds))
    if len(original_ds) != len(processed_ds):
        print(
            f"WARNING: dataset lengths differ. "
            f"original={len(original_ds):,}, processed={len(processed_ds):,}. "
            f"Comparing first {n:,} rows after alignment."
        )
        original_ds = original_ds.select(range(n))
        processed_ds = processed_ds.select(range(n))

    print(f"Examples to compare: {n:,}")
    print(f"Parallel workers: {args.num_proc}")
    print(f"Map batch size: {args.map_batch_size}")
    print("Sources:")
    print(
        "  original answer:",
        args.answer_key if original_has_answer else f"{args.original_conversations_key} assistant message",
    )
    print(
        "  processed answer:",
        args.processed_answer_key
        if processed_has_answer
        else f"{args.processed_conversations_key} assistant message",
    )
    print(
        "  question:",
        args.question_key
        if args.question_key and args.question_key in original_ds.column_names
        else f"{args.original_conversations_key} user message",
    )

    global PROCESSED_DS
    global GLOBAL_ARGS

    PROCESSED_DS = processed_ds
    GLOBAL_ARGS = {
        "id_key": args.id_key,
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "original_conversations_key": args.original_conversations_key,
        "processed_answer_key": args.processed_answer_key,
        "processed_conversations_key": args.processed_conversations_key,
    }

    metrics_features = Features(
        {
            "id": Value("string"),
            "processed_id": Value("string"),
            "id_match": Value("bool"),
            "question": Value("string"),
            "original_answer": Value("string"),
            "processed_answer": Value("string"),
            "normalized_original": Value("string"),
            "normalized_processed": Value("string"),
            "exact_match": Value("bool"),
            "token_f1": Value("float64"),
            "rouge_l_f1": Value("float64"),
            "length_ratio": Value("float64"),
            "original_yes_no": Value("string"),
            "processed_yes_no": Value("string"),
            "yes_no_consistent": Value("bool"),
            "important_token_recall": Value("float64"),
            "added_important_token_rate": Value("float64"),
            "numbers_preserved": Value("bool"),
            "information_preservation_pass": Value("bool"),
        }
    )

    metrics_ds = original_ds.map(
        compute_metrics_batch,
        batched=True,
        batch_size=args.map_batch_size,
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=original_ds.column_names,
        features=metrics_features,
        desc="Computing comparison metrics",
    )

    aggregate_and_write_outputs(metrics_ds, args)


if __name__ == "__main__":
    main()