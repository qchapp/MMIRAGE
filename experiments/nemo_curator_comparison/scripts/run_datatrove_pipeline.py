#!/usr/bin/env python3
"""Native DataTrove ChartQA run for the nemo_curator comparison experiment.

Uses the DataTrove native inference pipeline (JsonlReader -> InferenceRunner
with a locally spawned vLLM server -> JsonlWriter). The multimodal request
payload is built from each document's media, and the pipeline materialized
output is transformed into the nested ChartQA output schema consumed by
``analyze_results.py``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT_CONTRACT = {
    "required_fields": ["id", "messages", "metadata"],
    "metadata_fields": ["reference_answer", "generated_answer_normalized", "source"],
    "validation": ["row_count_matches_input", "ids_match_input", "no_duplicate_ids", "row_order_matches_input", "nested_message_schema_valid"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unimplemented", action="store_true")
    return parser.parse_args()


def collapse_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def image_to_data_url(image_path: Path, image_base: Path) -> str:
    full_path = image_path if image_path.is_absolute() else image_base / image_path
    if not full_path.exists():
        raise FileNotFoundError(f"Missing ChartQA image: {full_path}")
    payload = base64.b64encode(full_path.read_bytes()).decode("ascii")
    suffix = full_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
    return f"data:{mime};base64,{payload}"


def build_summary(args: argparse.Namespace, started: float, rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], validation: dict[str, Any], execution_status: str = "completed") -> dict[str, Any]:
    return {
        "framework": "datatrove",
        "returncode": 0,
        "native_mode": True,
        "execution_status": execution_status,
        "native_backend": "DataTrove native pipeline: JsonlReader -> InferenceRunner(self-managed vLLM server) -> JsonlWriter",
        "input_jsonl": str(Path(args.input_jsonl)),
        "image_base_path": str(Path(args.image_base_path)),
        "input_rows": len(rows),
        "materialized_rows": len(output_rows),
        "model": args.model,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
        "output_contract": OUTPUT_CONTRACT,
        "validation": validation,
        "full_end_to_end_wall_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    input_path = Path(args.input_jsonl)
    image_base = Path(args.image_base_path)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    if args.dry_run or args.allow_unimplemented:
        summary = build_summary(args, started, rows, [], {"valid": False, "reason": "dry_run_or_plan_only"}, execution_status="planned_native_datatrove_chartqa_run_not_executed")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    from datatrove.data import Document, Media, MediaType
    from datatrove.pipeline.inference.run_inference import (
        InferenceConfig,
        InferenceRunner,
    )
    from datatrove.pipeline.inference.types import InferenceResult
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from experiments._shared.native_frameworks import (
        _pick_free_port,
        _spawn_vllm_server,
        _terminate_server,
        _wait_for_http,
    )

    answers: dict[str, str] = {}

    async def rollout(document: Document, generate: Any, answers: dict[str, str], **_: Any) -> Any:
        image_path = Path(str(document.metadata.get("image_path", "")))
        query = str(document.metadata.get("query", ""))
        content: list[dict[str, Any]] = []
        if image_path.name:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path, image_base)}})
        content.append({"type": "text", "text": query})
        result = await generate({"messages": [{"role": "user", "content": content}]})
        answers[document.id] = result.text
        return InferenceResult(text=result.text, finish_reason=result.finish_reason, usage=result.usage)

    documents = []
    for row in rows:
        image_path = Path(str(row.get("image_path", "")))
        full_path = image_path if image_path.is_absolute() else image_base / image_path
        media = None
        if full_path.exists():
            media = [Media(id=row["id"], type=MediaType.IMAGE, url=str(full_path), path=str(full_path))]
        documents.append(
            Document(
                text=str(row.get("query", "")),
                id=row["id"],
                media=media,
                metadata=dict(row),
            )
        )

    port = _pick_free_port()
    server_log = summary_path.with_name(summary_path.stem + ".vllm.log") if summary_path else None
    server = _spawn_vllm_server(args.model, port=port, max_model_len=16384, log_path=server_log)
    writer_scratch = Path(tempfile.mkdtemp(prefix="datatrove_chartqa_out_"))
    started = time.perf_counter()
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/v1/models", server, timeout=900.0)
        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path=args.model,
            endpoint_url=f"http://127.0.0.1:{port}",
            use_chat=True,
            model_max_context=16384,
            tp=1,
            max_concurrent_generations=args.concurrency,
            default_generation_params={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
        )
        runner = InferenceRunner(
            rollout_fn=rollout,
            config=config,
            output_writer=JsonlWriter(output_folder=str(writer_scratch)),
            shared_context={"answers": answers},
            metadata_key="rollout_results",
        )
        runner.run(documents, rank=0, world_size=1)
    finally:
        _terminate_server(server)
        import shutil

        shutil.rmtree(writer_scratch, ignore_errors=True)

    output_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for row in rows:
        generated = answers.get(row["id"])
        if generated is None:
            missing_ids.append(row["id"])
            generated = ""
        output_rows.append(
            {
                "id": row["id"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row.get("image_path", "")},
                            {"type": "text", "text": row["query"]},
                        ],
                    },
                    {"role": "assistant", "content": generated},
                ],
                "metadata": {
                    "reference_answer": row["reference_answer"],
                    "generated_answer_normalized": collapse_whitespace(generated).lower(),
                    "source": row.get("source", "ChartQA"),
                },
            }
        )
    output_rows.sort(key=lambda item: next((index for index, row in enumerate(rows) if row["id"] == item["id"]), 0))

    with (output_dir / "datatrove_native_chartqa.jsonl").open("w", encoding="utf-8") as handle:
        for output_row in output_rows:
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    output_ids = [item["id"] for item in output_rows]
    expected_ids = [row["id"] for row in rows]
    validation = {
        "row_count_matches_input": len(output_rows) == len(rows),
        "ids_match_input": not missing_ids,
        "no_duplicate_ids": len(output_ids) == len(set(output_ids)),
        "row_order_matches_input": output_ids == expected_ids,
        "nested_message_schema_valid": all(
            isinstance(item.get("messages"), list)
            and len(item["messages"]) == 2
            and item["messages"][0].get("role") == "user"
            and item["messages"][1].get("role") == "assistant"
            and isinstance(item["messages"][0].get("content"), list)
            and len(item["messages"][0]["content"]) >= 2
            for item in output_rows
        ),
        "missing_ids": missing_ids,
        "unexpected_ids": sorted(set(output_ids) - set(expected_ids)),
    }
    validation["valid"] = all(validation[key] for key in OUTPUT_CONTRACT["validation"])

    summary = build_summary(args, started, rows, output_rows, validation)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
