#!/usr/bin/env python3
"""Run ChartQA transformation with NeMo Curator + Data Designer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_designer.config as dd
from data_designer.config.column_configs import GenerationStrategy
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks import DocumentBatch
from pydantic import BaseModel

from chartqa_custom import normalize_generated_answer, normalize_query


class VLMResult(BaseModel):
    answer: str
    rationale: str


@dataclass
class ResolveChartQAImagePathStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Resolve relative image paths for Data Designer while preserving output refs."""

    image_base_path: str
    name: str = "resolve_chartqa_image_paths"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        if "image_path" not in df.columns:
            raise ValueError("Missing required image_path column")
        base = Path(self.image_base_path).resolve()
        df["image_ref"] = df["image_path"]
        df["image_path"] = df["image_path"].map(lambda value: str(Path(value) if Path(str(value)).is_absolute() else base / str(value)))
        return DocumentBatch(dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf)


@dataclass
class RenderNestedChartQAStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Render Data Designer columns into the target nested training-data schema."""

    name: str = "render_nested_chartqa"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["id", "messages", "metadata"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        required = {"id", "image_ref", "normalized_query", "vlm_result", "reference_answer", "generated_answer_normalized", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for nested ChartQA render: {sorted(missing)}")
        rows = []
        for row in df.to_dict(orient="records"):
            vlm_result = row.get("vlm_result") or {}
            if isinstance(vlm_result, str):
                try:
                    vlm_result = json.loads(vlm_result)
                except json.JSONDecodeError:
                    vlm_result = {"answer": vlm_result, "rationale": ""}
            rows.append(
                {
                    "id": row["id"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": row["image_ref"]},
                                {"type": "text", "text": row["normalized_query"]},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": f"{vlm_result.get('answer', '')}\n\nRationale: {vlm_result.get('rationale', '')}",
                        },
                    ],
                    "metadata": {
                        "reference_answer": row["reference_answer"],
                        "generated_answer_normalized": row["generated_answer_normalized"],
                        "source": row.get("source", "ChartQA"),
                    },
                }
            )
        return DocumentBatch(dataset_name=batch.dataset_name, data=pd.DataFrame(rows), _metadata=batch._metadata, _stage_perf=batch._stage_perf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model", default=os.environ.get("CHARTQA_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--base-url", default=os.environ.get("CHARTQA_OPENAI_BASE_URL", "http://127.0.0.1:30000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("CHARTQA_OPENAI_API_KEY", "unused"))
    parser.add_argument("--provider-name", default="shared_sglang")
    parser.add_argument("--model-alias", default="chartqa_vlm")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-parallel-requests", type=int, default=64)
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--dry-run-validate", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[dd.DataDesignerConfigBuilder, list[dd.ModelProvider]]:
    provider = dd.ModelProvider(
        name=args.provider_name,
        endpoint=args.base_url,
        provider_type="openai",
        api_key=args.api_key,
    )
    model = dd.ModelConfig(
        alias=args.model_alias,
        model=args.model,
        provider=args.provider_name,
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_parallel_requests=args.max_parallel_requests,
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model])
    builder.add_column(
        dd.CustomColumnConfig(
            name="normalized_query",
            generator_function=normalize_query,
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="vlm_result",
            model_alias=args.model_alias,
            prompt=(
                "You are transforming ChartQA examples into training data. "
                "Use the chart image and the question to answer concisely.\n\n"
                "Question: {{ normalized_query }}\n\n"
                "Return JSON only with two string fields: answer and rationale."
            ),
            output_format=VLMResult,
            multi_modal_context=[dd.ImageContext(column_name="image_path")],
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="generated_answer_normalized",
            generator_function=normalize_generated_answer,
            generation_strategy=GenerationStrategy.CELL_BY_CELL,
        )
    )
    return builder, [provider]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    summary: dict[str, Any] = {"returncode": 0}
    try:
        config_builder, providers = build_config(args)
        if args.dry_run_validate:
            from data_designer.interface import DataDesigner

            DataDesigner(model_providers=providers).validate(config_builder)

        pipeline = Pipeline(name="chartqa_nemo_data_designer", description="ChartQA nested VLM transformation")
        pipeline.add_stage(JsonlReader(file_paths=args.input_jsonl, files_per_partition=args.files_per_partition))
        pipeline.add_stage(ResolveChartQAImagePathStage(image_base_path=args.image_base_path))
        pipeline.add_stage(DataDesignerStage(config_builder=config_builder, model_providers=providers, verbose=False))
        pipeline.add_stage(RenderNestedChartQAStage())
        pipeline.add_stage(JsonlWriter(path=args.output_dir))
        result_tasks = pipeline.run(executor=XennaExecutor(config={"execution_mode": "streaming"}))
        summary["result_tasks"] = len(result_tasks or [])
    except Exception as exc:
        summary["returncode"] = 1
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["full_end_to_end_wall_seconds"] = round(time.perf_counter() - started, 6)
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
