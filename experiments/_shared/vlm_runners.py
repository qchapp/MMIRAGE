"""Native VLM baselines for the vlm_enrichment task (image -> markdown caption).

Each ``run_<framework>_vlm(...)`` consumes the same workload rows (with ``id``,
``image_path`` and ``caption`` fields), sends the image and caption to the model
through the framework's own serving stack, and writes ``{id, formatted_description}``
rows to ``output_path``. Stats are returned in the MMIRAGE shard-status schema
so existing aggregators can consume them.

* ``run_sglang_vlm`` uses one SGLang HTTP server + the shared OpenAI client.
* ``run_datatrove_vlm`` uses DataTrove ``InferenceRunner`` with a self-managed
  vLLM server and native media support.
* ``run_nemo_curator_vlm`` uses NeMo Curator ``OpenAIClient`` with a self-managed
  vLLM server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from experiments._shared.io import read_jsonl, write_jsonl
from experiments._shared.native_frameworks import (
    _pick_free_port,
    _runner_stats,
    _spawn_vllm_server,
    _terminate_server,
    _wait_for_http,
)
from experiments._shared.sglang_client import image_data_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

# Single definition of the vlm_enrichment transformation. It must stay
# byte-identical (modulo the placeholder name) to the ``prompt`` of the
# ``formatted_description`` output in the MMIRAGE vlm recipe
# (experiments/task_comparison/vlm_enrichment/configs/mmirage_vlm.yaml);
# run_setup.py verifies this before launching any vlm run.
VLM_REFORMAT_TEMPLATE = (
    "Reformat the image description with markdown without adding anything else.\n"
    "Add titles and structure your output.\n\n"
    "Image description:\n"
    "{caption}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["sglang", "datatrove", "nemo_curator"], required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-base-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--text-field", default="caption")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


def _runner_status(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "rows_processed": int(stats["rows_processed"]),
        "stats": stats,
        "output_tokens": int(stats["output_tokens"]),
        "inference_runtime_seconds": stats["inference_runtime_seconds"],
        "runtime_seconds": stats["runtime_seconds"],
        "model_load_seconds": stats["model_load_seconds"],
    }


def _contract_rows(rows: List[Dict[str, Any]], answers: Dict[str, str], id_field: str) -> List[Dict[str, Any]]:
    contract: List[Dict[str, Any]] = []
    for row in rows:
        item = {id_field: row[id_field], "formatted_description": answers.get(str(row[id_field]), "")}
        if "source_index" in row:
            item["source_index"] = row["source_index"]
        contract.append(item)
    return contract


def _write_contract(output_path: Path, rows: List[Dict[str, Any]], answers: Dict[str, str], id_field: str) -> None:
    write_jsonl(output_path, _contract_rows(rows, answers, id_field))


def _output_tokens(answers: Dict[str, str], model: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return sum(len(tokenizer.encode(text)) for text in answers.values())


def _spawn_sglang_server(model: str, port: int, log_path: Optional[Path]) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--disable-custom-all-reduce",
        "--max-running-requests",
        "1000",
    ]
    env = os.environ.copy()
    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8")
        return subprocess.Popen(command, stdout=log_fh, stderr=log_fh, env=env, start_new_session=True)
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)


def _terminate_sglang_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_sglang_vlm(
    rows: List[Dict[str, Any]],
    model: str,
    image_base: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    image_field: str = "image_path",
    text_field: str = "caption",
    id_field: str = "id",
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """One SGLang server, then the shared OpenAI client for every row."""
    port = _pick_free_port()
    server_log = log_path.with_name(log_path.stem + ".sglang.log") if log_path else None
    server = _spawn_sglang_server(model, port, server_log)
    model_load_seconds = 0.0
    runtime_seconds = 0.0
    try:
        with tempfile.TemporaryDirectory(prefix="sglang_vlm_") as scratch:
            scratch_dir = Path(scratch)
            prompts_path = scratch_dir / "prompts.jsonl"
            summary_path = scratch_dir / "summary.json"
            output_rows_path = scratch_dir / "outputs.jsonl"
            started = time.perf_counter()
            _wait_for_http(f"http://127.0.0.1:{port}/v1/models", server, timeout=900.0)
            model_load_seconds = time.perf_counter() - started
            prompt_rows = [
                {**row, "__vlm_prompt_text": VLM_REFORMAT_TEMPLATE.format(caption=str(row.get(text_field, "")))}
                for row in rows
            ]
            write_jsonl(prompts_path, prompt_rows)
            command = [
                sys.executable,
                "-m",
                "experiments._shared.sglang_client",
                "--prompts-jsonl",
                str(prompts_path),
                "--output-jsonl",
                str(output_rows_path),
                "--summary-json",
                str(summary_path),
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--model",
                model,
                "--tokenizer",
                model,
                "--chat",
                "--text-field",
                "__vlm_prompt_text",
                "--image-field",
                image_field,
                "--image-base-path",
                str(image_base),
                "--id-field",
                id_field,
                "--max-tokens",
                str(max_new_tokens),
                "--temperature",
                str(temperature),
                "--top-p",
                str(top_p),
                "--concurrency",
                str(concurrency),
                "--gpu-count",
                str(gpu_count),
                "--path",
                "sglang_vlm",
            ]
            wall = time.perf_counter()
            subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
            generation_seconds = time.perf_counter() - wall
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runtime_seconds = model_load_seconds + generation_seconds
            answers: Dict[str, str] = {}
            failed_rows: List[Dict[str, Any]] = []
            for item in read_jsonl(output_rows_path):
                if item.get("status") == "success":
                    answers[str(item.get(id_field))] = item.get("output_text", "")
                else:
                    failed_rows.append(item)
            if rows and not answers and failed_rows:
                samples = "; ".join(str(r.get("error")) for r in failed_rows[:3])
                raise RuntimeError(
                    f"sglang_vlm returned 0/{len(rows)} successful rows; first errors: {samples}. "
                    f"Server log: {server_log}"
                )
            _write_contract(output_path, rows, answers, id_field)
            output_tokens = int(summary.get("total_output_tokens") or _output_tokens(answers, model))
            stats = _runner_stats(rows, 0, output_tokens, model_load_seconds, runtime_seconds)
            return stats
    finally:
        _terminate_sglang_server(server)


def run_datatrove_vlm(
    rows: List[Dict[str, Any]],
    model: str,
    image_base: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    image_field: str = "image_path",
    text_field: str = "caption",
    id_field: str = "id",
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """DataTrove InferenceRunner with native media against a self-managed vLLM server."""
    from datatrove.data import Document, Media, MediaType
    from datatrove.pipeline.inference.run_inference import (
        InferenceConfig,
        InferenceRunner,
    )
    from datatrove.pipeline.inference.types import InferenceResult
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    documents = []
    for row in rows:
        image_path = Path(str(row.get(image_field, "")))
        full_path = image_path if image_path.is_absolute() else image_base / image_path
        media = None
        if full_path.exists():
            media = [Media(id=str(row[id_field]), type=MediaType.IMAGE, url=str(full_path), path=str(full_path))]
        documents.append(
            Document(
                text=VLM_REFORMAT_TEMPLATE.format(caption=str(row.get(text_field, ""))),
                id=str(row[id_field]),
                media=media,
                metadata=dict(row),
            )
        )

    answers: Dict[str, str] = {}

    async def rollout(document: Document, generate: Callable, shared: Dict[str, str], **_: Any) -> Any:
        content: List[Dict[str, Any]] = []
        image_path = Path(str(document.metadata.get(image_field, "")))
        full_path = image_path if image_path.is_absolute() else image_base / image_path
        if full_path.exists():
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path, image_base)}})
        content.append({"type": "text", "text": document.text})
        result = await generate({"messages": [{"role": "user", "content": content}]})
        shared[document.id] = result.text
        return InferenceResult(text=result.text, finish_reason=result.finish_reason, usage=result.usage)

    port = _pick_free_port()
    server_log = log_path.with_name(log_path.stem + ".vllm.log") if log_path else None
    server = _spawn_vllm_server(model, port=port, max_model_len=8192, log_path=server_log)
    model_load_seconds = 0.0
    runtime_seconds = 0.0
    writer_scratch = Path(tempfile.mkdtemp(prefix="datatrove_vlm_out_"))
    try:
        started = time.perf_counter()
        _wait_for_http(f"http://127.0.0.1:{port}/v1/models", server, timeout=900.0)
        model_load_seconds = time.perf_counter() - started
        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path=model,
            endpoint_url=f"http://127.0.0.1:{port}",
            use_chat=True,
            tp=1,
            max_concurrent_generations=max(1, concurrency),
            default_generation_params={
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            },
        )
        runner = InferenceRunner(
            rollout_fn=rollout,
            config=config,
            output_writer=JsonlWriter(output_folder=str(writer_scratch)),
            shared_context={"shared": answers},
            metadata_key="rollout_results",
        )
        runner.run(documents, rank=0, world_size=1)
        runtime_seconds = time.perf_counter() - started
    finally:
        _terminate_server(server)
        shutil.rmtree(writer_scratch, ignore_errors=True)
    _write_contract(output_path, rows, answers, id_field)
    output_tokens = _output_tokens(answers, model)
    stats = _runner_stats(rows, 0, output_tokens, model_load_seconds, runtime_seconds)
    return stats


def run_nemo_curator_vlm(
    rows: List[Dict[str, Any]],
    model: str,
    image_base: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    image_field: str = "image_path",
    text_field: str = "caption",
    id_field: str = "id",
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """NeMo Curator pipeline with the framework's OpenAIClient and a local vLLM server."""
    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.models.client.openai_client import OpenAIClient
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.tasks import DocumentBatch

    class CaptionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
        name = "native_vlm_caption"

        def __init__(self, base_url: str, api_key: str = "unused"):
            super().__init__()
            self.client = OpenAIClient(base_url=base_url, api_key=api_key)

        def inputs(self) -> tuple[list[str], list[str]]:
            return ["data"], []

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], ["formatted_description"]

        def process(self, batch: DocumentBatch) -> DocumentBatch:
            df = batch.to_pandas().copy()
            descriptions: list[str] = []
            for row in df.to_dict(orient="records"):
                image_path = Path(str(row.get(image_field, "")))
                full_path = image_path if image_path.is_absolute() else image_base / image_path
                content: List[Dict[str, Any]] = []
                if full_path.exists():
                    content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path, image_base)}})
                content.append(
                    {"type": "text", "text": VLM_REFORMAT_TEMPLATE.format(caption=str(row.get(text_field, "")))}
                )
                result = self.client.query_model(
                    messages=[{"role": "user", "content": content}],
                    model=model,
                    generation_config={
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_new_tokens,
                        "n": 1,
                        "seed": None,
                        "stop": None,
                        "stream": False,
                        "top_k": None,
                        "extra_kwargs": {},
                    },
                )
                descriptions.append(result[0] if result else "")
            df["formatted_description"] = descriptions
            return DocumentBatch(dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf)

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = log_path.with_name(log_path.stem + ".vllm.log") if log_path else None
    server = _spawn_vllm_server(model, port=port, max_model_len=8192, log_path=server_log)
    model_load_seconds = 0.0
    runtime_seconds = 0.0
    answers: Dict[str, str] = {}
    try:
        started = time.perf_counter()
        _wait_for_http(f"{base_url}/v1/models", server, timeout=900.0)
        model_load_seconds = time.perf_counter() - started
        with tempfile.TemporaryDirectory(prefix="nemo_curator_vlm_") as scratch:
            input_path = Path(scratch) / "shard.jsonl"
            output_dir = Path(scratch) / "output"
            write_jsonl(input_path, rows)
            pipeline = Pipeline(name="native_vlm_caption", description="Native VLM caption via NeMo Curator")
            pipeline.add_stage(JsonlReader(file_paths=str(input_path)))
            pipeline.add_stage(CaptionStage(base_url=f"{base_url}/v1"))
            pipeline.add_stage(JsonlWriter(path=str(output_dir)))
            pipeline.run(executor=RayDataExecutor())
            for file_path in sorted(output_dir.glob("*.jsonl*")):
                if str(file_path).endswith(".gz"):
                    import gzip

                    opener = gzip.open
                else:
                    opener = open
                with opener(file_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            item = json.loads(line)
                            answers[str(item.get(id_field))] = item.get("formatted_description", "")
            runtime_seconds = time.perf_counter() - started
    finally:
        _terminate_server(server)
    _write_contract(output_path, rows, answers, id_field)
    output_tokens = _output_tokens(answers, model)
    stats = _runner_stats(rows, 0, output_tokens, model_load_seconds, runtime_seconds)
    return stats


VLM_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "sglang": run_sglang_vlm,
    "datatrove": run_datatrove_vlm,
    "nemo_curator": run_nemo_curator_vlm,
}


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    runner = VLM_RUNNERS[args.framework]
    log_path = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{args.framework}.log"
    stats = runner(
        rows=rows,
        model=args.model,
        image_base=Path(args.image_base_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
        output_path=Path(args.output_jsonl),
        log_path=log_path,
        image_field=args.image_field,
        text_field=args.text_field,
        id_field=args.id_field,
        gpu_count=args.gpu_count,
    )
    status = _runner_status(stats)
    status_path = Path(args.status_json)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
