"""Framework-native text-generation backends for native competitor baselines.

Each ``run_<framework>(...)`` function consumes the same list of workload rows
and writes the same contract JSONL (one line per row, in input order) with the
required fields filled in. Statistics are returned in a dict matching the
MMIRAGE shard status schema so existing aggregators can consume them.

The DataTrove, Distilabel, Ray Data LLM, and raw SGLang paths execute the
framework's own inference machinery (a framework-managed vLLM/SGLang engine per
shard worker). The NeMo Curator path runs a genuine ``nemo_curator.pipeline``
with the framework's ``OpenAIClient`` against a locally spawned vLLM server.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from experiments._shared.io import read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REWRITE_PROMPT_TEMPLATE = (
    "You are helping construct a public text dataset.\n\n"
    "Rewrite the following user request into a helpful assistant response of 4 to 6 sentences.\n"
    "Preserve the user's intent.\n\n"
    "User request:\n"
    "{prompt_text}"
)


def build_prompt(prompt_text: str, prompt_style: str = "rewrite") -> str:
    if prompt_style == "raw":
        return prompt_text
    return REWRITE_PROMPT_TEMPLATE.format(prompt_text=prompt_text)


def _ordered_contract_rows(rows: list[dict[str, Any]], answers: dict[str, str], id_field: str = "stable_id") -> list[dict[str, Any]]:
    """Emit contract rows in input order; the id column is ``id_field``."""
    return [
        {
            id_field: row[id_field],
            "source_index": row["source_index"],
            "prompt_sha256": row.get("prompt_sha256"),
            "prompt_text": row["prompt_text"],
            "answer": answers.get(str(row[id_field]), ""),
        }
        for row in rows
    ]


def _runner_stats(rows: list[dict[str, Any]], input_tokens: int, output_tokens: int, model_load_seconds: float, runtime_seconds: float) -> dict[str, Any]:
    """Build the MMIRAGE-compatible shard status ``stats`` dict."""
    return {
        "rows_processed": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_load_seconds": round(model_load_seconds, 6),
        "runtime_seconds": round(runtime_seconds, 6),
        "inference_runtime_seconds": round(max(0.0, runtime_seconds - model_load_seconds), 6),
        "gpu_util_mean": None,
        "gpu_util_samples": 0,
    }


def _chunked(sequence: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def _tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def run_raw_sglang(
    rows: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_new_tokens: int,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    prompt_style: str = "rewrite",
    id_field: str = "stable_id",
) -> dict[str, Any]:
    """In-process SGLang engine, one engine per shard worker."""
    from sglang import Engine

    prompts = [build_prompt(row["prompt_text"], prompt_style) for row in rows]
    started = time.perf_counter()
    engine = Engine(
        model_path=model,
        tokenizer_path=model,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.85,
    )
    model_load_seconds = time.perf_counter() - started
    tokenizer = _tokenizer(model)
    answers: dict[str, str] = {}
    output_tokens = 0
    infer_started = time.perf_counter()
    for chunk, chunk_prompts in zip(_chunked(rows, max(1, concurrency)), _chunked(prompts, max(1, concurrency))):
        outs = engine.generate(chunk_prompts, {"temperature": temperature, "max_new_tokens": max_new_tokens, "top_p": 1.0})
        for row, out in zip(chunk, outs):
            text = out.text if hasattr(out, "text") else out["text"]
            meta = out.meta_info if hasattr(out, "meta_info") else out["meta_info"]
            completion_tokens = int(meta.get("completion_tokens", 0) or 0)
            if not completion_tokens:
                completion_tokens = len(tokenizer.encode(text))
            answers[str(row[id_field])] = text
            output_tokens += completion_tokens
    inference_runtime = time.perf_counter() - infer_started
    engine.shutdown()
    output_rows = _ordered_contract_rows(rows, answers, id_field=id_field)
    input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
    stats = _runner_stats(rows, input_tokens, output_tokens, model_load_seconds, inference_runtime + model_load_seconds)
    _write_output(output_path, output_rows)
    return stats


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_vllm_server(model: str, port: int, max_model_len: int, log_path: Optional[Path]) -> subprocess.Popen:
    """Launch ``vllm serve`` (vLLM 0.23-compatible flags) in its own process group.

    Returns the Popen handle; the caller must terminate it. The child inherits
    ``SETUPTOOLS_USE_DISTUTILS=local`` so DataTrove/NeMo work on Python 3.12.
    """
    vllm_bin = shutil.which("vllm") or str(Path(sys.prefix) / "bin" / "vllm")
    cmd = [
        vllm_bin,
        "serve",
        model,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--trust-remote-code",
        "--disable-log-stats",
    ]
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "local"
    env.setdefault("USE_TF", "0")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    if log_path is not None:
        log_fh = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env, start_new_session=True)
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
    return proc


def _wait_for_http(url: str, proc: subprocess.Popen, timeout: float = 900.0) -> None:
    """Block until ``url`` answers 200 or ``proc`` exits / the timeout elapses."""
    deadline = time.monotonic() + timeout
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server process exited early (code={proc.returncode}); last_error={last_error}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except OSError as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server did not become ready within {timeout}s; last_error={last_error}")


def _terminate_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_datatrove(
    rows: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_new_tokens: int,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    prompt_style: str = "rewrite",
    id_field: str = "stable_id",
) -> dict[str, Any]:
    """DataTrove InferenceRunner against a self-managed vLLM server.

    DataTrove's bundled ``VLLMServer`` launches ``vllm serve`` with flags that
    were removed in vLLM 0.23, so we spawn our own server (with current flags)
    and point ``InferenceRunner`` at it via ``server_type="endpoint"``.
    """
    from datatrove.data import Document
    from datatrove.pipeline.inference.run_inference import (
        InferenceConfig,
        InferenceRunner,
    )
    from datatrove.pipeline.inference.types import InferenceResult
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    prompts = [build_prompt(row["prompt_text"], prompt_style) for row in rows]
    documents = [Document(text=prompt, id=str(row[id_field]), metadata=dict(row)) for row, prompt in zip(rows, prompts)]

    shared: dict[str, str] = {}

    async def rollout(document: Document, generate: Callable, answers: dict[str, str], **kwargs: Any):
        payload = {"messages": [{"role": "user", "content": document.text}]}
        result = await generate(payload)
        answers[document.id] = result.text
        return InferenceResult(text=result.text, finish_reason=result.finish_reason, usage=result.usage)

    tokenizer = _tokenizer(model)
    input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
    port = _pick_free_port()
    server_log = log_path.with_name(log_path.stem + ".vllm.log") if log_path else None
    proc = _spawn_vllm_server(model, port=port, max_model_len=8192, log_path=server_log)
    model_load_seconds: float = 0.0
    runtime: float = 0.0
    writer_scratch = Path(tempfile.mkdtemp(prefix="datatrove_runner_out_"))
    try:
        started = time.perf_counter()
        _wait_for_http(f"http://127.0.0.1:{port}/v1/models", proc, timeout=900.0)
        model_load_seconds = time.perf_counter() - started
        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path=model,
            endpoint_url=f"http://127.0.0.1:{port}",
            use_chat=True,
            tp=1,
            max_concurrent_generations=max(1, concurrency),
            default_generation_params={"temperature": temperature, "max_tokens": max_new_tokens},
        )
        runner = InferenceRunner(
            rollout_fn=rollout,
            config=config,
            output_writer=JsonlWriter(output_folder=str(writer_scratch)),
            shared_context={"answers": shared},
            metadata_key="rollout_results",
        )
        runner.run(documents, rank=0, world_size=1)
        runtime = time.perf_counter() - started
    finally:
        _terminate_server(proc)
        shutil.rmtree(writer_scratch, ignore_errors=True)
    output_rows = _ordered_contract_rows(rows, shared, id_field=id_field)
    output_tokens = sum(len(tokenizer.encode(answer)) for answer in shared.values())
    stats = _runner_stats(rows, input_tokens, output_tokens, model_load_seconds, runtime)
    _write_output(output_path, output_rows)
    return stats


def run_distilabel(
    rows: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_new_tokens: int,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    prompt_style: str = "rewrite",
    id_field: str = "stable_id",
) -> dict[str, Any]:
    """Distilabel pipeline with a native local vLLM-backed LLM task."""
    from datasets import Dataset
    from distilabel.llms import vLLM
    from distilabel.pipeline import Pipeline
    from distilabel.steps.tasks import TextGeneration

    records = []
    for row in rows:
        records.append(
            {
                "prompt": build_prompt(row["prompt_text"], prompt_style),
                id_field: row[id_field],
                "source_index": row["source_index"],
                "prompt_sha256": row.get("prompt_sha256"),
                "prompt_text": row["prompt_text"],
            }
        )
    dataset = Dataset.from_list(records)
    tokenizer = _tokenizer(model)
    input_tokens = sum(len(tokenizer.encode(record["prompt"])) for record in records)
    started = time.perf_counter()
    with Pipeline() as pipeline:
        step = TextGeneration(
            llm=vLLM(model=model, trust_remote_code=True),
            columns=["prompt"],
            output_mappings={"generation": "answer"},
        )
    distiset = pipeline.run(
        dataset=dataset,
        parameters={
            step.name: {
                "llm": {"generation_kwargs": {"temperature": temperature, "max_new_tokens": max_new_tokens}},
                "input_batch_size": max(1, concurrency),
            }
        },
    )
    split_name = next(iter(distiset.keys()))
    output_rows = []
    for item in distiset[split_name].to_list():
        output_rows.append(
            {
                id_field: item[id_field],
                "source_index": item["source_index"],
                "prompt_sha256": item.get("prompt_sha256"),
                "prompt_text": item["prompt_text"],
                "answer": item.get("answer", "") or "",
            }
        )
    output_rows.sort(key=lambda item: int(item["source_index"]))
    runtime = time.perf_counter() - started
    output_tokens = sum(len(tokenizer.encode(item["answer"])) for item in output_rows)
    stats = _runner_stats(rows, input_tokens, output_tokens, 0.0, runtime)
    _write_output(output_path, output_rows)
    return stats


def run_ray_data_llm(
    rows: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_new_tokens: int,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    prompt_style: str = "rewrite",
    id_field: str = "stable_id",
) -> dict[str, Any]:
    """Ray Data LLM native processor with a vLLM engine."""
    import ray
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig

    ray.init(ignore_reinit_error=True)
    tokenizer = _tokenizer(model)
    items = []
    for row in rows:
        items.append(
            {
                "prompt": build_prompt(row["prompt_text"], prompt_style),
                id_field: row[id_field],
                "source_index": row["source_index"],
                "prompt_sha256": row.get("prompt_sha256"),
                "prompt_text": row["prompt_text"],
                "sampling_params": {"temperature": temperature, "max_tokens": max_new_tokens},
            }
        )
    input_tokens = sum(len(tokenizer.encode(item["prompt"])) for item in items)
    dataset = ray.data.from_items(items)
    config = vLLMEngineProcessorConfig(
        model_source=model,
        task_type="generate",
        concurrency=1,
        batch_size=max(1, concurrency),
        engine_kwargs={"trust_remote_code": True, "gpu_memory_utilization": 0.85},
        apply_chat_template=False,
    )
    processor = build_processor(config)
    started = time.perf_counter()
    output_dataset = processor(dataset)
    result_rows = output_dataset.take_all()
    runtime = time.perf_counter() - started
    output_rows = []
    output_tokens = 0
    for item in result_rows:
        answer = item.get("generated_text", "") or ""
        output_rows.append(
            {
                id_field: item[id_field],
                "source_index": item["source_index"],
                "prompt_sha256": item.get("prompt_sha256"),
                "prompt_text": item["prompt_text"],
                "answer": answer,
            }
        )
        output_tokens += int(item.get("num_generated_tokens") or len(tokenizer.encode(answer)))
    output_rows.sort(key=lambda item: int(item["source_index"]))
    ray.shutdown()
    stats = _runner_stats(rows, input_tokens, output_tokens, 0.0, runtime)
    _write_output(output_path, output_rows)
    return stats


def run_nemo_curator(
    rows: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_new_tokens: int,
    concurrency: int,
    output_path: Path,
    log_path: Optional[Path] = None,
    prompt_style: str = "rewrite",
    id_field: str = "stable_id",
) -> dict[str, Any]:
    """NeMo Curator pipeline with the framework's OpenAIClient and a local vLLM server."""

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.models.client.openai_client import OpenAIClient
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.tasks import DocumentBatch

    class RewriteStage(ProcessingStage[DocumentBatch, DocumentBatch]):
        def __init__(self, base_url: str, api_key: str = "unused"):
            super().__init__()
            self.client = OpenAIClient(base_url=base_url, api_key=api_key)

        @property
        def name(self) -> str:
            return "native_rewrite_llm"

        def inputs(self) -> tuple[list[str], list[str]]:
            return ["data"], []

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], ["answer"]

        def process(self, batch: DocumentBatch) -> DocumentBatch:
            df = batch.to_pandas().copy()
            answers: list[str] = []
            for row in df.to_dict(orient="records"):
                prompt = build_prompt(str(row.get("prompt_text", "")), prompt_style)
                result = self.client.query_model(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    generation_config={
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "n": 1,
                        "seed": None,
                        "stop": None,
                        "stream": False,
                        "top_p": 1.0,
                        "top_k": None,
                        "extra_kwargs": {},
                    },
                )
                answers.append(result[0] if result else "")
            df["answer"] = answers
            return DocumentBatch(dataset_name=batch.dataset_name, data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf)

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    server_log = log_path.with_name(log_path.stem + ".vllm.log") if log_path else None
    server = _spawn_vllm_server(model, port=port, max_model_len=8192, log_path=server_log)
    model_load_seconds = 0.0
    try:
        _wait_for_http(f"{base_url}/v1/models", server, timeout=900.0)
        model_load_seconds = time.perf_counter() - started
        with tempfile.TemporaryDirectory(prefix="nemo_curator_") as scratch:
            input_path = Path(scratch) / "shard.jsonl"
            output_dir = Path(scratch) / "output"
            _write_output(input_path, rows)
            pipeline = Pipeline(name="native_rewrite_ultrachat", description="Native text rewrite via NeMo Curator")
            pipeline.add_stage(JsonlReader(file_paths=str(input_path)))
            pipeline.add_stage(RewriteStage(base_url=base_url))
            pipeline.add_stage(JsonlWriter(path=str(output_dir)))
            pipeline.run(executor=RayDataExecutor())
            result_rows = []
            for file_path in sorted(output_dir.glob("*.jsonl*")):
                if str(file_path).endswith(".gz"):
                    import gzip

                    opener = gzip.open
                else:
                    opener = open
                with opener(file_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            result_rows.append(json.loads(line))
            answers = {str(item.get(id_field)): item.get("answer", "") for item in result_rows}
            output_rows = _ordered_contract_rows(rows, answers, id_field=id_field)
            runtime = time.perf_counter() - started
            tokenizer = _tokenizer(model)
            input_tokens = sum(len(tokenizer.encode(build_prompt(row["prompt_text"], prompt_style))) for row in rows)
            output_tokens = sum(len(tokenizer.encode(answer)) for answer in answers.values())
            stats = _runner_stats(rows, input_tokens, output_tokens, model_load_seconds, runtime)
            _write_output(output_path, output_rows)
            return stats
    finally:
        _terminate_server(server)


FRAMEWORK_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "raw_sglang": run_raw_sglang,
    "datatrove": run_datatrove,
    "distilabel": run_distilabel,
    "ray_data_llm": run_ray_data_llm,
    "nemo_curator": run_nemo_curator,
}


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def validate_contract(
    input_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    required_fields: list[str],
    stable_id_field: str = "stable_id",
    prompt_sha256_field: str = "prompt_sha256",
    source_index_field: str = "source_index",
) -> dict[str, Any]:
    """Check a native competitor output against the experiment output contract.

    Returns a dict with ``valid`` plus per-check results. The checks mirror the
    ``output_contract.validation`` lists in the native competitor configs.
    """
    input_ids = [str(row[stable_id_field]) for row in input_rows]
    output_ids = [str(row[stable_id_field]) for row in output_rows]
    input_indexes = [int(row[source_index_field]) for row in input_rows]
    output_indexes = [int(row[source_index_field]) for row in output_rows]
    input_shas = {str(row[stable_id_field]): row.get(prompt_sha256_field) for row in input_rows}

    schema_valid = 0
    schema_invalid_rows: list[str] = []
    sha_mismatches: list[str] = []
    for output_row in output_rows:
        missing = [field for field in required_fields if field not in output_row or output_row.get(field) in (None, "")]
        if missing:
            schema_invalid_rows.append(f"{output_row.get(stable_id_field)}:missing={missing}")
        else:
            schema_valid += 1
        expected_sha = input_shas.get(str(output_row.get(stable_id_field)))
        if expected_sha is not None and output_row.get(prompt_sha256_field) != expected_sha:
            sha_mismatches.append(str(output_row.get(stable_id_field)))

    missing_ids = sorted(set(input_ids) - set(output_ids))
    unexpected_ids = sorted(set(output_ids) - set(input_ids))
    duplicate_ids = sorted({item for item in output_ids if output_ids.count(item) > 1})
    order_mismatch = output_indexes != input_indexes

    checks = {
        "processed_rows_equals_input_rows": len(output_rows) == len(input_rows),
        "stable_id_set_matches_input": not missing_ids and not unexpected_ids,
        "no_duplicate_stable_ids": not duplicate_ids,
        "prompt_sha256_preserved": not sha_mismatches,
        "schema_valid_for_every_row": not schema_invalid_rows,
        "row_order_matches_input": not order_mismatch,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "missing_stable_ids": missing_ids,
        "unexpected_stable_ids": unexpected_ids,
        "duplicate_stable_ids": duplicate_ids,
        "schema_invalid_rows": schema_invalid_rows,
        "prompt_sha256_mismatches": sha_mismatches,
        "row_order_matches_input": not order_mismatch,
    }
