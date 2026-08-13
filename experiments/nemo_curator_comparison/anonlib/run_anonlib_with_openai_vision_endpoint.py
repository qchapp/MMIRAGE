#!/usr/bin/env python3
"""Run unmodified AnonLib while forwarding SGLang Engine calls to an OpenAI VLM endpoint.

This benchmark-only adapter patches ``sglang.Engine`` so AnonLib's stock LLM
processor talks to an externally managed OpenAI-compatible VLM endpoint instead
of starting an in-process SGLang engine. It also registers the ``custom_pre``
and ``custom_post`` processor types used by the ChartQA recipe. Everything lives
in this file so the ``src/anonlib`` package stays untouched.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import importlib.util
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from transformers import AutoTokenizer

from anonlib.core.process.base import (
    BaseProcessor,
    BaseProcessorConfig,
    ProcessorRegistry,
    TokenCounts,
)
from anonlib.core.process.variables import OutputVar, VariableEnvironment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


def _image_to_data_url(image_ref: Any) -> str:
    if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://", "data:")):
        return image_ref
    path = Path(str(image_ref).removeprefix("file://"))
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


IMAGE_PLACEHOLDERS = ("<|vision_start|><|image_pad|><|vision_end|>", "<image>")


def _strip_image_tokens(text: str) -> str:
    for token in IMAGE_PLACEHOLDERS:
        text = text.replace(token, "")
    return text.strip()


class EndpointEngine:
    """Small stand-in for `sglang.Engine` used only by this benchmark process."""

    def __init__(self, **kwargs: Any) -> None:
        self.model_path = kwargs.get("model_path") or os.environ["ANONLIB_CHARTQA_MODEL_PATH"]
        self.base_url = os.environ["ANONLIB_CHARTQA_OPENAI_BASE_URL"].rstrip("/")
        self.timeout_seconds = int(os.environ.get("ANONLIB_CHARTQA_TIMEOUT_SECONDS", "900"))
        self.max_workers = max(1, int(os.environ.get("ANONLIB_CHARTQA_CONCURRENCY", "64")))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
            )
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return max(1, len(text.split()))

    def generate(
        self,
        prompt: Iterable[str] | str,
        sampling_params: dict[str, Any] | None = None,
        image_data: Iterable[Any] | Any | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if image_data is None:
            images = [None] * len(prompts)
        elif isinstance(image_data, list) and len(image_data) == len(prompts):
            images = image_data
        else:
            images = [image_data] * len(prompts)
        params = dict(sampling_params or {})
        if "max_new_tokens" in params and "max_tokens" not in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        json_schema = params.pop("json_schema", None)
        params = {key: value for key, value in params.items() if key in {"temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty", "seed"}}
        args = [(text, image, params, json_schema) for text, image in zip(prompts, images)]
        if self.max_workers == 1 or len(args) <= 1:
            return [self._generate_one(*item) for item in args]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._generate_one, *item) for item in args]
            return [future.result() for future in futures]

    def shutdown(self) -> None:
        return None

    def _generate_one(self, prompt: str, image_ref: Any, params: dict[str, Any], json_schema: str | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image_ref is not None:
            refs = image_ref if isinstance(image_ref, list) else [image_ref]
            for ref in refs:
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(ref)}})
        content.append({"type": "text", "text": _strip_image_tokens(prompt)})
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": content}],
            **params,
        }
        if json_schema:
            try:
                schema = json.loads(json_schema)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "vlm_result", "schema": schema, "strict": True},
                }
            except json.JSONDecodeError:
                pass
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": "Bearer unused"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible VLM HTTP {exc.code}: {error_body}") from exc
        choices = body.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = message.get("content") or choices[0].get("text", "") if choices else ""
        if not isinstance(text, str):
            text = str(text)
        usage = body.get("usage") or {}
        return {
            "text": text,
            "meta_info": {
                "prompt_tokens": int(usage.get("prompt_tokens") or self._count_tokens(prompt)),
                "completion_tokens": int(usage.get("completion_tokens") or self._count_tokens(text)),
            },
        }


@dataclass
class CustomProcessorConfig(BaseProcessorConfig):
    """Config for deterministic script-driven processors (``custom_pre``/``custom_post``).

    The referenced script is expected to expose ``function_name(row) -> dict``
    returning the new variable values computed from the row.
    """

    script_path: str = ""
    function_name: str = ""
    max_workers: int = 4
    start_method: str = "spawn"
    timeout_ms: int = 5000
    max_timeouts: int = 10
    max_errors: int = 10
    fallback_value: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomOutputVar(OutputVar):
    """Output variable produced by a custom script processor."""

    type: str = ""

    def is_computable(self, vars: Iterable[Any]) -> bool:
        return True


@dataclass
class CustomPreProcessorConfig(CustomProcessorConfig):
    type: Literal["custom_pre"] = "custom_pre"


@dataclass
class CustomPostProcessorConfig(CustomProcessorConfig):
    type: Literal["custom_post"] = "custom_post"


@dataclass
class CustomPreOutputVar(CustomOutputVar):
    type: Literal["custom_pre"] = "custom_pre"


@dataclass
class CustomPostOutputVar(CustomOutputVar):
    type: Literal["custom_post"] = "custom_post"


class CustomProcessor(BaseProcessor[CustomOutputVar]):
    """Processor that runs a deterministic Python function from a user script.

    The function receives the full row (as a plain dict) and returns a dict of
    new variable values. Timeouts and exceptions fall back to the configured
    ``fallback_value``.
    """

    def __init__(self, config: CustomProcessorConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._func = self._load_function(config.script_path, config.function_name)
        self._executor = ThreadPoolExecutor(max_workers=max(1, config.max_workers))
        self._timeouts = 0
        self._errors = 0

    @staticmethod
    def _load_function(script_path: str, function_name: str) -> Any:
        path = Path(script_path).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load custom processor script {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name, None)
        if function is None:
            raise AttributeError(f"Function {function_name!r} not found in {path}")
        return function

    def batch_process_sample(
        self, batch: list[VariableEnvironment], output_var: CustomOutputVar
    ) -> list[VariableEnvironment]:
        return [self._process_one(env, output_var) for env in batch]

    def _process_one(
        self, env: VariableEnvironment, output_var: CustomOutputVar
    ) -> VariableEnvironment:
        row = dict(env.to_dict())
        try:
            future = self._executor.submit(self._func, row)
            result = future.result(timeout=self.config.timeout_ms / 1000.0)
        except concurrent.futures.TimeoutError:
            self._timeouts += 1
            result = dict(self.config.fallback_value or {})
        except Exception:
            self._errors += 1
            result = dict(self.config.fallback_value or {})
        if not isinstance(result, dict):
            result = {output_var.name: result}
        return env.with_variable(output_var.name, result)

    def get_token_counts(self) -> TokenCounts:
        return TokenCounts(input_tokens=0, output_tokens=0)

    def get_load_time(self) -> float:
        return 0.0

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


def patch_sglang_engine() -> None:
    import sglang

    sglang.Engine = EndpointEngine
    for name, config_cls, output_cls in (
        ("custom_pre", CustomPreProcessorConfig, CustomPreOutputVar),
        ("custom_post", CustomPostProcessorConfig, CustomPostOutputVar),
    ):
        ProcessorRegistry.register_types(name, config_cls, output_cls)
        ProcessorRegistry.register(name, config_cls, output_cls)(CustomProcessor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_sglang_engine()
    from anonlib.shard_process import main as shard_main

    os.environ["ANONLIB_COLLECT_STATS"] = "1"
    started = time.perf_counter()
    old_argv = sys.argv
    returncode = 0
    try:
        sys.argv = ["anonlib.shard_process", "--config", str(Path(args.config).resolve())]
        try:
            shard_main()
        except SystemExit as exc:
            returncode = int(exc.code or 0)
    except Exception:
        returncode = 1
        raise
    finally:
        sys.argv = old_argv
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(
                {"returncode": returncode, "full_end_to_end_wall_seconds": round(time.perf_counter() - started, 6)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    sys.exit(returncode)


if __name__ == "__main__":
    main()
