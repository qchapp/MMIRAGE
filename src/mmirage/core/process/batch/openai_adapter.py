"""Concrete OpenAI implementation of batch submission contracts."""

import io
import json
import os
from typing import Any, Dict, List, Mapping, Sequence

from openai import OpenAI

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult


class OpenAIBatchAdapter(BatchSubmissionAdapter):
    """Provider adapter for OpenAI Batch API."""

    required_credentials = ("api_key",)

    @property
    def adapter_name(self) -> str:
        return "openai-batch-adapter"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    def build_request(
        self,
        custom_id: str,
        payload: Mapping[str, Any],
        config: BatchProviderConfig,
    ) -> Mapping[str, Any]:
        openai_config = self._as_openai_config(config)
        body = dict(payload)
        expected_schema = body.pop("expected_schema", None)
        body.setdefault("model", openai_config.model)

        if isinstance(expected_schema, list) and all(isinstance(k, str) for k in expected_schema):
            properties = {key: {"type": "string"} for key in expected_schema}
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": expected_schema,
                        "additionalProperties": False,
                    },
                },
            }

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": openai_config.batch_endpoint,
            "body": body,
        }

    def estimate_request_bytes(self, request: Mapping[str, Any]) -> int:
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return len(serialized.encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Mapping[str, Any]],
        config: BatchProviderConfig,
    ) -> Mapping[str, Any]:
        openai_config = self._as_openai_config(config)

        client = self._create_client(openai_config)

        jsonl_lines = [
            json.dumps(req, ensure_ascii=False, separators=(",", ":")) for req in requests
        ]
        jsonl_payload = "\n".join(jsonl_lines).encode("utf-8")

        file_response = client.files.create(
            file=(f"batch_chunk-{chunk_id}.jsonl", io.BytesIO(jsonl_payload)),
            purpose="batch",
        )

        metadata = dict(openai_config.metadata)
        metadata["chunk_id"] = chunk_id

        batch_response = client.batches.create(
            input_file_id=self._read_attr(file_response, "id"),
            endpoint=openai_config.batch_endpoint,
            completion_window=openai_config.completion_window,
            metadata=metadata,
        )

        return {
            "id": self._read_attr(batch_response, "id"),
            "status": self._read_attr(batch_response, "status"),
            "endpoint": self._read_attr(batch_response, "endpoint"),
            "input_file_id": self._read_attr(file_response, "id"),
            "chunk_id": chunk_id,
        }

    def check_batch_status(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> BatchSubmissionResult:
        openai_config = self._as_openai_config(config)
        client = self._create_client(openai_config)

        retrieved = client.batches.retrieve(provider_batch_id)
        raw_result = {
            "id": self._read_attr(retrieved, "id"),
            "status": self._read_attr(retrieved, "status"),
        }
        return self.parse_submission_result(raw_result=raw_result, request_count=0)

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        openai_config = self._as_openai_config(config)
        client = self._create_client(openai_config)

        retrieved = client.batches.retrieve(provider_batch_id)
        status = str(self._read_attr(retrieved, "status") or "unknown")
        output_file_id = self._read_attr(retrieved, "output_file_id")

        if status != "completed" or not output_file_id:
            raise ValueError(
                f"Batch '{provider_batch_id}' is not completed or has no output file (status={status})."
            )

        content_response = client.files.content(output_file_id)
        jsonl_text = self._extract_content_text(content_response)

        rows: List[Dict[str, Any]] = []
        for line in jsonl_text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            rows.append(dict(json.loads(raw)))

        return rows

    def parse_submission_result(
        self,
        raw_result: Mapping[str, Any],
        request_count: int,
    ) -> BatchSubmissionResult:
        batch_id = str(raw_result.get("id") or raw_result.get("batch_id") or "")
        status = str(raw_result.get("status") or "unknown")

        return BatchSubmissionResult(
            provider_batch_id=batch_id,
            status=status,
            submitted_request_count=request_count,
            raw_response=dict(raw_result),
        )

    @staticmethod
    def _as_openai_config(config: BatchProviderConfig) -> OpenAIBatchConfig:
        if isinstance(config, OpenAIBatchConfig):
            return config
        raise TypeError("OpenAIBatchAdapter requires OpenAIBatchConfig")

    @staticmethod
    def _create_client(config: OpenAIBatchConfig) -> OpenAI:
        api_key = (config.credentials.get("api_key", "") or "").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Provide credentials.api_key or set OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        return OpenAI(**client_kwargs)

    @staticmethod
    def _extract_content_text(content_response: Any) -> str:
        text = getattr(content_response, "text", None)
        if isinstance(text, str):
            return text

        read = getattr(content_response, "read", None)
        if callable(read):
            data = read()
            if isinstance(data, bytes):
                return data.decode("utf-8")
            if isinstance(data, str):
                return data

        content = getattr(content_response, "content", None)
        if isinstance(content, bytes):
            return content.decode("utf-8")
        if isinstance(content, str):
            return content

        raise ValueError("Unable to parse OpenAI files.content response payload.")

    @staticmethod
    def _read_attr(obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key)
