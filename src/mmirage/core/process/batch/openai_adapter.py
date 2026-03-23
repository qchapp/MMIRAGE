"""Concrete OpenAI implementation of batch submission contracts."""

import io
import json
from typing import Any, Mapping, Sequence

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
        body.setdefault("model", openai_config.model)

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

        client_kwargs = {"api_key": openai_config.credentials.get("api_key", "")}
        if openai_config.base_url:
            client_kwargs["base_url"] = openai_config.base_url
        client = OpenAI(**client_kwargs)

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
    def _read_attr(obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key)
