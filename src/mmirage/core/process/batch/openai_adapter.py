"""Concrete OpenAI implementation of batch submission contracts."""

import base64
import copy
import io
import json
import mimetypes
import os
from typing import Any, Dict, List, Mapping, Sequence

from openai import AuthenticationError, OpenAI

from mmirage.config.batch_provider import BatchProviderConfig
from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult


class OpenAIBatchAdapter(BatchSubmissionAdapter):
    """Provider adapter for OpenAI Batch API."""

    required_credentials = ("api_key",)

    def build_request(
        self,
        custom_id: str,
        payload: Dict[str, Any],
        config: BatchProviderConfig,
    ) -> Mapping[str, Any]:
        openai_config = self._check_openai_config(config)
        body = copy.deepcopy(payload)
        expected_schema = body.get("expected_schema")
        if expected_schema is not None and (
            not isinstance(expected_schema, list)
            or not all(isinstance(key, str) for key in expected_schema)
        ):
            raise ValueError(
                "expected_schema must be a list of strings, "
                f"got {type(expected_schema).__name__}"
            )
        body.setdefault("model", openai_config.model)
        self._convert_local_images_to_data_uris(body)

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

    @staticmethod
    def _convert_local_images_to_data_uris(body: Dict[str, Any]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue

            content = message.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue

                image_url = part.get("image_url")
                if not isinstance(image_url, dict):
                    continue

                url = image_url.get("url")
                if not isinstance(url, str):
                    continue

                # Keep remote/data URLs untouched.
                if url.startswith("http://") or url.startswith("https://") or url.startswith("data:"):
                    continue

                if os.path.exists(url):
                    image_url["url"] = OpenAIBatchAdapter._local_file_to_data_uri(url)

    @staticmethod
    def _local_file_to_data_uri(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def estimate_request_bytes(self, request: Mapping[str, Any]) -> int:
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        return len(serialized.encode("utf-8"))

    def submit_chunk(
        self,
        chunk_id: str,
        requests: Sequence[Mapping[str, Any]],
        config: BatchProviderConfig,
    ) -> Dict[str, Any]:
        openai_config = self._check_openai_config(config)
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
        openai_config = self._check_openai_config(config)
        client = self._create_client(openai_config)

        retrieved = client.batches.retrieve(provider_batch_id)
        raw_result = {
            "id": self._read_attr(retrieved, "id"),
            "status": self._read_attr(retrieved, "status"),
        }
        return self.parse_submission_result(raw_result=raw_result)

    def retrieve_results(
        self,
        provider_batch_id: str,
        config: BatchProviderConfig,
    ) -> Sequence[Dict[str, Any]]:
        """Download completed OpenAI batch rows and normalize text into ``generated_text``.

        OpenAI batch outputs can surface the assistant payload in nested
        response bodies, so this method flattens the provider-specific shape
        before returning rows to the provider-agnostic collector.
        """
        openai_config = self._check_openai_config(config)
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
            row = dict(json.loads(raw))
            if "generated_text" not in row:
                generated_text = self._extract_generated_text(row)
                if generated_text:
                    row["generated_text"] = generated_text
            rows.append(row)

        return rows

    def parse_submission_result(
        self,
        raw_result: Mapping[str, Any],
    ) -> BatchSubmissionResult:
        batch_id = str(raw_result.get("id") or raw_result.get("batch_id") or "")
        status = raw_result.get("status", "unknown")

        return BatchSubmissionResult(
            provider_batch_id=batch_id,
            status=status,
            raw_response=raw_result,
        )

    @staticmethod
    def _check_openai_config(config: BatchProviderConfig) -> OpenAIBatchConfig:
        """Validate that `config` is an `OpenAIBatchConfig` and return it.

        Raises `TypeError` when the provided `config` is not an
        `OpenAIBatchConfig`.
        """
        if isinstance(config, OpenAIBatchConfig):
            return config
        raise TypeError("OpenAIBatchAdapter requires OpenAIBatchConfig")

    @staticmethod
    def _extract_generated_text(row: Mapping[str, Any]) -> str:
        response = row.get("response")
        if not isinstance(response, Mapping):
            return ""

        body = response.get("body")
        if not isinstance(body, Mapping):
            return ""

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            text = body.get("text")
            return text if isinstance(text, str) else ""

        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content

            text = first.get("text")
            if isinstance(text, str):
                return text

        text = body.get("text")
        if isinstance(text, str):
            return text

        return ""

    @staticmethod
    def _create_client(config: OpenAIBatchConfig) -> OpenAI:
        api_key = config.credentials.get("api_key", "").strip()
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Provide credentials.api_key or set OPENAI_API_KEY."
            )

        try:
            client_kwargs = {"api_key": api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            return OpenAI(**client_kwargs)
        except AuthenticationError as exc:
            raise ValueError(f"OpenAI authentication failed: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to create OpenAI client: {exc}") from exc

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
