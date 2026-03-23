import json

import pytest

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult
from mmirage.core.process.batch.registry import BatchAdapterFactory, BatchAdapterRegistry


def test_openai_build_request_matches_expected_structure():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    config = OpenAIBatchConfig(model="gpt-4.1-mini")
    adapter = OpenAIBatchAdapter()
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }

    request = adapter.build_request(custom_id="row-001", payload=payload, config=config)

    assert request == {
        "custom_id": "row-001",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    }


def test_openai_estimate_request_bytes_matches_utf8_json_size():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    adapter = OpenAIBatchAdapter()
    request = {
        "custom_id": "accented",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"message": "caf\u00e9"},
    }

    expected = len(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    assert adapter.estimate_request_bytes(request) == expected


def test_openai_submit_chunk_uses_mocked_openai_client(monkeypatch):
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    captured = {}

    class FakeFiles:
        def create(self, *, file, purpose):
            file_name, file_obj = file
            assert file_name == "batch_chunk-chunk-01.jsonl"
            assert purpose == "batch"
            file_content = file_obj.read().decode("utf-8")
            captured["jsonl"] = file_content

            class _FileResp:
                id = "file_123"

            return _FileResp()

    class FakeBatches:
        def create(self, **kwargs):
            captured["batch_create_kwargs"] = kwargs

            class _BatchResp:
                id = "batch_123"
                status = "validating"
                endpoint = kwargs["endpoint"]

            return _BatchResp()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.files = FakeFiles()
            self.batches = FakeBatches()

    monkeypatch.setattr(
        "mmirage.core.process.batch.openai_adapter.OpenAI",
        FakeClient,
    )

    config = OpenAIBatchConfig(
        model="gpt-4.1-mini",
        completion_window="24h",
        batch_endpoint="/v1/chat/completions",
        metadata={"pipeline": "unit"},
        credentials={"api_key": "test-key"},
    )
    adapter = OpenAIBatchAdapter()
    requests = [
        adapter.build_request(
            custom_id="r1",
            payload={"messages": [{"role": "user", "content": "Hi"}]},
            config=config,
        )
    ]

    raw_result = adapter.submit_chunk(chunk_id="chunk-01", requests=requests, config=config)

    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["batch_create_kwargs"] == {
        "input_file_id": "file_123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"pipeline": "unit", "chunk_id": "chunk-01"},
    }
    assert raw_result["id"] == "batch_123"
    assert raw_result["status"] == "validating"
    assert raw_result["input_file_id"] == "file_123"

    jsonl_lines = [line for line in captured["jsonl"].split("\n") if line.strip()]
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0]) == requests[0]


def test_openai_parse_submission_result_normalizes_payload():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    adapter = OpenAIBatchAdapter()
    raw = {
        "id": "batch_123",
        "status": "in_progress",
        "input_file_id": "file_123",
    }

    result = adapter.parse_submission_result(raw_result=raw, request_count=4)

    assert isinstance(result, BatchSubmissionResult)
    assert result.provider_batch_id == "batch_123"
    assert result.status == "in_progress"
    assert result.submitted_request_count == 4
    assert result.raw_response == raw


def test_factory_resolves_openai_adapter_from_registry():
    from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter

    BatchAdapterRegistry.clear()
    config = OpenAIBatchConfig(model="gpt-4.1-mini", credentials={"api_key": "key"})

    adapter = BatchAdapterFactory.from_config(config)

    assert isinstance(adapter, OpenAIBatchAdapter)
