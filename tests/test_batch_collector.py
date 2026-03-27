import json

from mmirage.config.openai_batch import OpenAIBatchConfig


def test_collect_and_merge_reconstructs_rows_deterministically(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import collect_and_merge

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_1",
                        "custom_id_to_source_index": {"c1": 2, "c2": 0},
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_1",
                        "custom_id_to_source_index": {"c1": 2, "c2": 0},
                    }
                ),
                "malformed-line",
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_2",
                        "custom_id_to_source_index": {"c3": 1},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "merged.jsonl"

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            if provider_batch_id == "batch_1":
                return [
                    {
                        "custom_id": "c1",
                        "response": {
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": '{"question":"q2","answer":"a2"}'
                                        }
                                    }
                                ]
                            }
                        },
                    },
                    {
                        "custom_id": "c2",
                        "response": {
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": '{"question":"q0","answer":"a0"}'
                                        }
                                    }
                                ]
                            }
                        },
                    },
                ]
            return [
                {
                    "custom_id": "c3",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": '{"question":"q1","answer":"a1"}'
                                    }
                                }
                            ]
                        }
                    },
                }
            ]

    fake_adapter = FakeAdapter()
    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: fake_adapter,
    )

    provider_configs = {"openai": OpenAIBatchConfig(credentials={"api_key": "k"})}
    rows = collect_and_merge(
        metadata_output_path=str(metadata_path),
        provider_configs=provider_configs,
        output_path=str(output_path),
    )

    assert [r["source_index"] for r in rows] == [0, 1, 2]
    assert [r["custom_id"] for r in rows] == ["c2", "c3", "c1"]
    assert [r["conversations"][0]["content"] for r in rows] == ["q0", "q1", "q2"]
    assert [r["conversations"][1]["content"] for r in rows] == ["a0", "a1", "a2"]
    assert fake_adapter.calls == [("batch_1", "openai"), ("batch_2", "openai")]

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [r["source_index"] for r in written] == [0, 1, 2]
    assert [r["conversations"][0]["content"] for r in written] == ["q0", "q1", "q2"]


def test_collect_and_merge_raises_for_missing_provider_config(tmp_path):
    from mmirage.core.process.batch.collector import collect_and_merge

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_1",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        collect_and_merge(
            metadata_output_path=str(metadata_path),
            provider_configs={},
            output_path=str(tmp_path / "out.jsonl"),
        )
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "No provider config" in str(e)


def test_collect_and_merge_outputs_caption_for_plain_text_content(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import collect_and_merge

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_plain",
                "custom_id_to_source_index": {"img_1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "merged_plain.jsonl"

    class FakeAdapter:
        def retrieve_results(self, provider_batch_id, config):
            return [
                {
                    "custom_id": "img_1",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": "A black cat sitting on a sofa."
                                    }
                                }
                            ]
                        }
                    },
                }
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: FakeAdapter(),
    )

    rows = collect_and_merge(
        metadata_output_path=str(metadata_path),
        provider_configs={"openai": OpenAIBatchConfig(credentials={"api_key": "k"})},
        output_path=str(output_path),
    )

    assert rows == [
        {
            "source_index": 0,
            "custom_id": "img_1",
            "caption": "A black cat sitting on a sofa.",
        }
    ]
