import json
from types import SimpleNamespace

import pytest
from mmirage.config.openai_batch import OpenAIBatchConfig


def test_collect_and_merge_reconstructs_rows_deterministically(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import _read_metadata_records, collect_and_merge

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
                        "generated_text": '{"question":"q2","answer":"a2"}',
                    },
                    {
                        "custom_id": "c2",
                        "generated_text": '{"question":"q0","answer":"a0"}',
                    },
                ]
            return [
                {
                    "custom_id": "c3",
                    "generated_text": '{"question":"q1","answer":"a1"}',
                }
            ]

    fake_adapter = FakeAdapter()
    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: fake_adapter,
    )

    provider_configs = {"openai": OpenAIBatchConfig(credentials={"api_key": "k"})}
    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
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
    from mmirage.core.process.batch.collector import _read_metadata_records, collect_and_merge

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
        records = _read_metadata_records(str(metadata_path))
        collect_and_merge(
            records=records,
            provider_configs={},
            output_path=str(tmp_path / "out.jsonl"),
        )
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "No provider config" in str(e)


def test_collect_and_merge_outputs_caption_for_plain_text_content(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import _read_metadata_records, collect_and_merge

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
                    "generated_text": "A black cat sitting on a sofa.",
                }
            ]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: FakeAdapter(),
    )

    records = _read_metadata_records(str(metadata_path))
    rows = collect_and_merge(
        records=records,
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


def test_collector_main_uses_config_and_records(tmp_path, monkeypatch):
    from mmirage.core.process.batch import collector

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "provider_batch_id": "batch_main",
                "custom_id_to_source_index": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(processors=[SimpleNamespace(batch_provider={"provider": "openai"})])
    captured = {}

    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    def _fake_collect_and_merge(records, provider_configs, output_path_arg):
        captured["records"] = records
        captured["provider_configs"] = provider_configs
        captured["output_path"] = output_path_arg
        return [{"source_index": 0, "custom_id": "c1", "caption": "ok"}]

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.collect_and_merge",
        _fake_collect_and_merge,
    )

    rc = collector.main(
        [
            "--metadata-path",
            str(metadata_path),
            "--output-path",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert rc == 0
    assert len(captured["records"]) == 1
    assert captured["records"][0]["provider"] == "openai"
    assert "openai" in captured["provider_configs"]
    assert captured["output_path"] == str(output_path)


def test_collector_main_raises_when_metadata_provider_missing_in_config(tmp_path, monkeypatch):
    from mmirage.core.process.batch import collector

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "provider": "mistral",
                "provider_batch_id": "batch_mistral",
                "custom_id_to_source_index": {"m1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    # Config intentionally only defines openai, not mistral.
    cfg = SimpleNamespace(processors=[SimpleNamespace(batch_provider={"provider": "openai"})])
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    with pytest.raises(ValueError, match="missing from YAML batch_provider config"):
        collector.main(
            [
                "--metadata-path",
                str(metadata_path),
                "--output-path",
                str(output_path),
                "--config",
                str(config_path),
            ]
        )


def test_collect_and_merge_routes_multiple_providers(tmp_path, monkeypatch):
    from mmirage.core.process.batch.collector import _read_metadata_records, collect_and_merge
    from mmirage.core.process.batch.provider_resolution import resolve_provider_configs

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "provider_batch_id": "batch_openai",
                        "custom_id_to_source_index": {"o1": 1},
                    }
                ),
                json.dumps(
                    {
                        "provider": "unit",
                        "provider_batch_id": "batch_unit",
                        "custom_id_to_source_index": {"u1": 0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(
                batch_provider={"provider": "openai", "credentials": {"api_key": "k"}}
            ),
            SimpleNamespace(batch_provider={"provider": "unit"}),
        ]
    )

    records = _read_metadata_records(str(metadata_path))
    provider_configs = resolve_provider_configs(records, cfg)

    class OpenAIAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            return [{"custom_id": "o1", "generated_text": "openai"}]

    class UnitAdapter:
        def __init__(self):
            self.calls = []

        def retrieve_results(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            return [{"custom_id": "u1", "generated_text": "unit"}]

    adapters = {
        "openai": OpenAIAdapter(),
        "unit": UnitAdapter(),
    }

    monkeypatch.setattr(
        "mmirage.core.process.batch.collector.BatchAdapterFactory.from_config",
        lambda config: adapters[config.provider],
    )

    output_path = tmp_path / "merged.jsonl"
    rows = collect_and_merge(
        records=records,
        provider_configs=provider_configs,
        output_path=str(output_path),
    )

    assert [row["custom_id"] for row in rows] == ["u1", "o1"]
    assert [row["caption"] for row in rows] == ["unit", "openai"]
    assert ("batch_openai", "openai") in adapters["openai"].calls
    assert ("batch_unit", "unit") in adapters["unit"].calls


def test_collector_main_raises_for_invalid_batch_provider_config(tmp_path, monkeypatch):
    from mmirage.core.process.batch import collector

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
    output_path = tmp_path / "out.jsonl"
    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("processors: []\n", encoding="utf-8")

    cfg = SimpleNamespace(
        processors=[
            SimpleNamespace(batch_provider={"provider": "openai", "batch_endpoint": "v1"})
        ]
    )
    monkeypatch.setattr("mmirage.config.utils.load_mmirage_config", lambda path: cfg)

    with pytest.raises(ValueError, match="batch_endpoint must start with '/'"):
        collector.main(
            [
                "--metadata-path",
                str(metadata_path),
                "--output-path",
                str(output_path),
                "--config",
                str(config_path),
            ]
        )
