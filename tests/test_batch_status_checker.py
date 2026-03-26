from io import StringIO

from mmirage.config.openai_batch import OpenAIBatchConfig
from mmirage.core.process.batch.adapter import BatchSubmissionResult


def test_extract_unique_provider_batches_handles_malformed_and_duplicates(tmp_path):
    from mmirage.core.process.batch.status_checker import extract_unique_provider_batches

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                "not-json",
                '{"provider":"openai"}',
                '{"provider":"openai","provider_batch_id":"batch_2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    pairs = extract_unique_provider_batches(str(metadata_path))

    assert pairs == [("openai", "batch_1"), ("openai", "batch_2")]


def test_run_status_checker_prints_summary_with_factory_dispatch(tmp_path, monkeypatch):
    from mmirage.core.process.batch.status_checker import run_status_checker

    metadata_path = tmp_path / "receipts.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                '{"provider":"openai","provider_batch_id":"batch_1"}',
                '{"provider":"openai","provider_batch_id":"batch_2"}',
                '{"provider":"openai","provider_batch_id":"batch_1"}',
            ]
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def check_batch_status(self, provider_batch_id, config):
            self.calls.append((provider_batch_id, config.provider))
            status = "completed" if provider_batch_id == "batch_1" else "in_progress"
            return BatchSubmissionResult(
                provider_batch_id=provider_batch_id,
                status=status,
                submitted_request_count=0,
                raw_response={"id": provider_batch_id, "status": status},
            )

    fake_adapter = FakeAdapter()

    monkeypatch.setattr(
        "mmirage.core.process.batch.status_checker.BatchAdapterFactory.from_config",
        lambda config: fake_adapter,
    )

    output = StringIO()
    config_map = {
        "openai": OpenAIBatchConfig(credentials={"api_key": "k"}),
    }

    results = run_status_checker(
        metadata_output_path=str(metadata_path),
        provider_configs=config_map,
        output=output,
    )

    assert [(r.provider_batch_id, r.status) for r in results] == [
        ("batch_1", "completed"),
        ("batch_2", "in_progress"),
    ]
    assert fake_adapter.calls == [
        ("batch_1", "openai"),
        ("batch_2", "openai"),
    ]

    printed = output.getvalue()
    assert "Batch batch_1 (openai): completed" in printed
    assert "Batch batch_2 (openai): in_progress" in printed
