"""Provider-agnostic batch processing contracts and registry."""

from anonlib.config.openai_batch import OpenAIBatchConfig
from anonlib.core.process.batch.adapter import (
    BatchSubmissionAdapter,
    BatchSubmissionResult,
)
from anonlib.core.process.batch.chunking import BatchRequestChunker, RequestChunk
from anonlib.core.process.batch.collector import collect_and_merge
from anonlib.core.process.batch.openai_adapter import OpenAIBatchAdapter
from anonlib.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from anonlib.core.process.batch.registry import (
    BatchAdapterFactory,
    BatchAdapterRegistry,
)
from anonlib.core.process.batch.status_checker import (
    extract_unique_provider_batches,
    run_status_checker,
)

__all__ = [
    "BatchSubmissionAdapter",
    "BatchSubmissionResult",
    "collect_and_merge",
    "BatchRequestChunker",
    "RequestChunk",
    "BatchSubmissionOrchestrator",
    "OpenAIBatchAdapter",
    "OpenAIBatchConfig",
    "BatchAdapterFactory",
    "BatchAdapterRegistry",
    "extract_unique_provider_batches",
    "run_status_checker",
]
