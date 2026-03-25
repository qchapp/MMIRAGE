"""Provider-agnostic batch processing contracts and registry."""

from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult
from mmirage.core.process.batch.chunking import BatchRequestChunker, RequestChunk
from mmirage.core.process.batch.openai_adapter import OpenAIBatchAdapter
from mmirage.core.process.batch.orchestrator import BatchSubmissionOrchestrator
from mmirage.core.process.batch.registry import BatchAdapterFactory, BatchAdapterRegistry
from mmirage.config.openai_batch import OpenAIBatchConfig

__all__ = [
    "BatchSubmissionAdapter",
    "BatchSubmissionResult",
    "BatchRequestChunker",
    "RequestChunk",
    "BatchSubmissionOrchestrator",
    "OpenAIBatchAdapter",
    "OpenAIBatchConfig",
    "BatchAdapterFactory",
    "BatchAdapterRegistry",
]
