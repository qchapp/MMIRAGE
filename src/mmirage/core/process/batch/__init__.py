"""Provider-agnostic batch processing contracts and registry."""

from mmirage.core.process.batch.adapter import BatchSubmissionAdapter, BatchSubmissionResult
from mmirage.core.process.batch.registry import BatchAdapterFactory, BatchAdapterRegistry

__all__ = [
    "BatchSubmissionAdapter",
    "BatchSubmissionResult",
    "BatchAdapterFactory",
    "BatchAdapterRegistry",
]
