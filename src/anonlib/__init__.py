"""AnonLib - declarative dataset processing with generative models.

A platform for processing datasets using generative models including
vision-language models (VLMs).
"""

from __future__ import annotations

__version__ = "0.1.4"

from anonlib.config.config import AnonLibConfig, ProcessingParams
from anonlib.config.loading import LoadingParams
from anonlib.config.utils import load_anonlib_config

__all__ = [
    "AnonLibConfig",
    "ProcessingParams",
    "LoadingParams",
    "load_anonlib_config",
    "__version__",
]
