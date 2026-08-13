"""Configuration loading utilities for AnonLib pipeline."""

import os
from typing import Any, Dict, List, TypeAlias, Union, cast

import yaml
from dacite import Config, from_dict

# Register built-in processors/loaders.
#
# We import configuration modules (lightweight) here so the registries know how
# to construct config/output-var objects from YAML without importing heavy
# processor implementations (e.g. torch/transformers).
import anonlib.core.loader.jsonl  # noqa: F401
import anonlib.core.loader.local_hf  # noqa: F401
import anonlib.core.process.processors.image_gen.config  # noqa: F401
import anonlib.core.process.processors.llm.config  # noqa: F401
from anonlib.config.batch_provider import BatchProviderConfig
from anonlib.config.config import AnonLibConfig
from anonlib.core.loader.base import BaseDataLoaderConfig, DataLoaderRegistry
from anonlib.core.process.base import BaseProcessorConfig, OutputVar, ProcessorRegistry
from anonlib.core.process.batch.provider_resolution import (
    resolve_single_provider_config,
)
from anonlib.core.process.processors.image_gen.config import ImageOutputMode

EnvValue: TypeAlias = Union[str, List["EnvValue"], Dict[str, "EnvValue"]]


def load_anonlib_config(config_path: str) -> AnonLibConfig:
    """Load AnonLib configuration from a YAML file.

    Supports environment variable expansion and dynamic processor/loader
    configuration based on registered types. See the configuration guide for
    complete YAML examples.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AnonLibConfig: Parsed and validated configuration object.
    """

    with open(config_path, "r") as f:
        cfg: EnvValue = yaml.safe_load(f) or {}

    def expand_env_vars(obj: EnvValue) -> EnvValue:
        if isinstance(obj, dict):
            return {key: expand_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj

    def image_output_mode_hook(value: Any) -> ImageOutputMode:
        if isinstance(value, ImageOutputMode):
            return value
        return ImageOutputMode(value)

    def processor_config_hook(data: Dict[str, Any]) -> BaseProcessorConfig:
        clz = ProcessorRegistry.get_config_cls(data["type"])
        return from_dict(clz, data, config=config)

    def loader_config_hook(data: Dict[str, Any]) -> BaseDataLoaderConfig:
        clz = DataLoaderRegistry.get_config_cls(data["type"])
        return from_dict(clz, data, config=config)

    def output_var_hook(data: Dict[str, Any]) -> OutputVar:
        clz = ProcessorRegistry.get_output_var_cls(data["type"])
        return from_dict(clz, data, config=config)

    def batch_provider_hook(data: Dict[str, Any]) -> BatchProviderConfig:
        return resolve_single_provider_config(data)

    cfg = expand_env_vars(cfg)
    config = Config(
        type_hooks={
            ImageOutputMode: image_output_mode_hook,
            BaseProcessorConfig: processor_config_hook,
            BaseDataLoaderConfig: loader_config_hook,
            OutputVar: output_var_hook,
            BatchProviderConfig: batch_provider_hook,
        }
    )
    cfg_obj = from_dict(AnonLibConfig, cast(dict, cfg), config=config)

    return cfg_obj
