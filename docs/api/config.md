# `anonlib.config` — Configuration

## `anonlib.config.config`

Main configuration dataclasses for the AnonLib pipeline.

```{eval-rst}
.. automodule:: anonlib.config.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `anonlib.config.loading`

Dataset loading and sharding configuration.

```{eval-rst}
.. automodule:: anonlib.config.loading
   :members:
   :undoc-members:
   :show-inheritance:
```

## `anonlib.config.batch_provider`

Provider-agnostic batch submission configuration used by the OpenAI Batch API integration and any future batch providers.

```{eval-rst}
.. automodule:: anonlib.config.batch_provider
   :members:
   :undoc-members:
   :show-inheritance:
```

## `anonlib.config.openai_batch`

OpenAI-specific batch configuration extending {class}`~anonlib.config.batch_provider.BatchProviderConfig`.

```{eval-rst}
.. automodule:: anonlib.config.openai_batch
   :members:
   :undoc-members:
   :show-inheritance:
```

## `anonlib.config.utils`

YAML parsing helpers and configuration loaders.

```{eval-rst}
.. automodule:: anonlib.config.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
