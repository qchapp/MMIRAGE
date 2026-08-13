# `anonlib.core.process` — Processors

## Variables

```{eval-rst}
.. automodule:: anonlib.core.process.variables
   :members:
   :undoc-members:
   :show-inheritance:
```

## Base processor

```{eval-rst}
.. automodule:: anonlib.core.process.base
   :members:
   :undoc-members:
   :show-inheritance:
```

## Mapper

```{eval-rst}
.. automodule:: anonlib.core.process.mapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## LLM processor

### Configuration

```{eval-rst}
.. automodule:: anonlib.core.process.processors.llm.config
   :members:
   :undoc-members:
   :show-inheritance:
```

### Implementation

```{eval-rst}
.. automodule:: anonlib.core.process.processors.llm.llm_processor
   :members:
   :undoc-members:
   :show-inheritance:
```

## Batch processing

The batch subsystem handles asynchronous, provider-backed inference (e.g. OpenAI Batch API). It is activated by setting `batch_provider` in the processor configuration.

### Orchestrator

```{eval-rst}
.. automodule:: anonlib.core.process.batch.orchestrator
   :members:
   :undoc-members:
   :show-inheritance:
```

### Adapter (provider-neutral interface)

```{eval-rst}
.. automodule:: anonlib.core.process.batch.adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### OpenAI adapter

```{eval-rst}
.. automodule:: anonlib.core.process.batch.openai_adapter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Chunking

```{eval-rst}
.. automodule:: anonlib.core.process.batch.chunking
   :members:
   :undoc-members:
   :show-inheritance:
```

### Status checker

```{eval-rst}
.. automodule:: anonlib.core.process.batch.status_checker
   :members:
   :undoc-members:
   :show-inheritance:
```
