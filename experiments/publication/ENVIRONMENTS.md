# Experiment environments

The publication suite uses separate Python environments because the native competitor dependency sets are not mutually compatible.

| Environment | Purpose |
|---|---|
| main MMIRAGE environment | MMIRAGE, SGLang, workload preparation, orchestration |
| `.venv-datatrove` | DataTrove runner |
| `.venv-nemo_curator` | NeMo Curator runner |
| `.venv-distilabel` | Distilabel recovery runner |
| `.venv-ray_data_llm` | Ray Data LLM recovery runner |

The publication drivers use the following overrides when non-default interpreter locations are required:

```text
MMIRAGE_DATATROVE_PYTHON
MMIRAGE_NEMO_CURATOR_PYTHON
MMIRAGE_DISTILABEL_PYTHON
MMIRAGE_RAY_DATA_LLM_PYTHON
```

Pinned/minimal competitor requirement files are stored in `experiments/publication/environment/`. They document the environments used by the runners; they are not installed automatically by the publication driver.

Python 3.12 competitor environments that import vLLM require `SETUPTOOLS_USE_DISTUTILS=local`; the publication drivers set this before preflight and execution.

The H100 driver records GPU model, UUID, memory and driver version in the publication manifest. It resolves and caches exact Hugging Face model commit SHAs before entering timed regions, then sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The A100 driver verifies the H100 model revision and downloads the exact same snapshot before switching offline.
