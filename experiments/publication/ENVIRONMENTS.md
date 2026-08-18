# Experiment environments

The evaluation uses one main MMIRAGE/SGLang environment and isolated Python environments for framework baselines whose dependency sets are not mutually compatible. All packages listed in `environment/*_requirements.txt` are pinned to exact versions.

## Main environment

Install MMIRAGE with its GPU dependencies in the environment used to launch the publication drivers:

```bash
python -m pip install -e ".[gpu]"
```

This interpreter is used for MMIRAGE, workload preparation, SGLang, and orchestration. The shell drivers invoke it as `python`.

## Competitor environments

A convenient layout is to create the environments at the repository root:

```text
.venv-datatrove/bin/python
.venv-nemo_curator/bin/python
.venv-distilabel/bin/python
.venv-ray_data_llm/bin/python
```

For example, with `uv` and Python 3.12:

```bash
uv venv --python 3.12 .venv-datatrove
uv pip install --python .venv-datatrove/bin/python -r experiments/publication/environment/datatrove_uv_requirements.txt

uv venv --python 3.12 .venv-nemo_curator
uv pip install --python .venv-nemo_curator/bin/python -r experiments/publication/environment/nemo_curator_uv_requirements.txt

uv venv --python 3.12 .venv-distilabel
uv pip install --python .venv-distilabel/bin/python -r experiments/publication/environment/distilabel_uv_requirements.txt

uv venv --python 3.12 .venv-ray_data_llm
uv pip install --python .venv-ray_data_llm/bin/python -r experiments/publication/environment/ray_data_llm_uv_requirements.txt
```

The H100 and A100 drivers use the repository-root locations above by default. If the environments live elsewhere, set the corresponding variables to the **Python interpreter executable**, not to the environment directory:

```bash
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
export MMIRAGE_DISTILABEL_PYTHON="$PWD/.venv-distilabel/bin/python"
export MMIRAGE_RAY_DATA_LLM_PYTHON="$PWD/.venv-ray_data_llm/bin/python"
```

`MMIRAGE_DISTILABEL_PYTHON` and `MMIRAGE_RAY_DATA_LLM_PYTHON` are needed only for the H100 recovery experiment. The A100 transfer run uses DataTrove and NeMo Curator only.

The separate `raw_sglang_uv_requirements.txt` records the exact packages for the direct SGLang environment used when reproducing that baseline independently; the publication drivers otherwise use SGLang from the main MMIRAGE environment.

Python 3.12 environments importing vLLM require `SETUPTOOLS_USE_DISTUTILS=local`; the publication drivers export it before preflight and execution.

## Credentials and model revisions

Set a Hugging Face access token in the environment before launching either hardware suite:

```bash
export HF_TOKEN=...
```

The H100 driver resolves and caches exact Hugging Face model commit SHAs before timed execution, records them with GPU model, UUID, memory, and driver version, and then sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The A100 driver verifies the H100 workload hashes and model revision and downloads that exact model snapshot before switching offline.
