# Experiment environments

The evaluation uses one main MMIRAGE/SGLang environment and isolated Python environments for framework baselines whose dependency sets are not mutually compatible. All packages listed in `environment/*_requirements.txt` are pinned to exact versions.

## Main environment

Install the fully pinned Python 3.12/CUDA 12.9 environment used for the publication runs, then install the repository itself without resolving dependencies again:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  -r requirements-hpc-lock.txt
python -m pip install --no-deps -e .
```

This interpreter is used for MMIRAGE, workload preparation, SGLang, and orchestration. The shell drivers invoke it as `python`. The direct MMIRAGE and `gpu` dependencies in `requirements-hpc-lock.txt` must remain compatible with `pyproject.toml`; the lock also fixes the transitive packages and CUDA wheels used for the reported environment.

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

The H100 and A100 drivers use the repository-root locations above by default. If the environments live elsewhere, set the corresponding variables to the **Python interpreter executable**, not to the environment directory. For example:

```bash
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
export MMIRAGE_DISTILABEL_PYTHON="$PWD/.venv-distilabel/bin/python"
export MMIRAGE_RAY_DATA_LLM_PYTHON="$PWD/.venv-ray_data_llm/bin/python"
```

A value such as `$PWD/.venv-datatrove` is **not** sufficient; the variable must end in the executable itself, e.g. `.venv-datatrove/bin/python`.

`MMIRAGE_DISTILABEL_PYTHON` and `MMIRAGE_RAY_DATA_LLM_PYTHON` are needed only for the H100 recovery experiment. The A100 transfer run uses DataTrove and NeMo Curator only.

The separate `raw_sglang_uv_requirements.txt` records the exact packages for the direct SGLang environment used when reproducing that baseline independently; the publication drivers otherwise use SGLang from the main MMIRAGE environment.

Python 3.12 environments importing vLLM require `SETUPTOOLS_USE_DISTUTILS=local`. The H100/A100 shell drivers export it automatically. When running an individual experiment directly through `orchestrate.py`, set it in the calling shell first:

```bash
export SETUPTOOLS_USE_DISTUTILS=local
export TOKENIZERS_PARALLELISM=false
```

## Credentials and model revisions

Set a Hugging Face access token in the environment before launching either hardware suite or preparing a standalone workload:

```bash
export HF_TOKEN=...
```

The H100 driver resolves and caches exact Hugging Face model commit SHAs before timed execution, records them with GPU model, UUID, memory, and driver version, and then sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The A100 driver verifies the H100 workload hashes and model revision and downloads that exact model snapshot before switching offline.
