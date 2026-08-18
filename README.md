<p align="center">
  <img src="docs/_static/logo.svg" alt="MMIRAGE" width="480">
</p>

# MMIRAGE

MMIRAGE is a framework for large-scale dataset reformatting and augmentation with language, vision-language, and image-generation models. It provides declarative YAML pipelines, sharded local and SLURM execution, resumable processing, and structured output rendering.

## Installation

Clone the repository and install the base package:

```bash
git clone <anonymous-repository-url> MMIRAGE
cd MMIRAGE
pip install -e .
```

For local SGLang-backed LLM/VLM execution, install a CUDA-enabled PyTorch build compatible with the target machine and then install the GPU extra:

```bash
pip install -e ".[gpu]"
```

The experiment environments and package pins are documented in [`experiments/publication/ENVIRONMENTS.md`](experiments/publication/ENVIRONMENTS.md). Image-generation support is optional:

```bash
pip install -e ".[image_gen]"
```

## Basic usage

A MMIRAGE pipeline is described by YAML. The configuration defines processors, dataset loading/sharding, variables extracted from each row, output prompts, output schema, and execution/retry behavior.

```bash
mmirage run --config configs/config_mock.yaml
mmirage check --config configs/config_mock.yaml
mmirage check --config configs/config_mock.yaml --retry
mmirage merge --config configs/config_mock.yaml
```

Shard outputs can also be merged directly without a configuration file:

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

MMIRAGE supports text and multimodal `llm` processors, provider `batch_api` execution, image generation, custom Python processors, JSON/Jinja-based structured outputs, and local or SLURM sharding.

## Experiments

Evaluation code and reproduction instructions are in [`experiments/README.md`](experiments/README.md). The experiments cover strong scaling, shard recovery, text-task generalization, multimodal enrichment, and endpoint-matched SGLang overhead.

Exact comparison settings are documented in [`experiments/publication/PROTOCOL.md`](experiments/publication/PROTOCOL.md), with interpretation constraints in [`experiments/publication/LIMITATIONS.md`](experiments/publication/LIMITATIONS.md).

## Statistics

Pass `--stats` to MMIRAGE execution to collect shard timing, GPU utilization, row throughput, and token throughput where applicable:

```bash
mmirage run --config configs/config_mock.yaml --stats
mmirage stats --config configs/config_mock.yaml
```

## Documentation

Build the local Sphinx documentation with:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install --no-deps -e .
python -m sphinx -b html -j auto docs docs/_build/html --keep-going
```

The generated documentation is written to `docs/_build/html/`.

## Architecture

MMIRAGE uses a modular architecture:

```text
mmirage/
├── config/           # configuration loading and validation
├── core/
│   ├── loader/       # dataset loaders
│   ├── process/      # processors and variable system
│   └── writer/       # output rendering
├── shard_process.py  # shard worker
└── merge_shards.py   # shard merge utility
```
