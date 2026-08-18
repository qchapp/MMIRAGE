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

The publication environments and package pins are documented in [`experiments/publication/ENVIRONMENTS.md`](experiments/publication/ENVIRONMENTS.md). Image-generation support is optional:

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

## Publication experiments

The submission experiments have a deliberately small, scientific-question-oriented layout under [`experiments/`](experiments/):

| Experiment | Purpose |
|---|---|
| [`experiments/scaling/`](experiments/scaling/) | UltraChat rewrite throughput and H100 strong scaling, plus the four-A100 transfer point |
| [`experiments/recovery/`](experiments/recovery/) | deterministic shard recovery after injected failures |
| [`experiments/text_shortening/`](experiments/text_shortening/) | CNN/DailyMail summarization generalization |
| [`experiments/vlm_enrichment/`](experiments/vlm_enrichment/) | MedTrinity multimodal enrichment |
| [`experiments/sglang_overhead/`](experiments/sglang_overhead/) | endpoint-matched MMIRAGE vs raw SGLang abstraction overhead |

The canonical publication entry points are:

```bash
bash experiments/publication/run_h100.sh --dry-run
bash experiments/publication/run_h100.sh

bash experiments/publication/run_a100.sh --dry-run
bash experiments/publication/run_a100.sh
```

The H100 suite prepares deterministic workloads, resolves exact model revisions, records workload/model/hardware provenance, then executes the timed stages serially. The A100 suite reuses the exact H100-prepared scaling and endpoint-overhead workloads and refuses mismatched commit or workload hashes.

Before packaging an artifact, run the mechanical equivalence verifier:

```bash
python experiments/publication/verify_refactor.py
```

The verifier compares the refactored experiment implementation against the frozen publication baseline, checks the semantic execution plan and configs, confirms the MMIRAGE source tree is unchanged, validates Python/shell syntax, and rejects stale experiment paths in user-facing documentation.

See [`experiments/publication/README.md`](experiments/publication/README.md), [`experiments/publication/PROTOCOL.md`](experiments/publication/PROTOCOL.md), and [`experiments/publication/LIMITATIONS.md`](experiments/publication/LIMITATIONS.md) for the full reproduction and interpretation contract.

A read-only progress dashboard is available during unattended runs:

```bash
python experiments/progress_tracker.py --suite h100
python experiments/progress_tracker.py --suite a100
```

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
