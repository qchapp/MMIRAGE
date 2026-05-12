# MMIRAGE

MMIRAGE (**M**odular **M**ultimodal **I**ntelligent **R**eformatting and **A**ugmentation **G**eneration **E**ngine) is a framework for dataset reformatting and augmentation with language models and vision-language models.

This artifact provides the code and configuration required to reproduce the MedTrinity demo formatting pipeline used in the submitted workshop paper.

## Reproducing the MedTrinity Demo Pipeline

This section explains how to download the MedTrinity demo dataset and run the MMIRAGE formatting pipeline.

The reproduction uses the MedTrinity demo subset, loaded with the Hugging Face config `25M_demo`. The saved local dataset contains the columns `image`, `id`, and `caption`. The image is stored directly inside the Hugging Face dataset, so no separate `image_base_path` is required.

## 1. Environment setup

Clone or unpack the anonymous artifact repository provided with the submission.

MMIRAGE can be run either in a Python virtual environment or inside Docker.

### Option A: Python virtual environment

Clone the repository:

```bash
git clone <artifact-repo-url> MMIRAGE
cd MMIRAGE
```

Or unpack the anonymous artifact archive:

```bash
unzip MMIRAGE-artifact.zip
cd MMIRAGE
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install MMIRAGE:

```bash
pip install --upgrade pip
pip install -e .
```

Install the GPU extra when using the SGLang-backed `llm` processor for local GPU inference:

```bash
pip install -e ".[gpu]"
```

If using GPU-backed models through SGLang, ensure that the CUDA and GPU dependencies required by MMIRAGE and SGLang are available in the environment.

### Option B: Docker environment

The repository provides Docker and Docker Compose configurations.

The `docker-compose.yml` defines two services:

- `mmirage` (GPU)
- `mmirage-cpu`

### Prebuilt images

Prebuilt images are available through GHCR:

```text
ghcr.io/epflight/mmirage:latest-gpu
ghcr.io/epflight/mmirage:latest-cpu
```

Example usage:

```bash
# GPU
docker pull ghcr.io/epflight/mmirage:latest-gpu
docker run --rm -it --gpus all ghcr.io/epflight/mmirage:latest-gpu

# CPU
docker pull ghcr.io/epflight/mmirage:latest-cpu
docker run --rm -it ghcr.io/epflight/mmirage:latest-cpu
```

### GPU container

The GPU container requires:

- NVIDIA GPU drivers
- NVIDIA Container Toolkit / `nvidia-container-runtime`
- Docker with GPU support enabled

Build and run:

```bash
docker compose build mmirage
docker compose run --rm -it mmirage
```

### CPU-only container

The CPU image installs MMIRAGE without the GPU extra and is intended for workflows that do not instantiate the SGLang-backed `llm` processor.

Build and run:

```bash
docker compose build mmirage-cpu
docker compose run --rm -it mmirage-cpu
```

You can verify GPU visibility inside the container with:

```bash
nvidia-smi
```

## 2. Accept MedTrinity access and log in to Hugging Face

Open the MedTrinity dataset page and accept the dataset access conditions:

```text
https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M
```

Then log in locally:

```bash
hf auth login
```

## 3. Download the MedTrinity demo dataset

Choose a local data directory:

```bash
export DATA_ROOT=/path/to/data
export MEDTRINITY_DEMO=${DATA_ROOT}/medtrinity_demo
```

Download and save the demo dataset locally:

```bash
python scripts/download_medtrinity_demo.py \
  --output-dir ${MEDTRINITY_DEMO}
```

Expected output:

```text
${MEDTRINITY_DEMO}/
  dataset_info.json
  state.json
  data-*.arrow
  medtrinity_demo_download_metadata.json
```

## 4. Verify the dataset

Run:

```bash
python scripts/verify_medtrinity_demo.py \
  --dataset-path ${MEDTRINITY_DEMO}
```

The verification script checks that:

- the dataset can be loaded with `datasets.load_from_disk`
- the required columns `image`, `id`, and `caption` exist
- images decode correctly as PIL images
- captions and ids are non-empty strings

## 5. Run the local pipeline

Set the output and cache directories:

```bash
export SCRATCH=/path/to/scratch
export HF_HOME=/path/to/hf
```

Run MMIRAGE with the local reviewer configuration:

```bash
mmirage run --config configs/medtrinity_demo_local.yaml --stats
```

Outputs are written to:

```text
${SCRATCH}/medtrinity_demo_conversations_formatted_local
```

Output schema:

```json
{
  "id": "...",
  "conversations": [
    {
      "role": "user",
      "content": "Describe this medical image."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "modalities": [
    "<image stored in the dataset>"
  ]
}
```

Inspect run statistics:

```bash
mmirage stats --config configs/medtrinity_demo_local.yaml
```

## 6. Run the 16-node SLURM pipeline

Set:

```bash
export DATA_ROOT=/path/to/data
export MEDTRINITY_DEMO=${DATA_ROOT}/medtrinity_demo
export SCRATCH=/path/to/scratch
export HF_HOME=/path/to/hf
export EDF_ENV=/path/to/edf
```

Run:

```bash
mmirage run --config configs/medtrinity_demo_16nodes.yaml --stats
```

The SLURM configuration uses 16 logical shards:

```yaml
num_shards: 16
shard_id: "$SLURM_ARRAY_TASK_ID"
```

Each shard requests one node:

```yaml
nodes: 1
gpus: 4
```

When all array tasks run concurrently, the pipeline uses 16 nodes total.

Inspect statistics:

```bash
mmirage stats --config configs/medtrinity_demo_16nodes.yaml
```

## 7. Relevant artifact files

```text
configs/medtrinity_demo_local.yaml
configs/medtrinity_demo_16nodes.yaml
scripts/download_medtrinity_demo.py
scripts/verify_medtrinity_demo.py
```

## Features

MMIRAGE supports:

- text and image processing with LLMs and VLMs
- YAML-configurable pipelines
- Jinja2 prompt templating
- JMESPath-based dataset extraction
- distributed sharded execution
- local and SLURM execution modes
- configurable structured outputs
- modular processors, loaders, and writers

## Basic CLI usage

Run a pipeline:

```bash
mmirage run --config configs/config_mock.yaml
```

Check shard status:

```bash
mmirage check --config configs/config_mock.yaml
```

Retry failed shards:

```bash
mmirage check --config configs/config_mock.yaml --retry
```

Merge shard outputs:

```bash
mmirage merge --config configs/config_mock.yaml
```

Inspect statistics:

```bash
mmirage stats --config configs/config_mock.yaml
```

## Example configuration

The following simplified configuration illustrates the main MMIRAGE sections.

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768

loading_params:
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/image/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 32

processing_params:
  inputs:
    - name: medical_image
      key: image
      type: image
    - name: original_caption
      key: caption
      type: text

  outputs:
    - name: enhanced_caption
      type: llm
      output_type: plain
      prompt: |
        Describe the medical image in detail.
        Original caption for context: {{ original_caption }}

  remove_columns: false
  output_schema:
    image: "{{ medical_image }}"
    caption: "{{ enhanced_caption }}"
    original_caption: "{{ original_caption }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

Main configuration sections:

- `processors`: model and inference configuration
- `loading_params`: dataset loading and sharding
- `processing_params`: prompts, extracted variables, and output schema
- `execution_params`: local or SLURM execution behavior

## Statistics

Enable runtime and throughput statistics with:

```bash
mmirage run --config configs/config_mock.yaml --stats
```

Inspect results:

```bash
mmirage stats --config configs/config_mock.yaml
```

Common metrics include:

- `runtime_seconds`
- `rows_processed`
- `throughput_rows_per_sec`
- `gpu_util_mean`
- `tokens_per_sec_per_gpu`
- `gpu_days_per_billion_tokens`

Token metrics are `null` when no LLM processor is active. GPU metrics are `null` when `nvidia-smi` is unavailable or `--stats` was not enabled.

## Repository structure

```text
mmirage/
├── config/           # Configuration loading and validation
├── core/
│   ├── loader/       # Dataset loaders
│   ├── process/      # Processors and variable system
│   │   └── processors/
│   │       └── llm/  # LLM/VLM processor
│   └── writer/       # Output rendering
├── shard_process.py  # Main shard processing script
└── merge_shards.py   # Shard merging utility
```

## Useful references

- Jinja2: https://jinja.palletsprojects.com/en/stable/
- JMESPath: https://jmespath.org/
- SGLang: https://github.com/sgl-project/sglang
- DataTrove benchmark: https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark