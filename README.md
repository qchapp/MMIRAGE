<p align="center">
  <img src="docs/_static/logo.svg" alt="MMIRAGE" width="480">
</p>

# MMIRAGE

MMIRAGE is a framework for large-scale dataset reformatting and augmentation with
language, vision-language, and image-generation models. It provides declarative
YAML pipelines, sharded local and SLURM execution, resumable processing, and
structured output rendering.

## How to install

To install the library, clone it from GitHub and install it with pip.

### Base install

The base install does **not** include the local SGLang runtime:

```bash
git clone <anonymous-repository-url> MMIRAGE
cd MMIRAGE
pip install -e .
```

### GPU install (SGLang-backed `llm` processor)

MMIRAGE requires a **CUDA-enabled PyTorch installation** before installing the GPU extra.

Install a PyTorch build matching your CUDA runtime (example for CUDA 12.9):

```bash
pip install --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output should include:

- `+cu129`
- `12.9`
- `True` (when run on a GPU node)

Then install MMIRAGE GPU support:

```bash
pip install -e ".[gpu]"
```

`sglang` does not put an upper bound on `flash-attn-4` or `nvidia-cutlass-dsl`, so a fresh install can
resolve a combination that fails at H100 attention-backend initialization with
`AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'`. If the GPU
experiments fail that way, pin the versions recorded in `requirements-hpc-lock.txt`:

```bash
pip install flash-attn-4==4.0.0b15 nvidia-cutlass-dsl==4.5.2 apache-tvm-ffi==0.1.11
```

Install the SGLang diffusion extra when MMIRAGE should launch an image-generation server:

```bash
pip install -e ".[image_gen]"
```

> **Note:** On some platforms (e.g. clusters or ARM/aarch64), `pip install -e ".[gpu]"` alone may resolve to a CPU-only PyTorch build.

### Environment file

For testing and scripts that make use of the library, it is advised to create a `.env` file:

```bash
./scripts/generate_env.sh
```

## Local documentation

The Sphinx documentation is built and viewed locally; no automated deployment
is included.

```bash
python -m pip install -r docs/requirements.txt
python -m pip install --no-deps -e .
python -m sphinx -b html -j auto docs docs/_build/html --keep-going
```

Open `docs/_build/html/index.html` in a browser after the build completes.

## Docker

The `docker-compose.yml` defines two services, `mmirage` (GPU) and `mmirage-cpu`.

### GPU

The container requires an NVIDIA GPU. The `docker-compose.yml` is configured to request GPU access, but the host must have:
- NVIDIA GPU drivers installed
- NVIDIA Container Toolkit / `nvidia-container-runtime` configured for Docker
- A recent Docker Engine and Docker Compose version with GPU support enabled

Commands:

```bash
# Build
docker compose build mmirage

# Run
docker compose run --rm -it mmirage
```

### CPU-only

The CPU image installs MMIRAGE without the GPU extra. It supports workflows that do not instantiate the SGLang-backed `llm` processor, including `custom` and `batch_api` pipelines. Use `configs/config_mock_custom_module.yaml` as a local custom-processor example or one of the `config_mock_openai_batch*.yaml` files with the corresponding API credentials.

Commands:

```bash
# Build
docker compose build mmirage-cpu

# Run
docker compose run --rm -it mmirage-cpu
```

## Key features

- **Multimodal Support**: Process both text and images with vision-language models
- Easily configurable with a YAML file which configures the following parameters:
    - The prompt to the LLM (using Jinja2 templating)
    - Variables with the name and their JMESPath key to a JSON
    - Image inputs for multimodal processing
- Parallelizable with multi-node support
    - The training pipeline uses distributed inference with sharding
- Support a variety of LLMs, VLMs (Vision-Language Models), and image generation models
- Support any dataset schemas (configurable with the YAML format)
- The ability to either output a JSON (or any other structured format) or plain text
- Modular architecture with pluggable processors, loaders, and writers
- Custom Python processors executed in isolated worker pools
- Asynchronous provider batch submission through the `batch_api` processor

## Example usage

### Running (single command)

Run the pipeline via the CLI. Retry behavior is driven by your YAML config:

- `execution_params.retry: true` → automatically retries failed shards until completion or `max_retries`
- `execution_params.retry: false` → submits/runs once; you can later trigger retries via `check`
- `execution_params.merge: true` → after a successful run, automatically merges shard outputs

```bash
mmirage run --config configs/config_mock.yaml
```

To check status only:

```bash
mmirage check --config configs/config_mock.yaml
```

To check status and submit retries for failed shards:

```bash
mmirage check --config configs/config_mock.yaml --retry
```

To merge shards from the CLI directly:

```bash
mmirage merge --config configs/config_mock.yaml
```

To merge shards without a config file (input directory + output directory only):

```bash
mmirage merge-dir --input-dir /path/to/shards --output-dir /path/to/merged
```

`--input-dir` can point either to a single dataset directory that contains `shard_*`
folders, or to a parent directory containing multiple dataset subdirectories.
If `shard_*` folders are present directly in `--input-dir`, MMIRAGE merges that
root dataset directly and ignores nested internal folders.

For multiple datasets, you can also choose a shared merge root:

```bash
mmirage merge --config configs/config_mock.yaml --output-root /path/to/merged
```

MMIRAGE still keeps datasets separate by creating one subdirectory per dataset under the root.

### Text-only: Reformatting dataset

Suppose you have a dataset with samples of the following format

```json
{
    "conversations" : [{"role": "user", "content": "Describe the image"}, {"role": "assistant", "content": "This is a badly formmatted answer"}],
    "modalities" : ["<the images>"]
}
```

The dataset contains assistant answers that are badly formatted. The goal would be to use a LLM to format our answer in Markdown. With MMIRAGE, it would be as simple as defining a YAML configuration file:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen3-8B
      tp_size: 4
      trust_remote_code: true
    default_sampling_params:
      temperature: 0.1
      top_p: 1.0
      max_new_tokens: 384

loading_params:
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/dataset
      type: loadable
      output_dir: /path/to/output/shards
  num_shards: 4
  shard_id: "$SLURM_ARRAY_TASK_ID"
  batch_size: 64

processing_params:
  inputs:
    - name: assistant_answer
      key: conversations[1].content
    - name: user_prompt
      key: conversations[0].content
    - name: modalities
      key: modalities

  outputs:
    - name: formatted_answer
      type: llm
      output_type: plain
      prompt: |
        Reformat the answer in a markdown format without adding anything else:
        {{ assistant_answer }}

  remove_columns: false
  output_schema:
    conversations:
      - role: user
        content: "{{ user_prompt }}"
      - role: assistant
        content: "{{ formatted_answer }}"
    modalities: "{{ modalities }}"

execution_params:
  mode: local
  retry: false
  merge: false
```

Configuration explanation:

- `processors`: List of processor configurations. Currently supports :
  - `llm` (text/VLM generation, run locally with SGLang).
  - `batch_api` (text/VLM generation submitted to a provider batch API, see [Batch API](docs/batch_api.md)).
  - `image_gen` (text-to-image generation).
  - `custom` (a user-provided Python function, see [Custom Module](docs/custom_module.md)).
- `loading_params`: Parameters for loading and sharding datasets.
  - `state_dir`: Optional shared directory for shard status/retry state. Defaults to `~/.cache/MMIRAGE/state_dir`.
  - `datasets`: List of dataset configurations with path, type, and output directory.
- `processing_params`:
  - `inputs`: Variables extracted from the input dataset using JMESPath queries.
  - `outputs`: Variables created by processors. Prompts use Jinja2 templating (`{{ variable }}`).
  - `output_schema`: Defines the structure of output samples.
- `execution_params`:
  - `mode`: "local" to run shard processing in the current Python environment or "slurm" to run through SLURM by submitting an sbatch array job.
  - `retry`: If true, MMIRAGE automatically retries failed shards until they succeed or `max_retries` is reached. If false, the pipeline runs/submits once, and retries can be triggered later via the check/retry CLI commands.
  - `merge`: If true, MMIRAGE merges shard outputs after a successful `run`. Merged datasets are written under each dataset `output_dir` in a `merged` subdirectory.

Merge output behavior with multiple datasets:
- Default (`run` with `execution_params.merge: true`, or `merge` without `--output-root`): each dataset is merged to its own `<dataset.output_dir>/merged`.
- Shared root (`merge --output-root ...`): one merged subdirectory is created per dataset under the root.

### Multimodal: Processing images with VLMs

MMIRAGE supports multimodal processing with vision-language models:

```yaml
processors:
  - type: llm
    server_args:
      model_path: Qwen/Qwen2-VL-7B-Instruct
      tp_size: 4
      trust_remote_code: true
    chat_template: qwen2-vl  # Required for VLMs
    default_sampling_params:
      temperature: 0.1
      top_p: 0.95
      max_new_tokens: 768

loading_params:
  state_dir: path/to/state/dir
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
      type: image  # Mark as image input
      image_base_path: /path/to/images  # Base directory for relative paths
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
```

### Image generation: Text-to-image pipeline

MMIRAGE supports image generation through an already-running HTTP server:

```yaml
processors:
  - type: image_gen
    backend: external
    external:
      base_url: http://127.0.0.1:30010/v1
      timeout_seconds: 900
      max_concurrent_requests: 4
      request_model: null
    default_sampling_params:
      num_inference_steps: 20
      guidance_scale: 7.5
    output_dir: /path/to/generated/images
    file_format: png

loading_params:
  state_dir: /path/to/state/dir
  datasets:
    - path: /path/to/prompts.jsonl
      type: JSONL
      output_dir: /path/to/output/shards
  num_shards: 1
  shard_id: 0
  batch_size: 8

processing_params:
  inputs:
    - name: prompt_text
      key: text

  outputs:
    - name: generated_image
      type: image_gen
      output_mode: path          # "path" or "pil"
      filename_template: "generated_{{ __shard_id }}_{{ __sample_index }}_{{ __source_hash }}"
      width: 512
      height: 512
      prompt: |
        Create an illustration of:
        {{ prompt_text }}

  remove_columns: false
  output_schema:
    text: "{{ prompt_text }}"
    image: "{{ generated_image }}"

execution_params:
  mode: local
  retry: false
```

Install optional image generation dependencies before running this config:

```bash
pip install -e ".[image_gen]"
```

Key multimodal features:
- `chat_template`: Specify the VLM chat template (e.g., `qwen2-vl`)
- `type: image`: Mark input variables as images
- `image_base_path`: Base directory for resolving relative image paths
- Supports PIL Images, URLs, and file paths

## Running Benchmarks And Experiments

Run all commands in this section from the repository root. Install the GPU extra
before running any SGLang-backed benchmark or experiment:

```bash
python -m pip install -e '.[gpu]'
python -c "import sglang; print(sglang.__version__)"
```

The expected SGLang version is `0.5.10`. The first run may download the model,
so ensure the machine can access Hugging Face and has sufficient cache space.

### Experiment runbooks

The reproducible paper experiments live under `experiments/<name>/`. The root
README is only an index; use the experiment README as the source of truth for
commands, output directories, expected files, common failures, and metadata to
archive.

| Experiment | Hardware or service needed | Safe preflight | Full rerun instructions |
|---|---|---|---|
| Raw SGLang overhead | One CUDA GPU for `Qwen/Qwen3-4B`; H100 command documented separately. | `python experiments/raw_sglang_overhead/scripts/run.py --workload-dir experiments/raw_sglang_overhead/workload --output-dir /tmp/mmirage_overhead_dry_run --repetitions 1 --dry-run` | [`experiments/raw_sglang_overhead/README.md`](experiments/raw_sglang_overhead/README.md) |
| Shard recovery | Kubernetes or Run:ai namespace, H100 access, `kubectl`, and a shared ReadWriteMany PVC. A local signal-based fallback is also provided. | `python experiments/shard_recovery/scripts/run_k8s.py --help` and `python experiments/shard_recovery/scripts/run_local.py --help` | [`experiments/shard_recovery/README.md`](experiments/shard_recovery/README.md) |
| Single-node H100 scaling | One node or pod with four visible H100 GPUs. | `bash experiments/single_node_h100_scaling/scripts/run_4gpu.sh --dry-run` | [`experiments/single_node_h100_scaling/README.md`](experiments/single_node_h100_scaling/README.md) |
| MMIRAGE vs NeMo Curator comparison | One H100-80GB for the full shared-SGLang ChartQA run, plus isolated MMIRAGE and NeMo Python environments. | Follow the mock-server smoke check in the runbook; `run_comparison.py --dry-run` validates planned commands without running either framework. | [`experiments/nemo_curator_comparison/README.md`](experiments/nemo_curator_comparison/README.md) |

Use fresh output directories for reruns unless the experiment README explicitly
tells you to pass `--overwrite`. Generated experiment outputs are intentionally
not committed.

To check a freshly cloned repository without launching inference, Kubernetes
pods, or GPU benchmarks, run:

```bash
python -m py_compile experiments/_shared/*.py \
  experiments/raw_sglang_overhead/scripts/*.py \
  experiments/shard_recovery/scripts/*.py \
  experiments/single_node_h100_scaling/scripts/*.py \
  experiments/nemo_curator_comparison/scripts/*.py
```

Then run the dry-run or smoke-check command in each experiment's own README.

For the global experiment layout policy, see
[`experiments/README.md`](experiments/README.md).

### Collecting statistics for any pipeline

Pass `--stats` to `run` or `submit` to enable per-shard benchmarking. This activates GPU
utilization polling and throughput tracking on compute nodes — disabled by default to
avoid unnecessary overhead.

```bash
# Local run with stats collection
mmirage run --config configs/config_mock.yaml --stats
```

After the run completes, inspect the results with:

```bash
mmirage stats --config configs/config_mock.yaml
```

This prints a JSON report containing per-shard statistics and an aggregate
summary.

Key metrics:
- **`runtime_seconds`** / **`runtime_human`**: time from when the shard started on the cluster (after dispatch), excluding queue wait time.
- **`overall_throughput_rows_per_sec`**: total rows / wall-clock time across all shards running in parallel.
- **`mean_gpu_util_pct`**: mean percentage GPU utilization across shards.
- **`tokens_per_sec_per_gpu`**: output tokens generated per second per GPU — the primary throughput metric used by frameworks such as [DataTrove](https://github.com/huggingface/datatrove).
- **`gpu_days_per_billion_tokens`**: total GPU-days consumed to generate 1 billion output tokens — useful for cost and scaling comparisons across different hardware configurations.
- Token metrics are `null` when no LLM processor was active, and GPU stats are `null` when `nvidia-smi` is unavailable or `--stats` was not passed.

### DataTrove-Compatible Throughput Benchmark

The config `configs/config_benchmark_datatrove.yaml` mirrors the DataTrove inference benchmark conditions:

| Setting | Value |
|---|---|
| Dataset | `simplescaling/s1K-1.1` (train split, 1 000 samples) |
| Prompt | raw `question` field, no system prompt |
| Output | up to 1 024 tokens per sample |
| Context | 2 048-token model max context |
| Model | `Qwen/Qwen3-4B` (DataTrove baseline: tp=1 on a single GPU) |

1. Download and save the benchmark dataset:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("simplescaling/s1K-1.1", split="train")
ds.save_to_disk("data/s1K-1.1")
PY
```

2. Choose the execution mode in `configs/config_benchmark_datatrove.yaml`.
For a direct one-GPU run, set:

```yaml
execution_params:
  mode: local
```

For SLURM, leave `mode: slurm` and replace `my_account` with the allocation on
the target cluster.

3. Run the pipeline with statistics enabled:

```bash
mmirage run --config configs/config_benchmark_datatrove.yaml --stats
```

4. Print the resulting per-shard and aggregate metrics:

```bash
mmirage stats --config configs/config_benchmark_datatrove.yaml
```

The processed shard is written below `data/benchmark_s1k/output/`, and execution
state and statistics are written below `data/benchmark_s1k/_pipeline_state/`.
This experiment measures MMIRAGE throughput under DataTrove-compatible settings;
it does not provide a same-server raw SGLang comparison.

Reference: [DataTrove inference benchmark](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark).

## Architecture

MMIRAGE uses a modular architecture:

```
mmirage/
├── config/           # Configuration loading and validation
├── core/
│   ├── loader/       # Dataset loaders (JSONL, HuggingFace)
│   ├── process/      # Processors (LLM, etc.) and variable system
│   │   └── processors/
│   │       └── llm/  # LLM processor with multimodal support
│   └── writer/       # Output rendering with Jinja2
├── shard_process.py  # Main processing script
└── merge_shards.py   # Shard merging utility
```

## Useful tools

- Jinja2 for template processing: [link](https://jinja.palletsprojects.com/en/stable/)
- JMESPath for JSON queries: [link](https://jmespath.org/)
- SGLang for fast inference: [link](https://github.com/sgl-project/sglang)
- Performance paper: [link](https://arxiv.org/abs/2408.02442)
- DataTrove Benchmark: [link](https://github.com/huggingface/datatrove/tree/main/examples/inference/benchmark)
