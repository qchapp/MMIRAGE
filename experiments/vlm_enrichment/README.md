# VLM enrichment

This experiment measures multimodal task generalization by restructuring medical-image descriptions with MMIRAGE, SGLang, DataTrove, and NeMo Curator on four H100 GPUs.

## Workload

Source dataset: [`UCSC-VLAA/MedTrinity-25M`](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M), configuration `25M_demo`, split `train`.

`scripts/prepare_workload.py` shuffles with seed `20260813`, keeps the first **83** samples with unique non-empty IDs, and saves each source image as a PNG so all frameworks reuse the same local image/caption pairs. The resolved dataset revision is recorded in `workload/metadata.json`.

Each framework receives the medical image, its original MedTrinity caption, and the same transformation instruction:

```text
Reformat the image description with markdown without adding anything else.
Add titles and structure your output.

Image description:
{caption}
```

## Execution

All paths use `Qwen/Qwen3-VL-4B-Instruct`, temperature `0.1`, top-p `0.9`, and a maximum of `1,024` new tokens. MMIRAGE uses batch size `64`; native runners are configured with concurrency `64`. The experiment uses physical GPUs `0`, `1`, `2`, and `3`, with three measured repetitions per framework.

The MMIRAGE recipe serializes the generated answer into its configured structured output containing `id`, `source_index`, `conversations`, and `modalities`. The native VLM harness records `id`, `source_index`, and the generated `formatted_description`. The source image/caption and transformation instruction are matched; the final serialization is framework-specific.

The NeMo Curator VLM path issues model requests row-by-row in the retained integration, while SGLang and DataTrove use their respective serving/inference paths.

## Run this experiment only

Create the competitor environments described in [`../publication/ENVIRONMENTS.md`](../publication/ENVIRONMENTS.md), then set the Python interpreter variables used by the DataTrove and NeMo workers:

```bash
export HF_TOKEN=...
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
```

Prepare the deterministic MedTrinity subset and cache the exact VLM snapshot before timing:

```bash
python experiments/vlm_enrichment/scripts/prepare_workload.py \
  --output-dir experiments/vlm_enrichment/workload

python experiments/publication/prefetch_models.py \
  --models Qwen/Qwen3-VL-4B-Instruct \
  --output-json experiments/vlm_enrichment/workload/model_revisions.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run only the VLM experiment:

```bash
python experiments/publication/orchestrate.py \
  --stage vlm \
  --repetitions 3 \
  --overwrite
```

Add `--dry-run` to inspect the exact framework commands without running inference.

## Outputs

Results and validation records are written under `results/`. Rows/s and end-to-end wall time are the primary performance measurements for this task.
