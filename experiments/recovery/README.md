# Recovery

This experiment evaluates shard reuse and recomputation after deliberately terminating active workers during generation on four H100 GPUs.

## Workload

Source dataset: [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), split `train_sft`.

The recovery workload is the first **4,000** rows of the same deterministic UltraChat selection used by the scaling experiment: shuffle seed `20260813`, first user request extracted from each conversation, normalized prompts, duplicate prompt hashes removed. The 4,000 rows are split into **16 logical shards of 250 rows each**.

The generation task is identical to scaling:

```text
You are helping construct a public text dataset.

Rewrite the following user request into a helpful assistant response of 4 to 6 sentences.
Preserve the user's intent.

User request:
{prompt_text}
```

All paths use `Qwen/Qwen3-4B`, temperature `0`, and at most `256` new tokens. MMIRAGE uses batch size `64`; native competitors use concurrency `64`.

## Injected failures

At most four logical shard workers run simultaneously, pinned across physical GPUs `0`, `1`, `2`, and `3`. Every condition has three repetitions.

MMIRAGE runs:

- `baseline`: no worker is terminated.
- `fail_1`: logical shard **3** is deliberately terminated.
- `fail_4`: logical shards **1, 5, 9, and 13** are deliberately terminated.

Direct SGLang, DataTrove, NeMo Curator, Distilabel, and Ray Data LLM run the `fail_1` and `fail_4` conditions.

The injected event is a **process-level worker failure during active generation**. For each targeted shard, the controller waits until the wrapper has entered its running state, then waits **30 seconds** and sends **SIGTERM** to that shard's worker/wrapper process. It does not kill the whole benchmark controller, node, or GPU. Non-targeted workers continue normally. After the initial wave finishes, recovery/retry relaunches shards without a valid completion marker.

For MMIRAGE, completed shard outputs are snapshotted before and after retry so reuse can be checked directly. For native competitors, validation checks that completed shard outputs are unchanged, retry is limited to incomplete or killed shards, and the final merged output contains no missing, duplicate, or unexpected IDs.

`run_local.py` controls the MMIRAGE local-shard experiment. `run_native_recovery_publication.py` invokes the native recovery controller and clears orphaned vLLM engine processes between waves when necessary. `extract_results.py` aggregates the recorded recovery evidence.

## Run this experiment only

Create the framework environments described in [`../publication/ENVIRONMENTS.md`](../publication/ENVIRONMENTS.md), then set the interpreter variables to the corresponding Python executables:

```bash
export HF_TOKEN=...
export MMIRAGE_DATATROVE_PYTHON="$PWD/.venv-datatrove/bin/python"
export MMIRAGE_NEMO_CURATOR_PYTHON="$PWD/.venv-nemo_curator/bin/python"
export MMIRAGE_DISTILABEL_PYTHON="$PWD/.venv-distilabel/bin/python"
export MMIRAGE_RAY_DATA_LLM_PYTHON="$PWD/.venv-ray_data_llm/bin/python"
export MMIRAGE_RECOVERY_ROOT="$PWD/experiments/recovery/workdir"
```

Prepare the shared deterministic UltraChat subset and cache the model before timing:

```bash
python experiments/scaling/scripts/prepare_workload.py \
  --output-dir experiments/scaling/workload \
  --shared-root "$MMIRAGE_RECOVERY_ROOT"

python experiments/publication/prefetch_models.py \
  --models Qwen/Qwen3-4B \
  --output-json experiments/scaling/workload/model_revisions.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run only the recovery conditions and then extract the MMIRAGE recovery summary:

```bash
python experiments/publication/orchestrate.py \
  --stage recovery \
  --repetitions 3 \
  --overwrite

python experiments/publication/orchestrate.py \
  --stage recovery_extract \
  --repetitions 3
```

To inspect the recovery commands without launching them, add `--dry-run` to the first command.

## Outputs

Standalone execution keeps raw controller state and native evidence under `experiments/recovery/workdir/`; `recovery_extract` writes the MMIRAGE aggregate JSON/CSV under `experiments/recovery/workdir/results/`. The full H100 driver additionally copies the publication evidence into `experiments/recovery/results/`.

The key recovery evidence is the set of shards reused unchanged, the set recomputed after failure, and final-output validity.
