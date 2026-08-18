# A Matrix (Publication Benchmark)

The A matrix is the consolidated publication benchmark suite. Its primary
UltraChat rewrite task is used for H100 strong scaling, the four-GPU A100
transfer point, and shard recovery. Two additional task-generalization
experiments use their own workloads and prompts: text shortening and VLM
enrichment.

Frameworks consume the same workload rows, semantic prompt/instruction, model,
decoding budget, and output contract within each comparison. Framework-native
prompt/chat serialization may differ.

| Setup | Frameworks | GPU points | Purpose |
|---|---|---|---|
| `gpu_scaling` | MMIRAGE, Direct SGLang (`raw_sglang`), DataTrove, NeMo Curator | 1 / 2 / 4 H100 | single-node strong scaling |
| `a100_4gpu` | same four | 4 A100 | accelerator-transfer point |
| `recovery` | MMIRAGE + Direct SGLang, DataTrove, NeMo Curator, Distilabel, Ray Data LLM | 4 H100 | shard-scoped recovery |
| `text_shortening` | MMIRAGE, DataTrove, NeMo Curator | 4 H100 | text task generalization |
| `vlm_enrichment` | MMIRAGE, SGLang, DataTrove, NeMo Curator | 4 H100 | multimodal task generalization |

## Corrected publication settings

The corrected publication suite uses:

- 3 repetitions per reported condition;
- MMIRAGE generation/loading batch size 64;
- native text/recovery concurrency 64;
- UltraChat rewrite: `Qwen/Qwen3-4B`, temperature 0, `max_new_tokens=256`;
- text shortening: temperature 0, `max_new_tokens=128`;
- VLM enrichment: `Qwen/Qwen3-VL-4B-Instruct`, temperature 0.1,
  `top_p=0.9`, `max_new_tokens=1024`;
- H100 scaling units serialized so 1-GPU and 2-GPU cells do not overlap on the
  same host;
- recovery failure injection at 30 seconds for `fail_1` and `fail_4`.

The historical 2026-08-15 fast-run cells used older batch/configuration
settings. They are provenance only and must not be reused for the corrected
publication benchmark.

## H100 publication run

Use the dedicated publication driver:

```bash
# Non-destructive plan/preflight check
bash run_all_h100_rerun.sh --dry-run

# Fresh corrected H100 publication run
bash run_all_h100_rerun.sh
```

The script verifies:

- exactly four H100 GPUs;
- the MMIRAGE/SGLang environment;
- all required competitor interpreters;
- Hugging Face authentication;

before deleting old publication outputs. Its `--dry-run` path does not delete
results, regenerate workloads, run inference, or execute recovery extraction.

The normal run prepares the fixed committed workload sizes without running the
smoke calibrator, clears prior publication outputs, and executes:

1. H100 strong scaling, serialized, 3 repetitions;
2. recovery, 3 repetitions per framework/condition cell;
3. recovery extraction over reps 1,2,3;
4. text shortening, 3 repetitions;
5. VLM enrichment, 3 repetitions.

## A100 publication transfer point

The A100 experiment must consume the **exact same prepared A-MATRIX workload**
as the H100 scaling experiment.

After the H100 publication driver prepares the workload:

1. copy the entire directory

   `experiments/a_matrix/workload/`

   from the H100 node to the A100 node;

2. do **not** run `prepare_workload.py` again on the A100 node;

3. preserve `metadata.json`, which records `workload_sha256` and the resolved
   dataset/model revisions;

4. run:

```bash
# Non-destructive hardware/workload/plan check
bash run_all_a100_rerun.sh --dry-run

# Fresh 4xA100 transfer point
bash run_all_a100_rerun.sh
```

`run_all_a100_rerun.sh` verifies that all four GPUs are A100 and recomputes the
SHA-256 of `workload.jsonl`, refusing to run if it does not match
`metadata.json`. The experiment is only the four-GPU point; there is no A100
1/2-GPU sweep.

Equivalent direct command after the workload has been copied and verified:

```bash
python experiments/a_matrix/scripts/run_setup.py \
  --setup a100_4gpu \
  --repetitions 3 \
  --overwrite
```

## Workload and sizes

`experiments/a_matrix/scripts/prepare_workload.py` produces a deterministic
UltraChat workload keyed by `stable_id`. Selection uses the committed seed,
normalization policy, and first unique prompt hashes.

`experiments/a_matrix/workload/metadata.json` records, among other fields:

- `dataset_revision_resolved`;
- `model_revision_resolved`;
- `workload_sha256`.

Two committed size controls are independent:

- `configs/workload_size.yaml`: scaling/A100 workload size;
- `configs/recovery_size.yaml`: recovery subset size.

With `--shared-root`, preparation also writes recovery-compatible
`subset.jsonl` and `id_order.jsonl` under the shared root.

## Recovery conditions

Recovery uses 16 logical shards with at most 4 active simultaneously.

- `baseline`: clean MMIRAGE execution;
- `fail_1`: terminate one designated worker, then retry incomplete work;
- `fail_4`: terminate four designated workers, then retry incomplete work.

MMIRAGE has three framework/condition cells (`baseline`, `fail_1`, `fail_4`).
Each of the five native competitors has two (`fail_1`, `fail_4`), for 13
framework/condition cells total. With three repetitions, the publication suite
contains **39 condition-repetition executions**.

Cross-framework recovery is intended primarily to report completed-shard reuse,
rows recomputed, retry behavior, and final validity. A fixed 30-second injection
does not imply that every framework has completed the same fraction of useful
generation at failure time.

## Task-generalization experiments

### Text shortening

This is a CNN/DailyMail article-to-summary task, not the UltraChat rewrite task.
MMIRAGE, DataTrove, and NeMo Curator use the same summarization instruction and
a matched 128-token generation budget.

### VLM enrichment

This is a MedTrinity image/text enrichment task using
`Qwen/Qwen3-VL-4B-Instruct`. The current NeMo VLM integration is useful as an
integration/contract result but is row-sequential internally, so it should not
be interpreted as a fully tuned NeMo throughput ceiling.

## Direct SGLang versus endpoint-matched overhead

A-MATRIX `raw_sglang` evaluates a **complete Direct SGLang runner**. It compares
complete execution paths and is not an inference-engine-identical overhead
microbenchmark.

`experiments/raw_sglang_overhead` remains a separate endpoint-matched
experiment. It is the appropriate experiment for estimating MMIRAGE
abstraction/orchestration overhead relative to the same SGLang serving endpoint.

## Direct `run_setup.py` usage

Useful non-publication/manual commands include:

```bash
python experiments/a_matrix/scripts/run_setup.py \
  --setup gpu_scaling --serial --repetitions 3 --dry-run

python experiments/a_matrix/scripts/run_setup.py \
  --setup recovery --repetitions 3 --dry-run

python experiments/a_matrix/scripts/run_setup.py \
  --setup text_shortening --repetitions 3 --dry-run

python experiments/a_matrix/scripts/run_setup.py \
  --setup vlm_enrichment --repetitions 3 --dry-run
```

`run_setup.py` also supports `--prepare`, `--extract`, `--overwrite`, `--gpus`,
and the historical `--reuse-fastruns` mechanism. Do not use
`--reuse-fastruns` for the corrected publication benchmark.

## Environment

The suite uses separate competitor environments for DataTrove, NeMo Curator,
Distilabel, and Ray Data LLM. The publication drivers read their interpreter
paths from:

- `MMIRAGE_DATATROVE_PYTHON`;
- `MMIRAGE_NEMO_CURATOR_PYTHON`;
- `MMIRAGE_DISTILABEL_PYTHON`;
- `MMIRAGE_RAY_DATA_LLM_PYTHON`.

Python 3.12 requires `SETUPTOOLS_USE_DISTUTILS=local` for the affected vLLM
competitor environments.
