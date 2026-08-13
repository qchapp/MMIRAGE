# Reproduce The ChartQA AnonLib vs NeMo Curator Comparison

This folder defines a controlled comparison for an end-to-end multimodal ChartQA transformation. It does not launch the full H100 benchmark automatically.

## Scientific Scope

Question: given the same heterogeneous multimodal source dataset and nested target training-data schema, what framework-specific machinery is required to express and execute the transformation, and what runtime overhead is introduced under an as-matched-as-possible inference path?

Do not interpret the experiment as ChartQA accuracy evaluation or VLM quality evaluation. The reference answers are preserved as heterogeneous source labels and metadata.

The current NeMo inspection found one important mismatch: NeMo Curator `v1.3.0` officially integrates `data-designer==0.5.5`, while standalone Data Designer latest docs are `v0.9.1`. The executable NeMo baseline uses Curator's official integrated pin. See `VERSIONS.md`.

## Files

- `prepare_chartqa.py`: downloads the pinned ChartQA subset, writes JSONL, image assets, manifest, and checksums.
- `anonlib/chartqa_anonlib.yaml`: AnonLib transformation recipe.
- `anonlib/chartqa_custom.py`: deterministic query and generated-answer normalization.
- `anonlib/run_anonlib_with_openai_vision_endpoint.py`: benchmark-only adapter that forwards AnonLib SGLang calls to an external OpenAI-compatible VLM endpoint without modifying `src/`. Counted in the implementation footprint as AnonLib runner/glue.
- `nemo_curator/chartqa_pipeline.py`: idiomatic Curator pipeline using `JsonlReader`, `DataDesignerStage`, a minimal render stage, and `JsonlWriter`.
- `nemo_curator/data_designer_config.yaml`: inspectable Data Designer column graph. Python is still required for callable custom columns and Curator nested rendering.
- `inference/launch_sglang_server.sh`: shared SGLang server launcher.
- `run_comparison.py`: balanced 3-repetition runner.
- `analyze_results.py`: output validation, summaries, footprint counting (each framework's runner script, custom functions, and declarative config), setup-time reporting, and LaTeX table generation.
- `measure_setup.py`: times a fresh environment creation + dependency install per framework and records it for `analyze_results.py`.

Root wrappers are also available under `scripts/`.

## Environment Isolation

Create separate environments. Do not install NeMo Curator into the AnonLib environment.

AnonLib environment:

```bash
uv venv .venv-anonlib --python 3.12
source .venv-anonlib/bin/activate
uv pip install --prerelease=allow -r experiments/nemo_curator_comparison/environment/anonlib_uv_requirements.txt
python3 -c "import anonlib, sglang; print('anonlib/sglang ok')"
deactivate
```

NeMo Curator/Data Designer environment:

```bash
uv venv .venv-nemo --python 3.12
source .venv-nemo/bin/activate
# nemo-curator pins jieba==0.42.1, which still builds with distutils (removed in
# Python 3.12). setuptools 75.8.0 provides the shim; install without isolation.
uv pip install setuptools==75.8.0
SETUPTOOLS_USE_DISTUTILS=local uv pip install --no-build-isolation --extra-index-url https://pypi.nvidia.com -r experiments/nemo_curator_comparison/environment/nemo_curator_uv_requirements.txt
python3 -c "import nemo_curator, data_designer; print(nemo_curator.__version__)"
deactivate
```

## Setup Effort Measurement

`measure_setup.py` creates a throwaway venv and installs the pinned requirements
while timing venv creation and dependency installation. The venv is deleted
afterwards; only the JSON record under `setup_times/{framework}.json` is kept.
`analyze_results.py` reads these records and reports `setup_seconds` per
framework (LaTeX "Setup time" column). uv reuses its wheel cache, so numbers are
warm-cache fresh-venv installs on the same machine:

```bash
source .venv-anonlib/bin/activate
python3 scripts/measure_setup.py --framework anonlib
python3 scripts/measure_setup.py --framework nemo
deactivate
```

Shared SGLang server environment, if separate:

```bash
uv venv .venv-sglang --python 3.12
source .venv-sglang/bin/activate
uv pip install 'sglang==0.5.10' 'transformers==5.3.0' pillow
deactivate
```

## Prepare ChartQA

Use the pinned ChartQA revision recorded in `prepare_chartqa.py`:

```bash
source .venv-anonlib/bin/activate
python3 scripts/prepare_chartqa.py \
  --output-dir experiments/nemo_curator_comparison/workload/chartqa \
  --num-rows 1000 \
  --seed 20260813 \
  --image-format path
deactivate
```

For a small smoke test:

```bash
source .venv-anonlib/bin/activate
python3 scripts/prepare_chartqa.py \
  --output-dir experiments/nemo_curator_comparison/workload/chartqa_smoke \
  --num-rows 8 \
  --seed 20260813 \
  --image-format path
deactivate
```

## Start Shared Inference Server

Recommended model for one H100-80GB:

```bash
source .venv-sglang/bin/activate
export CHARTQA_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
export CHARTQA_SGLANG_PORT=30000
export CHARTQA_TP_SIZE=1
export CHARTQA_DTYPE=bfloat16
bash experiments/nemo_curator_comparison/inference/launch_sglang_server.sh
```

Record the exact model revision before full runs:

```bash
huggingface-cli model-info Qwen/Qwen2.5-VL-7B-Instruct --revision main
```

If the shared endpoint is not usable by one client, record the mismatch in the run notes and do not report throughput as pure framework overhead.

## Smoke Test Without GPU

Use the mock server only for schema and script validation. It is not a benchmark. The NeMo path can use a fully fake model name. The AnonLib path still constructs its normal tokenizer/chat-template machinery before the benchmark endpoint adapter is called, so use a reachable Hugging Face model ID for `--model`. Only the tokenizer files are fetched (the adapter never loads model weights), so the download is small.

Terminal 1:

```bash
python3 experiments/nemo_curator_comparison/inference/mock_openai_vlm_server.py --port 30080
```

Terminal 2, AnonLib smoke:

```bash
source .venv-anonlib/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py \
  --workload-jsonl experiments/nemo_curator_comparison/workload/chartqa_smoke/chartqa_subset.jsonl \
  --image-base-path experiments/nemo_curator_comparison/workload/chartqa_smoke \
  --output-root experiments/nemo_curator_comparison/results_smoke_anonlib \
  --order anonlib \
  --repetitions 1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --base-url http://127.0.0.1:30080/v1 \
  --batch-size 4 \
  --concurrency 4 \
  --max-tokens 32
deactivate
```

Terminal 2, NeMo smoke:

```bash
source .venv-nemo/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py \
  --workload-jsonl experiments/nemo_curator_comparison/workload/chartqa_smoke/chartqa_subset.jsonl \
  --image-base-path experiments/nemo_curator_comparison/workload/chartqa_smoke \
  --output-root experiments/nemo_curator_comparison/results_smoke_nemo \
  --order nemo \
  --repetitions 1 \
  --model mock-chartqa-vlm \
  --base-url http://127.0.0.1:30080/v1 \
  --batch-size 4 \
  --concurrency 4 \
  --max-tokens 32
deactivate
```

Analyze smoke outputs:

```bash
source .venv-anonlib/bin/activate
python3 scripts/analyze_nemo_curator_comparison.py \
  --results-root experiments/nemo_curator_comparison/results_smoke_anonlib \
  --expected-input-jsonl experiments/nemo_curator_comparison/workload/chartqa_smoke/chartqa_subset.jsonl \
  --output-dir experiments/nemo_curator_comparison/results_smoke_anonlib/analysis
deactivate
```

Run the same analysis with `results_smoke_nemo` after NeMo smoke.

## Full Three-Repetition Run

Run frameworks serially on the same one-H100 pod with the shared server already running. Because the two frameworks live in separate environments, invoke the runner for each framework in the appropriate environment while preserving the balanced order manually:

```bash
# AnonLib rep 1
source .venv-anonlib/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py --order anonlib --repetitions 1 --output-root experiments/nemo_curator_comparison/results --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:30000/v1
deactivate

# NeMo reps 1 and 2
source .venv-nemo/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py --order nemo,nemo --repetitions 2 --nemo-start-rep 1 --output-root experiments/nemo_curator_comparison/results --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:30000/v1
deactivate

# AnonLib reps 2 and 3
source .venv-anonlib/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py --order anonlib,anonlib --repetitions 2 --anonlib-start-rep 2 --output-root experiments/nemo_curator_comparison/results --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:30000/v1
deactivate

# NeMo rep 3
source .venv-nemo/bin/activate
python3 experiments/nemo_curator_comparison/run_comparison.py --order nemo --repetitions 1 --nemo-start-rep 3 --output-root experiments/nemo_curator_comparison/results --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:30000/v1
deactivate
```

If you have both stacks installed in separate environments, pass each interpreter explicitly so a single balanced command can drive both frameworks from one process:

```bash
python3 scripts/run_nemo_curator_comparison.py \
  --order anonlib,nemo,nemo,anonlib,anonlib,nemo \
  --repetitions 3 \
  --workload-jsonl experiments/nemo_curator_comparison/workload/chartqa/chartqa_subset.jsonl \
  --image-base-path experiments/nemo_curator_comparison/workload/chartqa \
  --output-root experiments/nemo_curator_comparison/results \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --base-url http://127.0.0.1:30000/v1 \
  --batch-size 64 \
  --concurrency 64 \
  --max-tokens 128 \
  --anonlib-python .venv-anonlib/bin/python \
  --nemo-python .venv-nemo/bin/python
```

Re-run any existing per-framework result directory with `--overwrite`. A dry run
(`--dry-run`) only prints the planned commands and creates no per-run directories
(it still records provenance `environment.json`/`run_comparison_summary.json` under the output root).

## Analyze Full Results

```bash
source .venv-anonlib/bin/activate
python3 scripts/analyze_nemo_curator_comparison.py \
  --results-root experiments/nemo_curator_comparison/results \
  --expected-input-jsonl experiments/nemo_curator_comparison/workload/chartqa/chartqa_subset.jsonl \
  --output-dir experiments/nemo_curator_comparison/results/analysis
deactivate
```

Generated files:

- `raw_results.csv`
- `summary.csv`
- `summary.json`
- `implementation_footprint.csv`
- `output_validation.csv`
- `latex_table.tex`

The condensed analysis under `results/analysis/` is committed as experiment evidence.
The workload and the raw per-repetition outputs are not tracked: the workload is
regenerated by `prepare_chartqa.py`, and the raw run directories are archived as a
separate results bundle (see "Results Archival" below).

## Results Archival

Raw per-repetition outputs (Arrow/JSONL shards, `status.json`, launcher logs,
GPU-utilization samples) are gitignored. Before distributing or writing up the
experiment, archive them alongside the analysis and workload manifest:

```bash
cd experiments/nemo_curator_comparison
python3 -m zipfile -c nemo_curator_comparison_results.zip \
  results/analysis results/environment.json results/run_comparison_summary.json \
  setup_times
python3 -m zipfile -c nemo_curator_comparison_results_full.zip \
  results results_smoke_anonlib results_smoke_nemo setup_times
```

The first bundle (analysis + provenance, a few KB) is enough for paper tables;
the second also carries the full raw run artifacts.

## Expected Runtime Estimate

With `Qwen/Qwen2.5-VL-7B-Instruct`, 1,000 ChartQA rows, `max_tokens=128`, and concurrency 64 on one H100-80GB, a recorded run measured ~70–90 s end-to-end per AnonLib repetition (3×~48K output tokens, ~550–1085 tok/s/GPU) and ~107–146 s end-to-end per NeMo Curator/Data Designer repetition. The exact numbers depend on SGLang multimodal batching, image preprocessing, response-format support, and whether Data Designer retries or throttles requests. Run the 8-row smoke test first, then a 32 or 128 row pilot if the 1,000-row benchmark appears too short or too long.

## Main Table Policy

Only include `tok/s/GPU` in the paper’s main comparison table if both frameworks successfully use the same SGLang OpenAI-compatible endpoint with the same model revision, tokenizer, image preprocessing path, temperature, top-p, max tokens, and concurrency. Otherwise put throughput in a caveated appendix table or omit it.
