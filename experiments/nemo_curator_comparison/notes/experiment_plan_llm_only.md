EXPERIMENT PLAN (Plan A) — LLM-only variant of the AnonLib vs NeMo Curator/Data Designer comparison
==================================================================================================
Status: IMPLEMENTED PROTOCOL. Run the scripts in this experiment directory to generate results.

## Goal
Make the current comparison rely only on LLM-based refactoring of datasets, so that both
frameworks execute identical per-row work and the implementation-footprint comparison is
fair. No deterministic Python normalization on either side.

## Motivation
The committed experiment (branch experiment/nemo-curator-comparison) uses two deterministic
Python functions on BOTH sides (normalize_query, normalize_generated_answer). The footprint
comparison is therefore unfair: AnonLib's copy of this "run a Python function" capability is
a ~100-line benchmark-harness CustomProcessor inside the endpoint adapter, while NeMo gets
the same capability natively via Data Designer CustomColumnConfig. The LLM-only variant
removes custom functions entirely: a single structured LLM call produces all fields, so the
two frameworks differ only in framework machinery (which is what the benchmark should
measure). It also sidesteps the related LOC criticism that configs are reused/adapted from
examples rather than authored line-by-line: with both configs shrunk to near-identical tiny
declarative specs, LOC stops being a differentiator, reinforcing that runtime, validity, and
setup are the real comparison dimensions.

## Recipe change
- AnonLib config: configs/anonlib_chartqa.yaml — remove the custom_pre (normalize_query) and custom_post
  (normalize_generated_answer) processors plus their outputs and fallback blocks.
- AnonLib runner: scripts/run_anonlib_with_openai_vision_endpoint.py — remove the CustomProcessor stack
  (CustomProcessorConfig/CustomOutputVar/CustomPre/CustomPost classes, CustomProcessor class,
  and the custom_pre/custom_post registrations in patch_sglang_engine()).
- NeMo runner: scripts/run_nemo_curator_pipeline.py — remove the two CustomColumnConfig columns
  (generator_function=normalize_query / normalize_generated_answer).

## New recipe (what the LLM produces)
One structured LLM call per row returns four string fields:
- answer: the answer to the question
- rationale: one short sentence explaining the chart evidence
- normalized_query: the question, whitespace-normalized (LLM-performed)
- generated_answer_normalized: the answer, lowercased and whitespace-normalized (LLM-performed)

This keeps the exact nested training-data output schema, so scripts/analyze_results.py validity checks
are unchanged (metadata.{reference_answer, generated_answer_normalized, source};
messages[0].content = image + text).

### AnonLib changes
1. configs/anonlib_chartqa.yaml:
   - remove the custom_pre and custom_post processor blocks (and their fallback blocks)
   - llm output: output_schema -> answer, rationale, normalized_query,
     generated_answer_normalized (all str)
   - prompt: use the raw "{{ query }}" as the Question; instruct the model to also return the
     two normalized fields
   - processing_params.outputs: keep only vlm_result (type llm, output_type JSON)
   - output_schema: user text = {{ vlm_result.normalized_query }};
     metadata.generated_answer_normalized = {{ vlm_result.generated_answer_normalized }}
2. scripts/run_anonlib_with_openai_vision_endpoint.py:
   - delete CustomProcessorConfig, CustomOutputVar, CustomPreProcessorConfig,
     CustomPostProcessorConfig, CustomPreOutputVar, CustomPostOutputVar, CustomProcessor,
     and the custom_pre/custom_post registrations in patch_sglang_engine()
   - keep EndpointEngine (incl. json_schema/response_format handling) + patch_sglang_engine
     (llm-only) + main()
3. do not include a separate deterministic normalization module

### NeMo changes
1. scripts/run_nemo_curator_pipeline.py:
   - VLMResult: add normalized_query and generated_answer_normalized (all str)
   - build_config: remove the two CustomColumnConfig entries; vlm_result prompt uses raw
     "{{ query }}"
   - RenderNestedChartQAStage: read the four fields from row["vlm_result"]; drop the
     normalized_query / generated_answer_normalized DataFrame columns from the required set
2. configs/nemo_data_designer.yaml: omit the two custom columns; output_format gains
   the two fields
3. do not include a separate deterministic normalization module

## Analysis checks
`scripts/analyze_results.py` reports, rather than asserts, exact mechanical normalization:
- whether query text equals the whitespace-collapsed source query;
- whether generated_answer_normalized equals the lowercase, whitespace-collapsed answer.

These metrics test whether the prompted LLM behavior matches deterministic normalization.
Do not claim behavioral equivalence unless the generated evidence supports it.

## Reproduction command
Use the balanced run:
   --order anonlib,nemo,nemo,anonlib,anonlib,nemo --repetitions 3 --overwrite
   --output-root experiments/nemo_curator_comparison/results
   --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:30000/v1
   --anonlib-python .venv/bin/python --nemo-python .venv-nemo/bin/python
   --batch-size 64 --concurrency 64 --max-tokens 256

Data Designer runtime seeds the input DataFrame, which makes `{{ query }}` referenceable.
Standalone `--dry-run-validate` lacks that seed and reports a spurious INVALID_REFERENCE.
The mock returns plain JSON when `response_format` is requested (AnonLib) and fenced JSON
otherwise (Data Designer), matching the two real client paths.
