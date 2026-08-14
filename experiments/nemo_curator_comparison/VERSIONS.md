# Version Pins And Capability Notes

This experiment was prepared against the current public documentation and source snapshots available on 2026-08-13.

## Primary Comparison Stack

- MMIRAGE repository commit: captured at run time with `git rev-parse HEAD`.
- NeMo Curator documentation: `latest`, listed as `v1.3.0 (26.07)` in `https://docs.nvidia.com/nemo/curator/llms.txt`.
- NeMo Curator source/tag: `v1.3.0`, commit `6b956ce8965820de1b638fedf6de0cbcf0cc46ba`.
- NeMo Curator integrated Data Designer dependency: `data-designer==0.5.5`, from Curator `v1.3.0` `pyproject.toml` `sdg_cpu` extra.
- Integrated Data Designer source/tag: `v0.5.5`, commit `d43ac1cb2ec769f0682bed68c27ee0216f582f55`.

## Current Standalone Data Designer Reference

- Data Designer documentation: `latest`, lists `v0.9.1` as the latest version in `https://docs.nvidia.com/nemo/datadesigner/llms.txt`.
- Data Designer source/tag: `v0.9.1`, commit `27acf141170eceb1e8242c132d56b49107462fce`.
- Important mismatch: Curator `v1.3.0` does not pin `data-designer==0.9.1` in its official `sdg_*` extras. The comparison implementation therefore uses Curator's official integrated Data Designer version by default, not standalone latest.

## Verified Capability Summary

- Curator integrates Data Designer through `nemo_curator.stages.synthetic.nemo_data_designer.data_designer.DataDesignerStage`.
- `DataDesignerStage` accepts either a `DataDesignerConfigBuilder` or a YAML/JSON config file path, converts the incoming Curator `DocumentBatch` to a Data Designer seed dataset with `DataFrameSeedSource`, calls `DataDesigner.preview(..., num_records=batch.num_items)`, and returns the enriched dataframe as a new `DocumentBatch`.
- Data Designer supports seed datasets, expression columns, LLM text columns, structured JSON columns, custom Python columns, dependency ordering through column references, image multimodal context, OpenAI-compatible providers, and analysis statistics.
- Curator `v1.3.0` documents a local OpenAI-compatible inference server via Ray Serve or Dynamo/vLLM. For this comparison we use an externally managed SGLang OpenAI-compatible endpoint for both systems where possible, to avoid changing either production codebase.
- Current standalone Data Designer `v0.9.1` expands multimodal context to image/audio/video. Curator's officially pinned `data-designer==0.5.5` supports image context relevant to ChartQA.
- Data Designer structured generation produces structured column values, but Curator's integrated path naturally returns an enriched dataframe. Rendering an arbitrary nested training-data JSONL record is therefore implemented as a small Curator `ProcessingStage` after `DataDesignerStage`, rather than claiming Data Designer alone is a final nested-schema renderer.

## Scope of Interpretation

NeMo Curator/Data Designer supports declarative multimodal transformation. This comparison measures the framework-specific machinery required for one defined workload: source-field mapping, dependent custom processing, VLM generation, post-processing, nested record rendering, materialization, and instrumentation. Its results do not establish a general expressiveness difference between the frameworks.
