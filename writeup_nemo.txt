You are revising the AnonLib paper to integrate a controlled LLM-only comparison against
NVIDIA NeMo Curator v1.3.0 with its integrated Data Designer v0.5.5.

Evidence bundle:
`experiments/nemo_curator_comparison/nemo_curator_comparison_results.zip`

Use these files as authoritative:
- results/analysis/summary.json and summary.csv: mean +/- sample SD over 3 repetitions.
- results/analysis/raw_results.csv: all six per-repetition measurements.
- results/analysis/output_validation.csv: validity and normalization-consistency counts.
- results/analysis/implementation_footprint.csv: nonblank/noncomment LOC and file counts.
- results/analysis/latex_table.tex: generated compact table (reformat to booktabs).
- results/environment.json and results/run_comparison_summary.json: provenance and launcher timing.
- setup_times/*.json: warm-cache fresh-venv install timing.

Experiment design:
- Same pinned 1,000-row ChartQA subset (seed 20260813) and image assets.
- One structured multimodal LLM call per row; no deterministic Python normalization on either
  framework. The model emits answer, normalized_query, generated_answer_normalized, rationale.
- Shared externally managed SGLang OpenAI-compatible endpoint serving
  Qwen/Qwen2.5-VL-7B-Instruct with tp_size=1 on one H100-80GB.
- temperature=0, top_p=1, max_tokens=256, batch/concurrency=64.
- Balanced order: anonlib,nemo,nemo,anonlib,anonlib,nemo; 3 repetitions each.
- Isolated framework environments. Setup measurements reuse the recorded warm wheel cache.

Verified final results (mean +/- SD):
- Structural validity: both frameworks materialized 1000/1000 schema-valid rows, in source
  order, with zero missing or duplicate IDs in all six repetitions.
- End-to-end time: AnonLib 142.66 +/- 0.56 s; NeMo 153.94 +/- 0.26 s.
- Rows/s: AnonLib 7.01 +/- 0.03; NeMo 6.50 +/- 0.01.
- AnonLib output throughput: 564.99 +/- 2.39 tok/s/GPU from 79,915.33 +/- 8.50 output
  tokens per repetition. NeMo is N/A because Data Designer does not expose equivalent token
  usage; do not present token throughput as a framework-to-framework metric.
- Launcher-inclusive time: AnonLib 151.64 +/- 1.04 s; NeMo 156.39 +/- 0.30 s.
- Setup time: AnonLib 120.82 s; NeMo 34.62 s (warm-cache fresh-venv uv install).
- Footprint: AnonLib 81 declarative LOC + 158 glue Python LOC; NeMo 45 + 180; 2 counted
  files each; pipeline components 4 vs 5. LOC is a caveated secondary metric because configs
  and harnesses were adapted from framework examples rather than authored from scratch.

Critical normalization result:
- Exact query whitespace-normalization: AnonLib 168/1000 in every repetition; NeMo
  296.67 +/- 3.79/1000.
- Exact lowercase + whitespace answer normalization: AnonLib 983/1000 in every repetition;
  NeMo 966.67 +/- 1.53/1000.
- The LLM often changed case, punctuation, symbols, units, number forms, or wording despite
  explicit mechanical instructions. Therefore this recipe is structurally equivalent across
  frameworks but is NOT behaviorally equivalent to deterministic normalization. Say this
  prominently; do not claim that an LLM substitutes reliably for deterministic preprocessing.
- A preliminary matched run at max_tokens=128 caused NeMo to drop 13-17 rows/repetition when
  prompted fenced JSON was truncated before all four required fields. The final controlled run
  used 256 for both frameworks and achieved 1000/1000. Do not mix preliminary numbers into the
  main table.

Recommended paper integration:
- Add an RQ4 subsection after RQ3 titled approximately “AnonLib vs. NeMo Curator/Data
  Designer on an LLM-only multimodal transformation.”
- Main table: Framework | Valid rows | End-to-end time | Rows/s. Optionally include AnonLib
  tok/s/GPU with NeMo marked N/A and a strong non-comparability footnote.
- Secondary/appended table: Declarative LOC | Glue Python LOC | Pipeline components | Setup.
- Report normalization-consistency rates in text or a compact secondary table; these are a
  substantive negative result about LLM-only refactoring, not framework correctness.
- Do not add box plots for n=3. A table with mean +/- SD is sufficient.
- Update the limitation that says no like-for-like framework comparison exists, but retain
  limitations about one machine/server, n=3, workload specificity, and client-path differences.

Required caveats:
1. This evaluates orchestration and structural materialization, not ChartQA answer accuracy.
2. NeMo uses its officially integrated older Data Designer pin (0.5.5), not standalone latest.
3. The clients obtain structured output differently: AnonLib sends server-side JSON schema;
   Data Designer prompts for fenced JSON and validates it client-side. This is real framework
   behavior but means the inference paths are not byte-identical.
4. Timers differ: framework-internal e2e is the primary metric; launcher-inclusive values are
   reported separately. AnonLib token throughput excludes about 1.2 s tokenizer loading.
5. GPU-utilization samples averaged all four pod GPUs although only one served inference;
   omit them from the main table.
6. Setup timing is warm-cache and includes AnonLib's SGLang dependency install.
7. Never compare 564.99 tok/s/GPU directly with the paper's decode-bound text-only benchmark.

Ground rules:
- Never fabricate numbers; use the bundle's CSV/JSON evidence.
- Do not describe AnonLib as generally superior: it was about 7.9% faster in this workload,
  while NeMo installed much faster and used fewer declarative lines.
- Keep validity (schema/materialization) distinct from normalization consistency and answer
  quality.
