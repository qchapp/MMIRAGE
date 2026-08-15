# RUNLOG — fast-run reproduction, 2026-08-15

- Pod: `mmirage-exp-4gpu-0-0` (Run:ai job `mmirage-exp-4gpu`, project `light-azgaoui`), node gpu101.rcp.epfl.ch, 4× H100 80GB, GPU ids 0–3.
- Repo: `/workspace/MMIRAGE`, branch `experiment/fast-runs`.
- HF cache on the persistent PVC (`/lightscratch/users/azgaoui/anonlib/hf`); `MMIRAGE_RECOVERY_ROOT=/workspace/mmirage-recovery`.
- Venvs: `.venv` (python 3.12, torch 2.9.1+cu129, sglang 0.5.10, flash-attn-4 4.0.0b15 / nvidia-cutlass-dsl 4.5.2 pinned per README), `.venv-datatrove` (datatrove 0.9.0 + vllm 0.23.0 + aiofiles), `.venv-nemo_curator` (nemo-curator 1.3.0 + vllm), each with `setuptools<76`.
- Raw result artifacts and run logs archived at `/lightscratch/users/azgaoui/anonlib/mmirage-fastrun-evidence-20260815/`.

## Fixes required on top of the committed runbooks

Environment/docs (see commit "Correct install runbooks after fresh-pod reproduction"):

- `uv pip install -e .` installs no GPU stack; the working sequence is CUDA torch
  first, then `--prerelease=allow -e ".[gpu]"`.
- A fresh resolve picks flash-attn-4 4.0.0b19; the README pins (b15 / cutlass-dsl
  4.5.2) must be applied before the first server start.
- `sgl-kernel` vs `sglang-kernel` both ship `deep_gemm/`; a mixed file set fails
  every SGLang server start with a deep_gemm AttributeError. Reinstall
  `sgl-kernel==0.3.21` last.
- MedTrinity-25M is gated (auto-approved); request access before running vlm cells.

Code (see commits "Fix native-competitor paths and calibrator robustness…" and
"Fix VLM native competitor runners…"):

- `raw_sglang_overhead/run.py` launched datatrove/nemo workers with `sys.executable`;
  added per-framework interpreter flags/env vars.
- datatrove venv lacked `aiofiles` (all saves failed as warnings → 0 rows,
  "successful" timing); nemo venv lacked `vllm`.
- nemo-curator 1.3.0 text stage: `name` must not be a `@property`; client base_url
  needs `/v1`.
- `native_shard_worker.py` now exits nonzero on 0 processed rows for non-empty input.
- `shard_recovery/run_local.py` now exits nonzero when shards fail that were not
  deliberately killed (previously an all-crashed baseline "passed" in 36.9 s).
- `run_smoke.py` merges `timing.json` on partial reruns; smoke measure commands
  pass `--overwrite`.
- VLM native path (`_shared/vlm_runners.py`) had never run end-to-end; five bugs:
  1. `PROJECT_ROOT` pointed at `experiments/` instead of the repo root — every
     sglang VLM shard worker crashed on `python -m experiments._shared.sglang_client`
     (same latent constant fixed in `native_frameworks.py`).
  2. `_contract_rows` dropped `source_index`, which the shard merge sorts on and
     validation compares — `row_order_matches_input` could never pass.
  3. The datatrove/nemo runners joined `image_base / image_path` and then passed the
     joined path to `image_data_url`, which joins the base again — every image
     lookup hit a doubled path.
  4. nemo `CaptionStage` defined `name` as a `@property` (forbidden in 1.3).
  5. nemo client base_url lacked `/v1`.
  None of these were catchable by the smoke suite, which measures only the MMIRAGE
  path — worth adding one native VLM smoke cell.

## Smoke + calibration (all five cells green after fixes)

| experiment | rows_smoke | wall (s) | load (s) | calibrated size | expected wall (s) |
|---|---|---|---|---|---|
| raw_sglang_overhead | 96 | 68.7 | 38.1 | 257 rows | 480.6 |
| single_node_h100_scaling | 1024 | 55.7 | 31.5 | 5430 rows | 480.0 |
| shard_recovery | 256 | 179.8 | — | 570 records | 600.7 (fail_8) |
| task_comparison/text_shortening | 1024 | 47.4 | 33.7 | 9471 rows | 480.0 |
| task_comparison/vlm_enrichment | 64 | 138.4 | 65.2 | 83 rows | 480.5 |

`over_budget: false`, `clamped: false` for every experiment on this pod.

## Experiments (literal README commands, calibrated sizes)

### 1. raw_sglang_overhead — 257 rows, 1 GPU, 1 repetition/path

throughput_retention **0.868** (relative orchestration overhead 0.132).

| path | wall (s) | generation (s) | model load (s) |
|---|---|---|---|
| raw_sglang | 83.9 | 32.2 | 36.0 |
| mmirage_sglang | 95.9 | 37.1 | 36.0 |
| datatrove | 77.5 | 30.1 | 42.0 |
| nemo_curator | 1183.3 | 1124.7 | 50.0 |

nemo_curator's client is row-sequential (~1 row/s/GPU), so its cell alone exceeds
the 480 s per-command budget; documented in `smoke/config.yaml` notes — the budget
covers the three concurrent-client paths.

### 2. single_node_h100_scaling — 5430 rows, 3 reps per GPU point

| GPUs | agg tok/s (mean±std) | steady-state tok/s | wall (s) | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 10937 ± 38 | 14623 | 126.8 | 1.00 | 1.000 |
| 2 | 16497 ± 69 | 26654 | 84.0 | 1.51 | 0.754 |
| 4 | 21771 ± 169 | 44985 | 63.7 | 1.99 | 0.498 |

End-to-end efficiency at 4 GPU is dominated by the fixed ~32 s model load on a
~64 s wall (fast-run artifact of the small workload); steady-state throughput
scales 14.6k → 26.7k → 45.0k tok/s (0.91 / 0.77 steady-state efficiency).

### 3. shard_recovery — 570 records, 16 shards, run_local.py

| condition | recomputed shards | dup IDs | missing IDs | outputs unchanged | wall (s) | extra wall (s) |
|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | — | 182.9 | 0.0 |
| fail_1 | 1 (shard 3) | 0 | 0 | true | 226.2 | 43.3 |
| fail_4 | 4 | 0 | 0 | true | 227.1 | 44.2 |
| fail_8 | 8 | 0 | 0 | true | 275.3 | 92.4 |

Recomputed-shard counts equal killed counts exactly, all 570 rows present in every
condition, order preserved, untouched shard outputs byte-identical.

### 4a. task_comparison/text_shortening — 9471 rows, 4 GPU, 3 reps/framework

cnn_dailymail, Qwen/Qwen3-4B. All 9 native reps `valid=true`, every integrity
check green (row counts, stable-ID sets, order, prompt sha256, schema).

| framework | wall (s), 3 reps | agg tok/s | rows/s | validation |
|---|---|---|---|---|
| MMIRAGE (4 GPU) | 90.8 ± 0.3 | 13349 ± 42 (steady-state 20945) | 104.3 | PASS ×3 |
| datatrove | 187.3 / 181.1 / 181.5 | ~13230 | ~52 | PASS ×3 |
| nemo_curator | 2734.9 / 2724.8 / 2738.8 | ~887 | ~3.5 | PASS ×3 |

nemo_curator's ~45 min/rep is the row-sequential client again (~20× the fast-run
budget); the literal README command was run to completion rather than cut.

### 4b. task_comparison/vlm_enrichment — 83 rows, 4 GPU, 3 reps/framework

MedTrinity-25M demo (gated, auto-approved), Qwen/Qwen3-VL-4B-Instruct.

| framework | wall (s), 3 reps | agg tok/s | validation |
|---|---|---|---|
| MMIRAGE (4 GPU) | 39.7 ± 0.7 | 253 (steady-state 876) | PASS ×3 |
| sglang native | 48.7 / 48.4 / 48.7 | ~1177 | PASS ×3 |
| datatrove native | 77.7 / 72.0 / 72.0 | ~805 | PASS ×3 |
| nemo_curator native | 217.6 / 178.3 / 177.7 | ~300 | PASS ×3 |

Token-throughput comparability caveat: the native runners count only final output
tokens on an 83-row workload where model load dominates the wall, so tok/s here is
not a scaling statement — the validation columns are the point of this experiment.
