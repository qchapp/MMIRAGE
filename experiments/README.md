# MMIRAGE publication experiments

This directory contains the experiments used for the publication evaluation of MMIRAGE. The layout follows the scientific questions directly; generated workloads and results are not source code and should remain untracked.

| Experiment | Purpose | Hardware |
|---|---|---|
| [`scaling/`](scaling/) | UltraChat rewrite throughput and strong scaling against Direct SGLang, DataTrove, and NeMo Curator | 1/2/4× H100; 4× A100 transfer point |
| [`recovery/`](recovery/) | deterministic shard recovery and recomputation after injected failures | 4× H100 |
| [`text_shortening/`](text_shortening/) | text-task generalization on CNN/DailyMail summarization | 4× H100 |
| [`vlm_enrichment/`](vlm_enrichment/) | multimodal enrichment on MedTrinity | 4× H100 |
| [`sglang_overhead/`](sglang_overhead/) | endpoint-matched MMIRAGE abstraction overhead relative to the same SGLang serving path | 1× H100 and 1× A100 |

The canonical unattended entry points are:

```bash
bash experiments/publication/run_h100.sh --dry-run
bash experiments/publication/run_h100.sh

bash experiments/publication/run_a100.sh --dry-run
bash experiments/publication/run_a100.sh
```

The A100 run reuses the exact scaling and endpoint-overhead workloads produced by the H100 run and verifies workload hashes, model revision, and Git commit before executing.

See [`publication/README.md`](publication/README.md) for the complete reproduction procedure, [`publication/PROTOCOL.md`](publication/PROTOCOL.md) for the comparison contract, and [`publication/LIMITATIONS.md`](publication/LIMITATIONS.md) for interpretation constraints.

## Monitoring

`progress_tracker.py` is read-only and can attach after an unattended run has already started:

```bash
python experiments/progress_tracker.py
python experiments/progress_tracker.py --suite h100 --once
python experiments/progress_tracker.py --suite a100 --json --once
```

Its time estimate is explicitly prior-based; the durable repetition counts are the authoritative progress signal.

## Refactor safety

The publication layout was refactored from the exact baseline recorded in `publication/_verification/baseline_equivalence.json`. Run:

```bash
python experiments/publication/verify_refactor.py
```

before packaging the artifact. The verifier checks byte-identical pure moves, canonical AST equivalence for runners whose only permitted changes are relocation-sensitive references, configuration equivalence, dependency resolution, syntax, and the semantic publication execution plan.
