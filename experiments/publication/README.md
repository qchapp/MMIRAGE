# Publication reproduction

The publication suite is intentionally serialized. Do not run multiple timed stages concurrently on the same node or shared storage backend if the measurements will be compared directly.

## H100 suite

Requirements: exactly four visible H100 GPUs, the MMIRAGE/SGLang environment, competitor environments listed in `ENVIRONMENTS.md`, and a Hugging Face token. The driver prepares deterministic workloads and resolves exact model revisions before deleting prior publication outputs or entering timed regions.

```bash
bash experiments/publication/run_h100.sh --dry-run
bash experiments/publication/run_h100.sh
```

The H100 driver executes, in order:

1. clean-tracked-tree, environment and four-H100 preflight;
2. deterministic workload preparation for scaling/recovery, text shortening, VLM enrichment, and endpoint overhead;
3. exact model snapshot prefetch and revision lock;
4. publication manifest creation with Git commit, workload hashes, dataset revisions, model revisions, and hardware metadata;
5. offline timed H100 strong scaling;
6. the recovery matrix and evidence extraction;
7. text shortening;
8. VLM enrichment;
9. endpoint-matched SGLang overhead.

All reported experiment cells use three repetitions.

## A100 transfer point

Use the same refactor commit as the H100 run. The A100 node needs exactly four visible A100 GPUs. It must see the H100-produced directories below, either through shared storage or by copying them without modification:

```text
experiments/scaling/workload/
experiments/sglang_overhead/workload/
```

Then run:

```bash
bash experiments/publication/run_a100.sh --dry-run
bash experiments/publication/run_a100.sh
```

The A100 driver refuses to execute if the tracked checkout is dirty, or if the Git commit or workload hashes differ from the H100 manifest. It verifies that the current upstream model revision is still the exact H100 revision, downloads that exact snapshot, switches to offline mode, and runs only the four-GPU scaling transfer point plus the one-GPU endpoint-overhead experiment.

The A100 result is accelerator-portability evidence. Unless the CPU, storage, driver, networking and software stack are otherwise controlled to be identical, it should not be presented as a causal measurement of accelerator speed alone.

## Verification before packaging

Run the refactor verifier from the repository root:

```bash
python experiments/publication/verify_refactor.py
```

A publication artifact should not be packaged if the verifier reports any executable-logic, configuration, dependency, syntax, source-boundary, documentation-reference, or semantic-plan difference from the frozen baseline.

## Anonymized submission export

The development-only verifier intentionally records the exact source baseline branch and Git commit in `experiments/publication/_verification/`. Those identifiers can reveal the public repository even when `.git/` history is removed. For a double-blind submission, first run the verifier successfully on this branch, then create the anonymous source bundle without:

```text
.git/
experiments/publication/_verification/
experiments/publication/verify_refactor.py
```

Also exclude generated workloads/results and any machine-local logs or credentials. The scientific experiment configs, runners, publication drivers, protocol, environment notes and limitations do not depend on the removed verification metadata at runtime.

## Monitoring

```bash
python experiments/progress_tracker.py --suite h100
python experiments/progress_tracker.py --suite a100
```

The tracker never mutates experiment state.
