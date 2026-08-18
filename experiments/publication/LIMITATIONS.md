# Interpretation and limitations

The experiments are designed to compare complete data-construction paths while keeping workloads and generation budgets controlled. Several measurements should not be over-interpreted.

- Scaling's Direct SGLang condition is a complete runner/path baseline, not a pure measurement of framework abstraction overhead. Use the endpoint-matched SGLang experiment for that claim.
- Framework-native chat/prompt serialization can differ even when the semantic instruction and generation budget are matched. Cross-framework input-token counters are therefore not directly comparable.
- The A100 four-GPU point demonstrates portability of the same locked workload/model across accelerator classes. It is not a pure causal accelerator benchmark unless the rest of the host/software stack is also identical.
- Recovery uses a common external relaunch controller for native competitors. The 30-second failure time occurs at a similar lifecycle phase but not at identical useful-work progress. Cross-framework recovery claims should focus on successful reuse, recomputation and final validity rather than treating wall time as native recovery-system latency.
- The NeMo VLM integration is internally row-sequential in the retained implementation. Its result demonstrates integration and task completion, not an optimized NeMo VLM throughput ceiling.
- VLM paths use framework-specific multimodal serialization. Output token rates should not be interpreted as an exact serving-engine comparison.
- The benchmark evaluates coordinated data construction and augmentation. It is not a benchmark of federated or decentralized model training itself.

The refactor of this directory is mechanically checked against the frozen publication baseline with `publication/verify_refactor.py`; this protects executable semantics from accidental changes caused by moving files and rewriting orchestration.
