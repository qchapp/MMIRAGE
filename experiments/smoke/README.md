# smoke

Calibrates the fast-run workload sizes on the pod that will actually run the
experiments. The pod agent runs this before the experiments (see
`AGENT_PROMPT.md` for the full runbook).

```
python experiments/smoke/run_smoke.py          # time one tiny cell per experiment
python experiments/smoke/calibrate.py --apply  # write configs/workload_size.yaml per experiment
```

- `config.yaml` — per-experiment smoke rows, budget, cell composition, timing sources.
- `run_smoke.py` — prepares a tiny workload and times the primary run command.
- `calibrate.py` — sizes each workload so its expected full wall stays under the
  per-command budget; `--apply` writes the size configs.
- `AGENT_PROMPT.md` — the pod-side agent instructions (paste into the pod agent).

Committed defaults in each experiment's `configs/workload_size.yaml` are safe
fallbacks used only if calibration is skipped. `timing.json` and
`calibration.json` record the smoke evidence; the pod agent commits them to the
branch after a run.
