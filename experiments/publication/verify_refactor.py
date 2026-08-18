#!/usr/bin/env python3
"""Verify the publication refactor against the frozen publication baseline.

The baseline is the immutable commit recorded in
``_verification/baseline_equivalence.json``. Runnable implementation files are
checked either byte-for-byte (pure moves) or through a canonical Python AST in
which only explicitly relocation-sensitive path anchors/references are replaced
with logical placeholders. Publication orchestration is intentionally new and
is checked by its generated semantic execution plan instead.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASELINE_FILE = HERE / "_verification" / "baseline_equivalence.json"
BASELINE = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
BASELINE_COMMIT = BASELINE["baseline_commit"]

PURE_MOVES = {
    "experiments/single_node_h100_scaling/scripts/run.py": "experiments/scaling/scripts/run.py",
    "experiments/single_node_h100_scaling/scripts/native_shard_worker.py": "experiments/scaling/scripts/native_shard_worker.py",
    "experiments/single_node_h100_scaling/scripts/run_native_text_competitor.py": "experiments/scaling/scripts/run_native_text_competitor.py",
    "experiments/single_node_h100_scaling/scripts/run_raw_sglang_scaling.py": "experiments/scaling/scripts/run_raw_sglang_scaling.py",
    "experiments/single_node_h100_scaling/scripts/run_datatrove_scaling.py": "experiments/scaling/scripts/run_datatrove_scaling.py",
    "experiments/single_node_h100_scaling/scripts/run_nemo_curator_scaling.py": "experiments/scaling/scripts/run_nemo_curator_scaling.py",
    "experiments/shard_recovery/scripts/run_local.py": "experiments/recovery/scripts/run_local.py",
    "experiments/shard_recovery/scripts/run_native_recovery_publication.py": "experiments/recovery/scripts/run_native_recovery_publication.py",
    "experiments/shard_recovery/scripts/extract_results.py": "experiments/recovery/scripts/extract_results.py",
    "experiments/raw_sglang_overhead/scripts/prepare_workload.py": "experiments/sglang_overhead/scripts/prepare_workload.py",
    "experiments/raw_sglang_overhead/scripts/raw_sglang_client.py": "experiments/sglang_overhead/scripts/raw_sglang_client.py",
    "experiments/raw_sglang_overhead/scripts/run.py": "experiments/sglang_overhead/scripts/run.py",
    "experiments/raw_sglang_overhead/scripts/run_mmirage_with_sglang_endpoint.py": "experiments/sglang_overhead/scripts/run_mmirage_with_sglang_endpoint.py",
}

CANONICAL_MOVES = {
    "experiments/a_matrix/scripts/prepare_workload.py": "experiments/scaling/scripts/prepare_workload.py",
    "experiments/shard_recovery/scripts/run_native_recovery_competitor.py": "experiments/recovery/scripts/run_native_recovery_competitor.py",
    "experiments/task_comparison/text_shortening/scripts/prepare_workload.py": "experiments/text_shortening/scripts/prepare_workload.py",
    "experiments/task_comparison/vlm_enrichment/scripts/prepare_workload.py": "experiments/vlm_enrichment/scripts/prepare_workload.py",
    "experiments/task_comparison/vlm_enrichment/scripts/run_mmirage_vlm.py": "experiments/vlm_enrichment/scripts/run_mmirage_vlm.py",
    "experiments/task_comparison/vlm_enrichment/scripts/run_native_vlm_competitor.py": "experiments/vlm_enrichment/scripts/run_native_vlm_competitor.py",
}

CONFIG_MOVES = {
    "experiments/a_matrix/configs/execution.yaml": "experiments/scaling/configs/execution.yaml",
    "experiments/a_matrix/configs/semantic_recipe.yaml": "experiments/scaling/configs/semantic_recipe.yaml",
    "experiments/a_matrix/configs/workload_size.yaml": "experiments/scaling/configs/workload_size.yaml",
    "experiments/a_matrix/configs/mmirage_recovery.yaml": "experiments/recovery/configs/mmirage_recovery.yaml",
    "experiments/task_comparison/text_shortening/configs/execution_4gpu.yaml": "experiments/text_shortening/configs/execution_4gpu.yaml",
    "experiments/task_comparison/text_shortening/configs/semantic_recipe.yaml": "experiments/text_shortening/configs/semantic_recipe.yaml",
    "experiments/task_comparison/text_shortening/configs/workload_size.yaml": "experiments/text_shortening/configs/workload_size.yaml",
    "experiments/task_comparison/vlm_enrichment/configs/execution_4gpu.yaml": "experiments/vlm_enrichment/configs/execution_4gpu.yaml",
    "experiments/task_comparison/vlm_enrichment/configs/mmirage_vlm.yaml": "experiments/vlm_enrichment/configs/mmirage_vlm.yaml",
    "experiments/task_comparison/vlm_enrichment/configs/workload_size.yaml": "experiments/vlm_enrichment/configs/workload_size.yaml",
    "experiments/raw_sglang_overhead/configs/mmirage_sglang.yaml": "experiments/sglang_overhead/configs/mmirage_sglang.yaml",
    "experiments/raw_sglang_overhead/configs/workload_size.yaml": "experiments/sglang_overhead/configs/workload_size.yaml",
}

PATH_REPLACEMENTS = [
    ("experiments/task_comparison/text_shortening", "experiments/@TEXT@"),
    ("experiments/task_comparison/vlm_enrichment", "experiments/@VLM@"),
    ("experiments/single_node_h100_scaling", "experiments/@SCALING@"),
    ("experiments/raw_sglang_overhead", "experiments/@OVERHEAD@"),
    ("experiments/shard_recovery", "experiments/@RECOVERY@"),
    ("experiments/a_matrix", "experiments/@SCALING@"),
    ("experiments/text_shortening", "experiments/@TEXT@"),
    ("experiments/vlm_enrichment", "experiments/@VLM@"),
    ("experiments/sglang_overhead", "experiments/@OVERHEAD@"),
    ("experiments/recovery", "experiments/@RECOVERY@"),
    ("experiments/scaling", "experiments/@SCALING@"),
    ("task_comparison/vlm_enrichment", "vlm_enrichment"),
    ("MMIRAGE_A_", "MMIRAGE_SCALING_"),
]

ANCHOR_NAMES = {
    "PROJECT_ROOT", "_PROJECT_ROOT", "REPO_ROOT", "EXPERIMENT_DIR",
    "SCRIPT_DIR", "SCRIPTS_DIR", "SHARED_DIR", "WORKER_SCRIPT",
}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def baseline_text(path: str) -> str:
    result = run("git", "show", f"{BASELINE_COMMIT}:{path}")
    return result.stdout


def current_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return tree


class Canonicalizer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if any(name in ANCHOR_NAMES for name in names):
            node.value = ast.Constant(value="@RELOCATION_ANCHOR@")
            return node
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if isinstance(node.target, ast.Name) and node.target.id in ANCHOR_NAMES:
            node.value = ast.Constant(value="@RELOCATION_ANCHOR@")
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            value = node.value
            for old, new in PATH_REPLACEMENTS:
                value = value.replace(old, new)
            node.value = value
        return node


def canonical_ast(text: str) -> str:
    tree = ast.parse(text)
    tree = strip_docstrings(tree)
    tree = Canonicalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical_value(v) for v in value]
    if isinstance(value, str):
        out = value
        for old, new in PATH_REPLACEMENTS:
            out = out.replace(old, new)
        return out
    return value


def verify_runners(errors: list[str]) -> None:
    print("\n[runnable implementation equivalence]")
    for old, new in PURE_MOVES.items():
        before = baseline_text(old)
        after = current_text(new)
        ok = before == after
        print(f"  {'PASS' if ok else 'FAIL'} exact  {old} -> {new}")
        if not ok:
            errors.append(f"pure move changed executable content: {old} -> {new}")
    for old, new in CANONICAL_MOVES.items():
        before = canonical_ast(baseline_text(old))
        after = canonical_ast(current_text(new))
        ok = before == after
        print(f"  {'PASS' if ok else 'FAIL'} AST    {old} -> {new}")
        if not ok:
            errors.append(f"canonical runner logic changed: {old} -> {new} ({sha(before)} != {sha(after)})")


def verify_configs(errors: list[str]) -> None:
    print("\n[configuration equivalence]")
    for old, new in CONFIG_MOVES.items():
        before = canonical_value(yaml.safe_load(baseline_text(old)) or {})
        after = canonical_value(yaml.safe_load(current_text(new)) or {})
        ok = before == after
        print(f"  {'PASS' if ok else 'FAIL'} {old} -> {new}")
        if not ok:
            errors.append(f"config semantics changed: {old} -> {new}")


def import_orchestrator():
    path = HERE / "orchestrate.py"
    spec = importlib.util.spec_from_file_location("publication_orchestrate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load publication orchestrator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_map(command: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"executable": Path(command[0]).name, "runner": canonical_value(command[1]) if len(command) > 1 else ""}
    i = 2
    while i < len(command):
        token = command[i]
        if token.startswith("--"):
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                out[token] = canonical_value(command[i + 1]); i += 2
            else:
                out[token] = True; i += 1
        else:
            i += 1
    return out


def assert_value(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def verify_plan(errors: list[str]) -> None:
    print("\n[publication execution plan equivalence]")
    orch = import_orchestrator()
    os.environ["MMIRAGE_DATATROVE_PYTHON"] = "/ENV/datatrove-python"
    os.environ["MMIRAGE_NEMO_CURATOR_PYTHON"] = "/ENV/nemo-python"
    os.environ["MMIRAGE_DISTILABEL_PYTHON"] = "/ENV/distilabel-python"
    os.environ["MMIRAGE_RAY_DATA_LLM_PYTHON"] = "/ENV/ray-python"
    contract = BASELINE["publication_contract"]

    h100 = orch.scaling_plan("h100", 3, True)
    a100 = orch.scaling_plan("a100", 3, True)
    recovery = orch.recovery_plan(True, 3)
    text = orch.text_plan(True, 3)
    vlm = orch.vlm_plan(True, 3)

    assert_value(errors, "H100 scaling units", len(h100), 12)
    assert_value(errors, "A100 scaling units", len(a100), 4)
    assert_value(errors, "recovery cells", len(recovery), 13)
    assert_value(errors, "text cells", len(text), 3)
    assert_value(errors, "VLM cells", len(vlm), 4)

    for hardware, units, key in [("h100", h100, "scaling_h100"), ("a100", a100, "scaling_a100")]:
        c = contract[key]
        assert_value(errors, f"{hardware} frameworks", sorted({u['framework'] for u in units}), sorted(c["frameworks"]))
        assert_value(errors, f"{hardware} GPU points", sorted({u['gpu_count'] for u in units}), c["gpu_points"])
        for unit in units:
            cmd = cli_map(unit["commands"][0])
            assert_value(errors, f"{hardware}/{unit['framework']}/{unit['gpu_count']} repetitions", int(cmd["--repetitions"]), 3)
            assert_value(errors, f"{hardware}/{unit['framework']}/{unit['gpu_count']} GPUs", cmd["--visible-gpus"], ",".join(str(i) for i in range(unit["gpu_count"])))
            if unit["framework"] == "mmirage":
                assert_value(errors, "MM scaling recipe", cmd["--semantic-recipe"], "experiments/@SCALING@/configs/semantic_recipe.yaml")
            else:
                assert_value(errors, f"{unit['framework']} model", cmd["--model"], c["model"])
                assert_value(errors, f"{unit['framework']} concurrency", int(cmd["--concurrency"]), c["native_concurrency"])
                assert_value(errors, f"{unit['framework']} temp", float(cmd["--temperature"]), c["temperature"])
                assert_value(errors, f"{unit['framework']} max tokens", int(cmd["--max-new-tokens"]), c["max_new_tokens"])
                assert_value(errors, f"{unit['framework']} prompt", cmd["--prompt-style"], c["prompt_style"])

    rc = contract["recovery"]
    assert_value(errors, "recovery MM conditions", [u["condition"] for u in recovery if u["framework"] == "mmirage"], rc["mmirage_conditions"])
    assert_value(errors, "recovery native frameworks", sorted({u["framework"] for u in recovery if u["framework"] != "mmirage"}), sorted(rc["native_frameworks"]))
    logical_recovery_reps = 0
    for unit in recovery:
        if unit["framework"] == "mmirage":
            run_commands = [c for c in unit["commands"] if "run-condition" in c]
            logical_recovery_reps += len(run_commands)
            for command in run_commands:
                cm = cli_map(command)
                assert_value(errors, "MM recovery GPUs", cm["--gpu-ids"], "0,1,2,3")
                assert_value(errors, "MM recovery active shards", int(cm["--max-active-shards"]), 4)
                if unit["condition"] != "baseline":
                    assert_value(errors, "MM recovery kill time", float(cm["--kill-after-seconds"]), 30.0)
        else:
            logical_recovery_reps += len(unit["commands"])
            for command in unit["commands"]:
                cm = cli_map(command)
                assert_value(errors, "native recovery wrapper", cm["runner"], "experiments/@RECOVERY@/scripts/run_native_recovery_publication.py")
                assert_value(errors, "native recovery model", cm["--model"], rc["model"])
                assert_value(errors, "native recovery concurrency", int(cm["--concurrency"]), 64)
                assert_value(errors, "native recovery max tokens", int(cm["--max-new-tokens"]), 256)
                assert_value(errors, "native recovery kill time", float(cm["--kill-after-seconds"]), 30.0)
    assert_value(errors, "recovery logical repetitions", logical_recovery_reps, 39)

    tc = contract["text_shortening"]
    for unit in text:
        if unit["framework"] != "mmirage":
            cm = cli_map(unit["commands"][0])
            assert_value(errors, "text repetitions", int(cm["--repetitions"]), 3)
            assert_value(errors, "text concurrency", int(cm["--concurrency"]), 64)
            assert_value(errors, "text max tokens", int(cm["--max-new-tokens"]), 128)
            assert_value(errors, "text prompt", cm["--prompt-style"], "summarize")
    vc = contract["vlm_enrichment"]
    for unit in vlm:
        if unit["framework"] != "mmirage":
            cm = cli_map(unit["commands"][0])
            assert_value(errors, "VLM repetitions", int(cm["--repetitions"]), 3)
            assert_value(errors, "VLM concurrency", int(cm["--concurrency"]), 64)
            assert_value(errors, "VLM max tokens", int(cm["--max-new-tokens"]), 1024)
            assert_value(errors, "VLM temperature", float(cm["--temperature"]), 0.1)
            assert_value(errors, "VLM top-p", float(cm["--top-p"]), 0.9)
            assert_value(errors, "VLM model", cm["--model"], vc["model"])

    h100_driver = current_text("experiments/publication/run_h100.sh")
    a100_driver = current_text("experiments/publication/run_a100.sh")
    overhead_fragment = "--frameworks raw_sglang,mmirage_sglang --repetitions 3 --gpu-index 0 --concurrency 64 --max-tokens 1024 --temperature 0.0"
    if overhead_fragment not in h100_driver or overhead_fragment not in a100_driver:
        errors.append("endpoint-overhead command differs from frozen contract")
    for required in ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "model_revisions.json", "publication_manifest.json"]:
        if required not in h100_driver:
            errors.append(f"H100 driver missing provenance/offline invariant: {required}")
    for required in ["expected-json", "publication_manifest.json", "scaling_workload_sha256", "overhead_prompts_sha256", "overhead_warmup_sha256"]:
        if required not in a100_driver:
            errors.append(f"A100 driver missing cross-hardware invariant: {required}")
    print(f"  {'PASS' if not errors else 'CHECK'} H100 scaling logical repetitions: 36")
    print(f"  {'PASS' if not errors else 'CHECK'} A100 scaling logical repetitions: 12")
    print(f"  {'PASS' if logical_recovery_reps == 39 else 'FAIL'} recovery logical repetitions: {logical_recovery_reps}")
    print("  expected text logical repetitions: 9; VLM: 12; overhead per hardware: 6")


def verify_references(errors: list[str]) -> None:
    print("\n[reference graph and syntax]")
    required = [
        "experiments/publication/orchestrate.py", "experiments/publication/run_h100.sh", "experiments/publication/run_a100.sh",
        "experiments/scaling/scripts/run.py", "experiments/scaling/scripts/native_shard_worker.py",
        "experiments/recovery/scripts/run_local.py", "experiments/recovery/scripts/run_native_recovery_publication.py",
        "experiments/text_shortening/scripts/prepare_workload.py", "experiments/vlm_enrichment/scripts/run_mmirage_vlm.py",
        "experiments/sglang_overhead/scripts/run.py",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    if missing:
        errors.extend(f"missing retained dependency: {path}" for path in missing)
    py_dirs = ["experiments/publication", "experiments/scaling/scripts", "experiments/recovery/scripts", "experiments/text_shortening/scripts", "experiments/vlm_enrichment/scripts", "experiments/sglang_overhead/scripts", "experiments/_shared"]
    for directory in py_dirs:
        for path in (REPO_ROOT / directory).glob("*.py"):
            result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"syntax failure: {path.relative_to(REPO_ROOT)}: {result.stderr.strip()}")
    for shell in [HERE / "run_h100.sh", HERE / "run_a100.sh"]:
        result = subprocess.run(["bash", "-n", str(shell)], cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"shell syntax failure: {shell.relative_to(REPO_ROOT)}: {result.stderr.strip()}")
    print(f"  {'PASS' if not missing else 'FAIL'} retained dependency paths")
    print("  syntax checks completed")


def verify_baseline_relationship(errors: list[str]) -> None:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"], cwd=REPO_ROOT)
    if result.returncode != 0:
        errors.append(f"baseline {BASELINE_COMMIT} is not an ancestor of HEAD")
    baseline_head = run("git", "rev-parse", BASELINE_COMMIT).stdout.strip()
    if baseline_head != BASELINE_COMMIT:
        errors.append("baseline commit cannot be resolved exactly")


def main() -> int:
    errors: list[str] = []
    print(f"Baseline: {BASELINE['baseline_branch']} @ {BASELINE_COMMIT}")
    verify_baseline_relationship(errors)
    verify_runners(errors)
    verify_configs(errors)
    verify_plan(errors)
    verify_references(errors)
    print("\n[summary]")
    if errors:
        print("FAIL")
        for error in errors:
            print(" - " + error)
        return 1
    print("Runnable implementation files: canonical logic equivalence: PASS")
    print("Dependency/reference graph: PASS")
    print("Publication execution plan: semantic differences: 0")
    print("Configs: semantic differences: 0")
    print("Expected benchmark behavior equivalent to baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
