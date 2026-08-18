#!/usr/bin/env python3
"""Prefetch publication models before timed runs and lock them to exact revisions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--expected-json", default=None,
                   help="Optional JSON mapping repo id -> expected commit SHA.")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    expected = {}
    if args.expected_json:
        expected = json.loads(Path(args.expected_json).read_text(encoding="utf-8"))
    api = HfApi(token=token)
    resolved = {}
    for repo_id in args.models:
        info = api.model_info(repo_id=repo_id, revision="main", token=token)
        sha = str(info.sha)
        expected_sha = expected.get(repo_id)
        if expected_sha and sha != expected_sha:
            raise SystemExit(
                f"FATAL: {repo_id} main moved: expected {expected_sha}, current {sha}. "
                "Refusing a cross-hardware run with a different model revision."
            )
        exact_path = Path(snapshot_download(repo_id=repo_id, revision=sha, token=token))
        main_path = Path(snapshot_download(repo_id=repo_id, revision="main", token=token))
        if exact_path.name != sha or main_path.name != sha:
            raise SystemExit(
                f"FATAL: cache resolution mismatch for {repo_id}: "
                f"exact={exact_path.name}, main={main_path.name}, expected={sha}"
            )
        resolved[repo_id] = sha
        print(f"model-cache PASS: {repo_id}@{sha} -> {main_path}")
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
