#!/usr/bin/env python3
"""Run unmodified MMIRAGE while forwarding sglang.Engine calls to HTTP SGLang."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


class EndpointEngine:
    """Small stand-in for sglang.Engine used only by this benchmark process."""

    def __init__(self, **kwargs: Any) -> None:
        self.model_path = kwargs.get("model_path") or os.environ["MMIRAGE_OVERHEAD_MODEL_PATH"]
        self.base_url = os.environ["MMIRAGE_OVERHEAD_SGLANG_BASE_URL"].rstrip("/")
        self.timeout_seconds = int(os.environ.get("MMIRAGE_OVERHEAD_TIMEOUT_SECONDS", "900"))
        self.max_workers = max(1, int(os.environ.get("MMIRAGE_OVERHEAD_CONCURRENCY", "64")))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
        )

    def generate(
        self,
        prompt: Iterable[str] | str,
        sampling_params: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        params = dict(sampling_params or {})
        if "max_new_tokens" in params and "max_tokens" not in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        params.pop("json_schema", None)
        if self.max_workers == 1 or len(prompts) <= 1:
            return [self._generate_one(item, params) for item in prompts]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._generate_one, item, params) for item in prompts]
            return [future.result() for future in futures]

    def shutdown(self) -> None:
        return None

    def _generate_one(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"model": self.model_path, "prompt": prompt, **params}
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang HTTP {exc.code}: {error_body}") from exc
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"SGLang response has no choices: {body!r}")
        text = choices[0].get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        return {
            "text": text,
            "meta_info": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": len(self.tokenizer.encode(text)),
            },
        }


def patch_sglang_engine() -> None:
    import sglang

    sglang.Engine = EndpointEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_sglang_engine()

    from mmirage.shard_process import main as shard_main

    os.environ["MMIRAGE_COLLECT_STATS"] = "1"
    started = time.monotonic()
    old_argv = sys.argv
    returncode = 0
    try:
        sys.argv = ["mmirage.shard_process", "--config", str(Path(args.config).resolve())]
        try:
            shard_main()
        except SystemExit as exc:
            returncode = int(exc.code or 0)
    except Exception:
        returncode = 1
        raise
    finally:
        sys.argv = old_argv
        wall_seconds = time.monotonic() - started
        Path(args.summary_json).write_text(
            json.dumps(
                {
                    "returncode": returncode,
                    "full_end_to_end_wall_seconds": round(wall_seconds, 6),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    sys.exit(returncode)


if __name__ == "__main__":
    main()
