#!/usr/bin/env python3
"""Minimal raw SGLang OpenAI-compatible completion client."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_prompt_rows(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"Prompt row lacks string 'prompt': {row!r}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def post_completion(
    base_url: str,
    api_key: Optional[str],
    timeout_seconds: int,
    payload: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    rows = load_prompt_rows(Path(args.prompts_jsonl), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )

    request_params: Dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        request_params["top_p"] = args.top_p

    def failure(
        index: int, row: Dict[str, Any], started: float, error: str
    ) -> Dict[str, Any]:
        return {
            "index": index,
            "source_index": row.get("source_index"),
            "status": "failed",
            "http_status": None,
            "latency_seconds": round(time.monotonic() - started, 6),
            "output_text": "",
            "output_tokens": 0,
            "error": error,
        }

    def run_one(index_and_row: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        index, row = index_and_row
        started = time.monotonic()
        try:
            status_code, response = post_completion(
                args.base_url,
                args.api_key,
                args.timeout_seconds,
                {**request_params, "prompt": row["prompt"]},
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"Response has no choices: {response!r}")
            text = choices[0].get("text", "")
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
            return {
                "index": index,
                "source_index": row.get("source_index"),
                "status": "success",
                "http_status": status_code,
                "latency_seconds": round(time.monotonic() - started, 6),
                "output_text": text,
                "output_tokens": len(tokenizer.encode(text)),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return failure(index, row, started, f"HTTP {exc.code}: {body}")
        except Exception as exc:
            return failure(index, row, started, f"{type(exc).__name__}: {exc}")

    started = time.monotonic()
    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        results = [run_one(item) for item in enumerate(rows)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_one, item) for item in enumerate(rows)]
            results = [future.result() for future in futures]
    wall_seconds = time.monotonic() - started
    results.sort(key=lambda item: item["index"])

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    total_output_tokens = sum(int(row["output_tokens"]) for row in results)
    success_count = sum(1 for row in results if row["status"] == "success")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": "raw_sglang",
        "rows": len(results),
        "success_count": success_count,
        "total_output_tokens": total_output_tokens,
        "generation_wall_seconds": round(wall_seconds, 6),
        "rows_per_second": round(len(results) / wall_seconds, 6)
        if wall_seconds > 0
        else None,
        "output_tokens_per_second_per_gpu": round(
            total_output_tokens / (wall_seconds * args.gpu_count), 6
        )
        if wall_seconds > 0 and args.gpu_count > 0
        else None,
        "gpu_count": args.gpu_count,
        "concurrency": concurrency,
        "generation_settings": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
