#!/usr/bin/env python3
"""Minimal OpenAI-compatible client used by raw-SGLang and VLM baselines.

Supports two payload shapes against an SGLang or vLLM OpenAI endpoint:

* text completions (``POST /v1/completions``) with a ``prompt`` string, and
* chat completions (``POST /v1/chat/completions``) with a text message and an
  optional local image attached as a ``data:`` URL (VLM tasks).

Results and a throughput summary are written as JSONL / JSON respectively. The
``--tokenizer`` is only used to count generated tokens when the server response
does not report usage.
"""

from __future__ import annotations

import argparse
import base64
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
    parser.add_argument("--path", default="raw_sglang", help="Label stored in the summary.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use /v1/chat/completions (single user message) instead of /v1/completions.",
    )
    parser.add_argument("--text-field", default="prompt", help="Row field with the prompt text.")
    parser.add_argument("--image-field", default=None, help="Optional row field with an image file path.")
    parser.add_argument("--image-base-path", default=None, help="Directory for relative --image-field paths.")
    parser.add_argument("--id-field", default="source_index", help="Row field copied to each output row.")
    return parser.parse_args()


def load_rows(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def post_request(
    base_url: str,
    endpoint: str,
    api_key: Optional[str],
    timeout_seconds: int,
    payload: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{endpoint}"
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


def image_data_url(path: Path, base: Optional[Path]) -> str:
    full_path = path if path.is_absolute() else base / path
    if not full_path.exists():
        raise FileNotFoundError(f"Missing image: {full_path}")
    payload = base64.b64encode(full_path.read_bytes()).decode("ascii")
    suffix = full_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{payload}"


def response_text(response: Dict[str, Any], chat: bool) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Response has no choices: {response!r}")
    message = choices[0]
    if chat:
        return message.get("message", {}).get("content", "") or ""
    return message.get("text", "") or ""


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.prompts_jsonl), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    image_base = Path(args.image_base_path) if args.image_base_path else None
    endpoint = "/v1/chat/completions" if args.chat else "/v1/completions"

    request_params: Dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        request_params["top_p"] = args.top_p

    def failure(index: int, row: Dict[str, Any], started: float, error: str) -> Dict[str, Any]:
        return {
            "index": index,
            args.id_field: row.get(args.id_field),
            "status": "failed",
            "http_status": None,
            "latency_seconds": round(time.monotonic() - started, 6),
            "output_text": "",
            "output_tokens": 0,
            "error": error,
        }

    def payload_for(row: Dict[str, Any]) -> Dict[str, Any]:
        if args.chat:
            content: List[Any] = []
            if args.image_field:
                content.append({"type": "image_url", "image_url": {"url": image_data_url(Path(str(row[args.image_field])), image_base)}})
            content.append({"type": "text", "text": str(row[args.text_field])})
            return {**request_params, "messages": [{"role": "user", "content": content}]}
        return {**request_params, "prompt": str(row[args.text_field])}

    def run_one(index_and_row: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        index, row = index_and_row
        started = time.monotonic()
        try:
            status_code, response = post_request(
                args.base_url,
                endpoint,
                args.api_key,
                args.timeout_seconds,
                payload_for(row),
            )
            text = response_text(response, args.chat)
            return {
                "index": index,
                args.id_field: row.get(args.id_field),
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
        "path": args.path,
        "rows": len(results),
        "success_count": success_count,
        "total_output_tokens": total_output_tokens,
        "generation_wall_seconds": round(wall_seconds, 6),
        "rows_per_second": round(len(results) / wall_seconds, 6) if wall_seconds > 0 else None,
        "output_tokens_per_second_per_gpu": round(total_output_tokens / (wall_seconds * args.gpu_count), 6)
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
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
