#!/usr/bin/env python3
"""Health and semantic smoke gate for the bundled vLLM endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def request(path: str, payload: dict | None = None, timeout: int = 30, base_url: str | None = None) -> dict:
    base = (base_url or f"http://127.0.0.1:{os.getenv('PORT', '8000')}").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--base-url", default=None, help="Endpoint root or OpenAI /v1 base URL")
    args = parser.parse_args()
    expected = os.getenv("SERVED_MODEL_NAME", "Qwen3.6-35B-A3B-NVFP4-Fast")
    models = request("/v1/models", base_url=args.base_url)
    ids = [item.get("id") for item in models.get("data", [])]
    if expected not in ids:
        raise RuntimeError(f"expected model {expected!r}; got {ids!r}")
    if args.health_only:
        return 0
    result = request(
        "/v1/chat/completions",
        {
            "model": expected,
            "messages": [{"role": "user", "content": "Compute 19 multiplied by 23. Return the integer answer."}],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
        base_url=args.base_url,
    )
    message = result["choices"][0]["message"]
    content = (message.get("content") or "") + " " + (message.get("reasoning") or "")
    print(json.dumps(result, indent=2))
    if "437" not in content:
        raise RuntimeError(f"semantic smoke failed: {content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
