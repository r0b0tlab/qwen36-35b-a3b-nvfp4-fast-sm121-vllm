#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import sys
import urllib.request

PATTERNS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def main() -> None:
    url, output = sys.argv[1:3]
    text = urllib.request.urlopen(url, timeout=15).read().decode()
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not any(name in line for name in PATTERNS):
            continue
        match = re.match(r"([^\s]+)\s+([-+0-9.eE]+)$", line)
        if match:
            rows.append({"series": match.group(1), "value": float(match.group(2))})
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"source": url, "metrics": rows}, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"source": url, "series": len(rows)}))


if __name__ == "__main__":
    main()
