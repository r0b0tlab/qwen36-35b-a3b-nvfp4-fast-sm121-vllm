#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> None:
    raw_dir, output = map(Path, sys.argv[1:3])
    candidates = sorted(raw_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no lm-eval JSON under {raw_dir}")
    source = candidates[0]
    data = json.loads(source.read_text())
    results = data.get("results", {}).get("gsm8k", {})
    config = data.get("config", {})
    payload = {
        "task": "gsm8k",
        "num_fewshot": 0,
        "sample_count": data.get("n-samples", {}).get("gsm8k", {}).get("effective") or results.get("alias"),
        "exact_match_flexible_extract": results.get("exact_match,flexible-extract"),
        "exact_match_flexible_extract_stderr": results.get("exact_match_stderr,flexible-extract"),
        "exact_match_strict_match": results.get("exact_match,strict-match"),
        "exact_match_strict_match_stderr": results.get("exact_match_stderr,strict-match"),
        "lm_eval_version": data.get("versions", {}).get("gsm8k"),
        "model": config.get("model"),
        "model_args": config.get("model_args"),
        "source_file": str(source.relative_to(output.parent)),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
