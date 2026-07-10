#!/usr/bin/env python3
"""Reject secrets, local identities, and oversized files before publication."""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv-bench", "raw", "gsm8k-smoke", ".release-cache", "__pycache__"}
SKIP_NAMES = {"telemetry.jsonl", "run.log", "progress.jsonl", "server.log"}
PATTERNS = {
    "private_ipv4": re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "home_path": re.compile(r"/home/[A-Za-z0-9_.-]+"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
}
def main() -> int:
    failures = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in SKIP for part in path.parts)
            or path.name in SKIP_NAMES
            or (path.name.startswith("c") and "-r" in path.name and path.suffix == ".log")
        ):
            continue
        rel = str(path.relative_to(ROOT))
        size = path.stat().st_size
        if size > 100_000_000:
            failures.append((rel, "oversized", str(size)))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                failures.append((rel, name, match.group(0)))
    if failures:
        for item in failures:
            print("FAIL", *item)
        return 1
    print("PASS public safety scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
