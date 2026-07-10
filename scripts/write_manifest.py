#!/usr/bin/env python3
"""Write and verify a SHA256 manifest for public run artifacts."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXCLUDED_NAMES = {"telemetry.jsonl", "run.log", "progress.jsonl", "MANIFEST.sha256"}


def excluded(path: Path) -> bool:
    return (
        path.name in EXCLUDED_NAMES
        or "raw" in path.parts
        or (path.name.startswith("c") and "-r" in path.name and path.suffix == ".log")
    )


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or excluded(path):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root)}")
    output = root / "MANIFEST.sha256"
    output.write_text("\n".join(rows) + "\n")
    for row in rows:
        expected, rel = row.split("  ", 1)
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {rel}")
    print(f"PASS manifest files={len(rows)} path={output}")


if __name__ == "__main__":
    main()
