#!/usr/bin/env python3
"""Sample GB10 GPU/host telemetry as JSONL until interrupted."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time

GPU_FIELDS = [
    "timestamp",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks_throttle_reasons.active",
]


def gpu_sample() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(cmd, text=True, timeout=10).strip().split(", ")
    return dict(zip(GPU_FIELDS, values, strict=False))


def mem_sample() -> dict:
    values = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key + "_kB"] = int(value.strip().split()[0])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--phase-file", default="")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8", buffering=1) as out:
        while True:
            try:
                phase = ""
                if args.phase_file and os.path.exists(args.phase_file):
                    phase = open(args.phase_file, encoding="utf-8").read().strip()
                row = {
                    "sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "phase": phase,
                    "gpu": gpu_sample(),
                    "host": mem_sample(),
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
            except Exception as exc:
                out.write(json.dumps({"sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": repr(exc)}) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
