#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import statistics
import sys
from pathlib import Path


def metric(data: dict, *names: str):
    for name in names:
        if name in data:
            return data[name]
    return None


def main() -> None:
    root, output = map(Path, sys.argv[1:3])
    rows = []
    for c in (1, 2, 4, 8, 16, 32):
        reps = []
        for rep in (1, 2, 3):
            p = root / f"c{c}-r{rep}.json"
            data = json.loads(p.read_text())
            reps.append(data)
        stable = reps[1:]
        fields = {
            "request_throughput_rps": ("request_throughput",),
            "output_throughput_tok_s": ("output_throughput",),
            "total_throughput_tok_s": ("total_token_throughput", "total_throughput"),
            "mean_ttft_ms": ("mean_ttft_ms",),
            "p50_ttft_ms": ("median_ttft_ms", "p50_ttft_ms"),
            "p99_ttft_ms": ("p99_ttft_ms",),
            "mean_tpot_ms": ("mean_tpot_ms",),
            "p99_tpot_ms": ("p99_tpot_ms",),
            "mean_itl_ms": ("mean_itl_ms",),
            "p99_itl_ms": ("p99_itl_ms",),
            "mean_e2el_ms": ("mean_e2el_ms",),
            "p99_e2el_ms": ("p99_e2el_ms",),
        }
        row = {"concurrency": c, "repetitions": 3, "warmup_rep_dropped": 1}
        for out_name, names in fields.items():
            vals = [metric(x, *names) for x in stable]
            vals = [float(x) for x in vals if x is not None]
            row[out_name] = statistics.mean(vals) if vals else None
        row["completed"] = sum(int(metric(x, "completed") or 0) for x in stable)
        row["failed"] = sum(int(metric(x, "failed") or 0) for x in stable)
        rows.append(row)
    payload = {"method": "vllm bench serve random 2048 input / 512 output / ignore_eos; three reps, first dropped", "rows": rows}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
