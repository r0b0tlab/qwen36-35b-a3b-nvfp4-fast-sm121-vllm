#!/usr/bin/env python3
"""Join active-GPU power telemetry with stable concurrency throughput."""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path


def number(value):
    try:
        return float(str(value).replace(" W", "").replace(" %", ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    telemetry_path, concurrency_path, output_path = map(Path, sys.argv[1:4])
    grouped = collections.defaultdict(list)
    for line in telemetry_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = row.get("phase") or "unlabeled"
        gpu = row.get("gpu") or {}
        utilization = number(gpu.get("utilization.gpu"))
        power = number(gpu.get("power.draw"))
        # Exclude launch/client bookkeeping and post-run idle time retained under
        # the same phase label. Power efficiency describes active inference.
        if utilization is not None and utilization >= 10.0 and power is not None:
            grouped[phase].append(power)

    concurrency = json.loads(concurrency_path.read_text()).get("rows", [])
    rows = []
    for row in concurrency:
        c = int(row["concurrency"])
        phase_names = [f"concurrency-c{c}-rep{rep}" for rep in (2, 3)]
        powers = [value for phase in phase_names for value in grouped.get(phase, [])]
        throughput = row.get("output_throughput_tok_s")
        mean_power = statistics.mean(powers) if powers else None
        rows.append(
            {
                "concurrency": c,
                "output_throughput_tok_s": throughput,
                "mean_active_power_w": mean_power,
                "active_power_samples": len(powers),
                "active_utilization_threshold_pct": 10.0,
                "output_tokens_per_joule": throughput / mean_power if throughput and mean_power else None,
                "joules_per_1k_output_tokens": mean_power * 1000.0 / throughput if throughput and mean_power else None,
                "source_phases": phase_names,
            }
        )
    payload = {
        "method": "Board-power samples at GPU utilization >=10% from stable repetitions 2 and 3, joined to output throughput from those repetitions. Idle/client-bookkeeping samples are excluded.",
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
