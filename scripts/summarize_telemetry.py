#!/usr/bin/env python3
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
    source, output = map(Path, sys.argv[1:3])
    grouped = collections.defaultdict(list)
    for line in source.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "gpu" in row:
            grouped[row.get("phase") or "unlabeled"].append(row)
    summary = {}
    for phase, rows in sorted(grouped.items()):
        power = [number(x["gpu"].get("power.draw")) for x in rows]
        util = [number(x["gpu"].get("utilization.gpu")) for x in rows]
        temp = [number(x["gpu"].get("temperature.gpu")) for x in rows]
        power = [x for x in power if x is not None]
        util = [x for x in util if x is not None]
        temp = [x for x in temp if x is not None]
        summary[phase] = {
            "samples": len(rows),
            "power_w_mean": statistics.mean(power) if power else None,
            "power_w_p95": sorted(power)[max(0, int(len(power) * 0.95) - 1)] if power else None,
            "power_w_max": max(power) if power else None,
            "gpu_util_pct_mean": statistics.mean(util) if util else None,
            "gpu_util_pct_max": max(util) if util else None,
            "temperature_c_mean": statistics.mean(temp) if temp else None,
            "temperature_c_max": max(temp) if temp else None,
        }
    payload = {"source": source.name, "phases": summary}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
