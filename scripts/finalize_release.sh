#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: scripts/finalize_release.sh RUN_ID}"
RUN="$ROOT/benchmarks/runs/$RUN_ID"
PORT="${PORT:-18080}"
CONTAINER="${CONTAINER:-qwen36-fast-vllm}"

for required in \
  "$RUN/gsm8k/results.json" \
  "$RUN/llama-benchy/results.json" \
  "$RUN/concurrency/summary.json" \
  "$RUN/telemetry.jsonl"; do
  [[ -s "$required" ]] || { echo "Missing release input: $required" >&2; exit 2; }
done

PORT="$PORT" SERVED_MODEL_NAME=Qwen3.6-35B-A3B-NVFP4-Fast \
  python3 "$ROOT/scripts/verify_server.py" > "$RUN/semantic-smoke.json"
python3 "$ROOT/scripts/summarize_telemetry.py" \
  "$RUN/telemetry.jsonl" "$RUN/telemetry-summary.json"
CONTAINER="$CONTAINER" PORT="$PORT" python3 "$ROOT/scripts/capture_release.py" "$RUN"
python3 "$ROOT/scripts/summarize_energy.py" \
  "$RUN/telemetry.jsonl" "$RUN/concurrency/summary.json" "$RUN/energy-efficiency.json"

mkdir -p "$ROOT/publication/html" "$ROOT/docs"
python3 "$ROOT/scripts/generate_report.py" "$RUN" "$ROOT/publication/html/index.html"
cp "$ROOT/publication/html/index.html" "$ROOT/docs/index.html"
python3 "$ROOT/scripts/verify_release.py" "$RUN" "$ROOT/publication/html/index.html"

python3 "$ROOT/scripts/write_manifest.py" "$RUN"
python3 "$ROOT/scripts/public_safety_scan.py"
printf 'Release artifacts finalized: %s\n' "$RUN"
