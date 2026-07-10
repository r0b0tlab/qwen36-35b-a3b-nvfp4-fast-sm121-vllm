#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-18080}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/benchmarks/runs/$RUN_ID/llama-benchy}"
PHASE_FILE="${PHASE_FILE:-$ROOT/benchmarks/runs/$RUN_ID/phase.txt}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/Qwen3.6-35B-A3B-NVFP4-Fast}"
mkdir -p "$OUT"
printf 'llama-benchy-depth-sweep\n' > "$PHASE_FILE"
"$ROOT/.venv-bench/bin/llama-benchy" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name Qwen3.6-35B-A3B-NVFP4-Fast \
  --tokenizer "$MODEL_HOST" \
  --pp 2048 \
  --tg 128 \
  --exact-tg \
  --depth 0 4096 8192 16384 \
  --runs 3 \
  --no-cache \
  --latency-mode api \
  --save-result "$OUT/results.json" \
  --format json \
  --emit-progress "$OUT/progress.jsonl" \
  2>&1 | tee "$OUT/run.log"
printf 'llama-benchy-complete\n' > "$PHASE_FILE"
