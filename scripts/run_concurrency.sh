#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-qwen36-fast-vllm}"
PORT="${PORT:-18080}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-NVFP4-Fast}"
MODEL_PATH="${MODEL_PATH:-/models/model}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/benchmarks/runs/$RUN_ID/concurrency}"
PHASE_FILE="${PHASE_FILE:-$ROOT/benchmarks/runs/$RUN_ID/phase.txt}"
mkdir -p "$OUT"

for c in 1 2 4 8 16 32; do
  prompts=$(( c * 2 ))
  if (( prompts < 8 )); then prompts=8; fi
  for rep in 1 2 3; do
    printf 'concurrency-c%s-rep%s\n' "$c" "$rep" > "$PHASE_FILE"
    docker exec "$CONTAINER" /usr/local/bin/vllm bench serve \
      --backend openai-chat \
      --base-url "http://127.0.0.1:$PORT" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --tokenizer "$MODEL_PATH" \
      --dataset-name random \
      --random-input-len 2048 \
      --random-output-len 512 \
      --num-prompts "$prompts" \
      --max-concurrency "$c" \
      --request-rate inf \
      --ignore-eos \
      --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result \
      --result-dir "/tmp" \
      --result-filename "qwen36-c${c}-r${rep}.json" \
      2>&1 | tee "$OUT/c${c}-r${rep}.log"
    docker cp "$CONTAINER:/tmp/qwen36-c${c}-r${rep}.json" "$OUT/c${c}-r${rep}.json"
  done
done
printf 'concurrency-complete\n' > "$PHASE_FILE"
python3 "$ROOT/scripts/summarize_concurrency.py" "$OUT" "$OUT/summary.json"
