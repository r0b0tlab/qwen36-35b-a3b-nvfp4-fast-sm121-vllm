#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-18080}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/benchmarks/runs/$RUN_ID/gsm8k}"
PHASE_FILE="${PHASE_FILE:-$ROOT/benchmarks/runs/$RUN_ID/phase.txt}"
mkdir -p "$OUT/raw"
printf 'gsm8k-full-0shot\n' > "$PHASE_FILE"

"$ROOT/.venv-bench/bin/python" -m lm_eval \
  --model local-chat-completions \
  --model_args "model=Qwen3.6-35B-A3B-NVFP4-Fast,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k \
  --num_fewshot 0 \
  --apply_chat_template \
  --batch_size auto \
  --gen_kwargs '{"max_gen_toks":2048,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  --log_samples \
  --output_path "$OUT/raw" \
  2>&1 | tee "$OUT/run.log"

"$ROOT/.venv-bench/bin/python" "$ROOT/scripts/normalize_gsm8k.py" "$OUT/raw" "$OUT/results.json"
printf 'gsm8k-complete\n' > "$PHASE_FILE"
