#!/usr/bin/env bash
set -euo pipefail

export MAX_JOBS="${MAX_JOBS:-6}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

if [[ "${1:-}" == "audit" || "${1:-}" == "--audit-only" ]]; then
  exec /usr/bin/python3 /usr/local/bin/audit_runtime.py
fi
if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

MODEL_PATH="${MODEL_ID:-/models/model}"
if [[ ! -e "$MODEL_PATH" ]] && [[ "$MODEL_PATH" == /* ]]; then
  echo "ERROR: model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

/usr/bin/python3 /usr/local/bin/audit_runtime.py

args=(
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
  --model "$MODEL_PATH"
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-NVFP4-Fast}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}"
  --moe-backend "${MOE_BACKEND:-auto}"
  --linear-backend "${LINEAR_BACKEND:-auto}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${MAX_MODEL_LEN:-65536}"
  --max-num-seqs "${MAX_NUM_SEQS:-32}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --trust-remote-code
)

if [[ "${LANGUAGE_MODEL_ONLY:-1}" == "1" ]]; then
  args+=(--language-model-only)
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE:-1}" == "1" ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_xml}")
fi
if [[ -n "${REASONING_PARSER:-qwen3}" ]]; then
  args+=(--reasoning-parser "${REASONING_PARSER:-qwen3}")
fi
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  args+=("${extra[@]}")
fi

printf 'Launching vLLM with:' >&2
printf ' %q' "${args[@]}" >&2
printf '\n' >&2

exec /usr/bin/python3 -m vllm.entrypoints.openai.api_server "${args[@]}" "$@"
