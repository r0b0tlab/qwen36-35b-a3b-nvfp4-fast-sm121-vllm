#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/Qwen3.6-35B-A3B-NVFP4-Fast}"
IMAGE="${IMAGE:-qwen36-35b-a3b-nvfp4-fast-sm121-vllm:dev}"
CONTAINER="${CONTAINER:-qwen36-fast-vllm}"
PORT="${PORT:-18080}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/qwen36-fast-vllm}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG-}"
if [[ "${ENABLE_MTP:-1}" == "1" && -z "$SPECULATIVE_CONFIG" ]]; then
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
fi
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/flashinfer" "$CACHE_ROOT/vllm"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  -e PORT="$PORT" \
  -e MODEL_ID=/models/model \
  -e SERVED_MODEL_NAME=Qwen3.6-35B-A3B-NVFP4-Fast \
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
  -e SPECULATIVE_CONFIG="$SPECULATIVE_CONFIG" \
  -e MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}" \
  -e LINEAR_BACKEND="${LINEAR_BACKEND:-auto}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
  -e LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}" \
  -e MAX_JOBS=6 \
  -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODEL_HOST:/models/model:ro" \
  -v "$CACHE_ROOT/huggingface:/root/.cache/huggingface" \
  -v "$CACHE_ROOT/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm" \
  "$IMAGE"

echo "container=$CONTAINER port=$PORT"
echo "logs: docker logs -f $CONTAINER"
