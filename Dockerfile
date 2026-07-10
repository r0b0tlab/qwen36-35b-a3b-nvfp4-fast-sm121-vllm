FROM sm121-vllm-v0240-nvfp4:nvfp4-sf-global-valid-v1

LABEL org.opencontainers.image.title="Qwen3.6-35B-A3B-NVFP4-Fast vLLM for GB10"
LABEL org.opencontainers.image.description="SM121-native vLLM 0.24.1-dev runtime for Unsloth Qwen3.6-35B-A3B-NVFP4-Fast with FlashInfer B12X, FP8 KV, and MTP K=2"
LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm"
LABEL org.opencontainers.image.licenses="MIT"

ENV MODEL_ID=/models/model \
    SERVED_MODEL_NAME=Qwen3.6-35B-A3B-NVFP4-Fast \
    HOST=0.0.0.0 \
    PORT=8000 \
    KV_CACHE_DTYPE=fp8 \
    ATTENTION_BACKEND=flashinfer \
    MOE_BACKEND=flashinfer_b12x \
    LINEAR_BACKEND=auto \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_MODEL_LEN=65536 \
    MAX_NUM_SEQS=32 \
    MAX_NUM_BATCHED_TOKENS=32768 \
    SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
    LANGUAGE_MODEL_ONLY=1 \
    ENABLE_AUTO_TOOL_CHOICE=1 \
    TOOL_CALL_PARSER=qwen3_xml \
    REASONING_PARSER=qwen3 \
    MAX_JOBS=6 \
    FLASHINFER_NVCC_THREADS=2 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUTE_DSL_ARCH=sm_121a

COPY scripts/start_vllm.sh /usr/local/bin/start_vllm.sh
COPY scripts/audit_runtime.py /usr/local/bin/audit_runtime.py
COPY scripts/verify_server.py /usr/local/bin/verify_server.py
RUN chmod +x /usr/local/bin/start_vllm.sh /usr/local/bin/audit_runtime.py /usr/local/bin/verify_server.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD /usr/bin/python3 /usr/local/bin/verify_server.py --health-only

ENTRYPOINT ["/usr/local/bin/start_vllm.sh"]
