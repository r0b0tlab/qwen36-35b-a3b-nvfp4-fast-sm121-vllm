# AGENTS.md

## Repository contract

This repository serves `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on NVIDIA GB10 / SM121. Correctness is a release gate, not a benchmark caveat.

## Production profile

- vLLM `0.24.1.dev0+gee0da84ab.d20260702`, CUDA 13.0, PyTorch 2.11 cu130.
- `KV_CACHE_DTYPE=fp8`.
- `MOE_BACKEND=flashinfer_b12x` for the quantized target model.
- `LINEAR_BACKEND=auto` so mixed FP8 and NVFP4 layers select compatible native kernels.
- `SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}`.
- FlashInfer attention, vLLM compile enabled, effective PIECEWISE CUDA graphs with MTP.
- No Marlin, emulation, metadata-only quantization, or eager-mode performance claims.

Do not switch production to NVFP4 KV. The experimental path loaded but failed semantic validation for this checkpoint. FP8 KV is the validated default.

Do not force the global linear backend to `flashinfer_b12x`: this mixed checkpoint has FP8 dense layers without a B12X implementation. Do not use B12X for the MTP draft MoE: that layer is unquantized and must use Triton in this runtime.

## Required validation order

1. Confirm the host is a real GB10 with compute capability 12.1.
2. Run `scripts/audit_runtime.py`.
3. Inspect the container environment after launch; launcher overrides must actually reach Docker.
4. Inspect startup logs for:
   - `quantization=compressed-tensors`
   - `kv_cache_dtype=fp8`
   - `SpeculativeConfig(method='mtp'` and two speculative tokens
   - `CutlassFP8ScaledMMLinearKernel`
   - `FlashInferCutlassNvFp4LinearKernel`
   - target `FLASHINFER_B12X` NVFP4 MoE
   - draft `TRITON` unquantized MoE
   - FlashInfer attention
   - effective `PIECEWISE` CUDA graph mode
5. Run `scripts/verify_server.py` and inspect sampled outputs.
6. Confirm non-zero MTP draft and accepted-token counters.
7. Run GSM8K 0-shot with flexible extraction before publishing throughput.
8. Run concurrency, latency, telemetry, and power benchmarks only after accuracy passes.
9. Run `scripts/public_safety_scan.py` before Git publication.

HTTP 200, `/v1/models`, native-kernel markers, and high token throughput do not establish valid logits. Operand substitution, punctuation loops, empty completions, or template leakage are hard failures.

## Operational safety

All Qwen deployment and benchmarking work belongs on the explicitly idle GB10 node. Never stop, prune, clean, or reconfigure the separate Hy3 quantization node. Reconfirm the protected process before destructive operations.

## Publication hygiene

- Never commit model weights, tokens, local virtual environments, raw private paths, or credentials.
- Preserve exact model revision, image digest, runtime commit, benchmark command, and telemetry method.
- Report requested and effective runtime modes separately.
- Credit Unsloth, Qwen, vLLM, FlashInfer, NVIDIA CUTLASS DSL, and relevant upstream contributors.
