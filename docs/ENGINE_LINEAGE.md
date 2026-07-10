# Engine lineage and upstream credit

The image is an ARM64/SM121 build of vLLM `0.24.1.dev0+gee0da84ab.d20260702` with CUDA 13.0, PyTorch `2.11.0+cu130`, FlashInfer `0.6.13`, and NVIDIA CUTLASS DSL `4.5.2`.

## Validated serving profile

- Checkpoint loader: vLLM compressed-tensors mixed precision.
- FP8 dense groups: `CutlassFP8ScaledMMLinearKernel`.
- NVFP4 dense GEMM: `FlashInferCutlassNvFp4LinearKernel`.
- NVFP4 target-model experts: explicit `FLASHINFER_B12X` on SM121.
- MTP draft experts: Triton, because the draft expert layer is unquantized and this vLLM build does not support B12X for unquantized MoE.
- Attention: FlashInfer.
- KV cache: FP8.
- Speculative decoding: the checkpoint's MTP head, two draft tokens.
- Effective CUDA graph mode: `PIECEWISE`; vLLM downgrades the requested full-and-piecewise mode when FlashInfer attention and speculative decoding are combined.

The B12X SM12x kernel integration originates in vLLM PR [#40082](https://github.com/vllm-project/vllm/pull/40082) and its corresponding FlashInfer/CUTLASS work. The runtime was validated on a real NVIDIA GB10 (compute capability 12.1), including native `compute_121a,code=sm_121a` JIT output.

## Why FP8 KV is the production default

The image contains experimental SM120/SM121 NVFP4-KV code derived from vLLM PR [#46329](https://github.com/vllm-project/vllm/pull/46329) and FlashInfer PR [#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684), authored by Jetha Chan. It is retained for research and auditing, but it is not the production default: this checkpoint emitted semantically corrupt output in the tested NVFP4-KV + MTP profile. FP8 KV passed deterministic semantic probes and the full accuracy gate.

## Runtime selection

The Unsloth model card explicitly recommends vLLM 0.24.0 or newer and MTP with two speculative tokens. SGLang current development work supports Qwen3.6 ModelOpt NVFP4 variants, but this artifact is a compressed-tensors mixed FP8/NVFP4 export. The pinned vLLM lineage both matches the model card and passed the exact checkpoint's loader, semantic, MTP, and native SM121 gates, so it is the selected implementation.

The model and calibration are by Unsloth and Qwen. Model weights are not redistributed by this repository or container.
