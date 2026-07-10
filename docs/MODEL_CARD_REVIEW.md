# Model-card and runtime review

Reviewed: 2026-07-10

## Checkpoint

`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`

Observed revision: `11fb65d632976b5f00cb42f157562238cf0c71e0`

The checkpoint is not uniformly NVFP4. Its compressed-tensors metadata defines:

- FP8 weight/activation groups for attention projections, Gated DeltaNet projections, and `lm_head`.
- NVFP4 group-16 expert and shared-expert weights/activations.
- Unquantized ignored modules, including the trained MTP layer.

That mixed layout is why a global B12X linear override is invalid: the FP8 layers need their own native FP8 kernel.

## Model-card recommendation

The model card recommends vLLM 0.24.0 or newer and demonstrates MTP with two speculative tokens:

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

It describes the `Fast` artifact as the optimized NVFP4 release and cites a roughly 1.79× throughput advantage in the author's reference environment.

## Runtime compatibility decision

### vLLM — selected

The pinned runtime is `0.24.1.dev0+gee0da84ab.d20260702` on ARM64, CUDA 13.0, PyTorch 2.11, FlashInfer 0.6.13, and CUTLASS DSL 4.5.2. It:

- resolves `Qwen3_5MoeForConditionalGeneration` and the Qwen3.5/3.6 MTP architecture;
- loads this checkpoint as `compressed-tensors` mixed precision;
- provides native SM121 FP8, NVFP4 dense, NVFP4 B12X MoE, FlashInfer attention, and MTP paths;
- passed deterministic semantic probes and full accuracy evaluation using the exact artifact.

For explicit B12X target experts, the MTP draft must set `moe_backend=triton`, because its expert layer is unquantized and B12X is not registered as an unquantized-MoE backend in this build.

### SGLang — compatible family, not selected for this artifact

SGLang development PR #27906 added Qwen3.6 ModelOpt mixed/regular NVFP4 and MTP support and reported strong GSM8K and throughput results on reference Blackwell hardware. PR #26496 added SM120 NVFP4 performance/usability tuning. Those are relevant alternatives for ModelOpt-formatted Qwen3.6 checkpoints.

This Unsloth artifact is a compressed-tensors mixed export, while the exact vLLM path is recommended by the model card and has been validated end to end on GB10. Changing engines would add a loader-format variable without evidence of an advantage for this exact checkpoint, so SGLang was not chosen.

## KV-cache decision

NVFP4 model weights and NVFP4 KV cache are separate capabilities. Experimental NVFP4 KV loaded and advertised very large capacity, but the tested NVFP4-KV + MTP profile generated operand substitutions and punctuation degeneration. It failed the semantic gate.

FP8 KV passed the same deterministic probes with MTP enabled and is the production default. NVFP4 KV remains a research-only option, not an optimization claim.

## Selected production profile

```text
KV cache:            FP8
Target NVFP4 MoE:    FLASHINFER_B12X
Mixed linear:        auto
Attention:           FlashInfer
MTP:                 2 tokens
MTP draft MoE:       Triton
Max model length:    65,536
Language path:       text-only
CUDA graphs:         PIECEWISE effective mode with MTP + FlashInfer
```

## Sources

- Unsloth model card: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
- vLLM B12X integration: https://github.com/vllm-project/vllm/pull/40082
- vLLM SM120 B12X backend-selection discussion: https://github.com/vllm-project/vllm/pull/47577
- SGLang Qwen3.6 ModelOpt mixed NVFP4 support: https://github.com/sgl-project/sglang/pull/27906
- SGLang SM120 NVFP4 tuning: https://github.com/sgl-project/sglang/pull/26496
