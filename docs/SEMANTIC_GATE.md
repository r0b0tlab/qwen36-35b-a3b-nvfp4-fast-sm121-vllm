# Semantic correctness gate

The release profile was selected by output correctness, not loader readiness.

## Isolation matrix

A deterministic chat probe asked for `19 × 23` and required the answer `437`. Experimental variables were passed through the host launcher, verified in the container environment, and confirmed in vLLM's resolved startup configuration.

| KV cache | Target experts | MTP | Result |
|---|---|---|---|
| NVFP4 | FlashInfer CUTLASS | 2 draft tokens | **Rejected** — operand substitutions and punctuation degeneration |
| FP8 | FlashInfer CUTLASS | disabled | Passed deterministic probes |
| FP8 | FlashInfer CUTLASS | 2 draft tokens | Passed; speculative counters and acceptance were non-zero |
| FP8 | FlashInfer B12X | 2 draft tokens; Triton draft experts | **Selected** — passed probes and full GSM8K 0-shot |

## Why the first ablation was invalid

The first launcher revision hard-coded NVFP4 KV and a non-empty MTP configuration in `docker run`. Shell-level overrides therefore never reached the container, making the nominal FP8/no-MTP cases mislabeled. The launcher was parameterized, and every corrected case was checked with both `docker inspect` and the resolved vLLM engine configuration before interpreting output.

## Release rule

The public profile is FP8 KV + B12X target experts + MTP K=2 with Triton draft experts. NVFP4 model weights and NVFP4 KV cache are separate capabilities; the latter is not enabled or advertised as production-ready for this checkpoint.

Readiness, HTTP 200, native-kernel markers, and high throughput are insufficient if generated text is invalid. Any operand substitution, punctuation loop, empty completion, or template leakage blocks publication.
