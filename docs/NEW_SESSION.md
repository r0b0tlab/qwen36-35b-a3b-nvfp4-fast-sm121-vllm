# New-session handoff

## Release state

This project is complete and publicly released. Do not rerun benchmarks or change the serving profile unless the user explicitly opens a new optimization or compatibility task.

- Repository: <https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm>
- Report: <https://r0b0tlab.github.io/qwen36-35b-a3b-nvfp4-fast-sm121-vllm/>
- Image: `ghcr.io/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm:latest`
- Release tag: `v0.1.0`
- Immutable manifest: `sha256:3361f68b966ae84940c068aa7fd522078fd7f3cb087c506edee93041ce031892`
- Benchmarked image ID: `sha256:f9d8f77b6656ddf60e8792242530dd2b9c43ef78bff364b90b82fd01d9cfd444`
- Release run: `20260710T145810Z`

The GHCR package is public. Anonymous manifest access and an anonymous pull by digest were verified.

## Known-good profile

```text
KV_CACHE_DTYPE=fp8
ATTENTION_BACKEND=flashinfer
MOE_BACKEND=flashinfer_b12x
LINEAR_BACKEND=auto
SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=32768
```

Effective CUDA graph mode is `PIECEWISE`. The target MoE uses FlashInfer B12X; the unquantized MTP draft MoE uses Triton. NVFP4 KV is rejected because it failed semantic validation with MTP. Do not substitute Marlin, emulation, or a metadata-only path.

## Final evidence

- GSM8K 0-shot flexible extract: **86.73%**, 1,319 samples.
- Output throughput: **80.61 tok/s at c1**, **268.81 tok/s at c8**, **344.18 tok/s at c32**.
- MTP acceptance: **409,662 / 474,532 = 86.33%**.
- Runtime/native audit: PASS on NVIDIA GB10 / SM121.
- Final semantic probe: PASS (`19 × 23 = 437`).
- Stable concurrency repetitions: zero failed requests.

## Resume checks

```bash
curl -fsS http://127.0.0.1:18080/v1/models
python3 scripts/verify_server.py --base-url http://127.0.0.1:18080
python3 scripts/verify_release.py benchmarks/runs/20260710T145810Z publication/html/index.html
python3 scripts/public_safety_scan.py
```

## Safety boundary

Qwen serving belongs on the dedicated Qwen GB10 node. The separate Hy3 quantization node is protected: do not stop, prune, restart, or reconfigure it. Reconfirm the Hy3 process before any destructive cluster operation.

## Read first

1. `AGENTS.md`
2. `README.md`
3. `docs/SEMANTIC_GATE.md`
4. `docs/ENGINE_LINEAGE.md`
5. `benchmarks/runs/20260710T145810Z/runtime-manifest.json`
