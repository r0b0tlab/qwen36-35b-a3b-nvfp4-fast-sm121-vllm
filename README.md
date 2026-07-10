# Qwen3.6 35B-A3B NVFP4 Fast on NVIDIA GB10

[![GPU: GB10 / SM121](https://img.shields.io/badge/GPU-GB10%20%2F%20SM121-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![vLLM: 0.24.1-dev](https://img.shields.io/badge/vLLM-0.24.1--dev-5B8DEF)](https://github.com/vllm-project/vllm)
[![Model: Qwen3.6 35B-A3B](https://img.shields.io/badge/Qwen3.6-35B--A3B-62F6FF)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

A correctness-gated, SM121-native vLLM deployment for
[`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
on one NVIDIA GB10.

The checkpoint is mixed precision, not uniformly NVFP4. The validated profile keeps those boundaries explicit:

- compressed-tensors FP8 dense groups;
- NVFP4 group-16 target experts;
- native FlashInfer/CUTLASS SM121 kernels;
- FP8 KV cache;
- FlashInfer B12X target MoE;
- two-token trained MTP;
- Triton for the unquantized MTP draft MoE;
- FlashInfer attention and effective piecewise CUDA graphs.

> Correctness comes before throughput. An experimental NVFP4-KV profile reached HTTP readiness but changed prompt operands and degenerated punctuation. It is rejected here; all published results use FP8 KV.

## Results

Benchmark run: `20260710T145810Z`

### Quality

| Task | Protocol | Result |
|---|---|---:|
| GSM8K | 0-shot, chat template, thinking disabled, greedy, flexible extraction | **86.73% ± 0.93%** |
| Deterministic arithmetic | `19 × 23`, 256-token allowance | **pass** |
| Exact-string and word-problem probes | greedy | **pass** |
| MTP cumulative acceptance | 474,270 proposed draft tokens | **86.33%** |

The strict GSM8K string metric is `0.0` because this no-few-shot protocol does not force the dataset's literal `#### answer` output convention. Flexible extraction is the declared primary metric.

### Random serving workload

2,048-token random input, exact 512-token output, temperature 0, three repetitions per point, first repetition discarded.

| Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 ITL |
|---:|---:|---:|---:|---:|---:|
| 1 | 80.6 | 404.7 | 468.8 ms | 11.51 ms | 39.31 ms |
| 2 | 128.4 | 644.8 | 457.9 ms | 14.67 ms | 48.84 ms |
| 4 | 185.9 | 933.2 | 865.7 ms | 19.39 ms | 59.39 ms |
| 8 | 268.8 | 1,349.6 | 1,560.7 ms | 25.13 ms | 331.39 ms |
| 16 | 285.6 | 1,433.9 | 8,105.8 ms | 38.02 ms | 390.28 ms |
| 32 | **344.2** | **1,727.9** | 13,080.1 ms | 64.21 ms | 1,235.71 ms |

Concurrency 8 is the practical knee for interactive service. Concurrency 32 maximizes aggregate output throughput but carries substantially higher queueing latency.

### llama-benchy depth sweep

2,048-token prompt, exact 128-token generation, concurrency 1, three runs per context depth.

| Prior context | Prefill tok/s | Decode tok/s | TTFR |
|---:|---:|---:|---:|
| 0 | 5,081.5 | 79.7 | 405.6 ms |
| 4,096 | 6,571.4 | 86.7 | 939.5 ms |
| 8,192 | 6,211.0 | 85.3 | 1,651.2 ms |
| 16,384 | 5,526.9 | 87.0 | 3,339.0 ms |

### Telemetry

- peak reported board power: **69.35 W**;
- peak GPU utilization: **96%**;
- peak temperature: **81 °C**;
- FP8 KV capacity: **5,960,499 tokens**;
- 65,536-token profile theoretical sequence capacity: about **90.95×**.

Open the self-contained visual report:

[`reports/qwen36-35b-a3b-nvfp4-fast-gb10-20260710.html`](reports/qwen36-35b-a3b-nvfp4-fast-gb10-20260710.html)

Machine-readable aggregates live under [`benchmarks/runs/20260710T145810Z`](benchmarks/runs/20260710T145810Z).

## Runtime lineage

| Component | Version / selection |
|---|---|
| Architecture | Linux aarch64, NVIDIA GB10, compute capability 12.1 |
| vLLM | `0.24.1.dev0+gee0da84ab.d20260702` |
| PyTorch | `2.11.0+cu130` |
| CUDA | 13.0 |
| FlashInfer | 0.6.13 |
| CUTLASS DSL | 4.5.2 |
| KV cache | FP8 |
| Attention | FlashInfer |
| Target MoE | `FLASHINFER_B12X` |
| MTP draft MoE | `TRITON` |
| Linear backend | `auto` |
| Effective CUDA graphs | `PIECEWISE` |

Full lineage and upstream credit: [`docs/ENGINE_LINEAGE.md`](docs/ENGINE_LINEAGE.md).
Model-card/runtime decision record: [`docs/MODEL_CARD_REVIEW.md`](docs/MODEL_CARD_REVIEW.md).

## Run

The model is mounted at runtime and is not redistributed in the container.

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.6-35B-A3B-NVFP4-Fast
export IMAGE=ghcr.io/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm:latest

docker run --rm --gpus all \
  --name qwen36-fast-vllm \
  --network host --ipc host --shm-size 32g \
  -e MODEL_PATH=/models/model \
  -e PORT=18080 \
  -v "$MODEL_DIR:/models/model:ro" \
  -v "$HOME/.cache/qwen36-fast-vllm:/root/.cache" \
  "$IMAGE"
```

The image defaults to the validated profile. Override values deliberately rather than copying an experimental profile:

```text
KV_CACHE_DTYPE=fp8
MOE_BACKEND=flashinfer_b12x
LINEAR_BACKEND=auto
SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}
```

Verify readiness and semantics:

```bash
curl -fsS http://127.0.0.1:18080/v1/models
python3 scripts/verify_server.py --base-url http://127.0.0.1:18080/v1
```

## Build

The Dockerfile derives from the r0b0tlab ARM64/SM121 vLLM runtime and adds the auditable entrypoint and validation tooling:

```bash
docker build -t qwen36-35b-a3b-nvfp4-fast-sm121-vllm:dev .
docker run --rm --gpus all qwen36-35b-a3b-nvfp4-fast-sm121-vllm:dev --audit-only
```

The exact runtime used for the benchmark is recorded in `runtime-manifest.json`, including its image ID. The final GHCR package is self-contained; the model weights remain a separate Hugging Face download.

## Reproduce benchmarks

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
PORT=18080 RUN_ID="$RUN_ID" bash scripts/run_gsm8k.sh
PORT=18080 RUN_ID="$RUN_ID" bash scripts/run_llama_benchy.sh
PORT=18080 RUN_ID="$RUN_ID" bash scripts/run_concurrency.sh
```

Start `scripts/telemetry.py` before the campaign and set the phase file used by each wrapper. Raw samples and server logs are intentionally excluded from publication; normalized outputs and the runtime manifest are retained.

## Validation policy

A release is invalid if any of these are missing:

1. standalone GPU runtime audit;
2. `/v1/models` readiness;
3. deterministic semantic output;
4. log evidence for the intended weight, KV, attention, MoE, and speculative paths;
5. non-zero MTP counters when MTP is enabled;
6. benchmark and telemetry artifacts generated from the same coherent profile.

See [`AGENTS.md`](AGENTS.md) for the full repository contract.

## Credits

- [Qwen](https://github.com/QwenLM) for Qwen3.6.
- [Unsloth](https://github.com/unslothai/unsloth) for the published mixed FP8/NVFP4 checkpoint.
- [vLLM](https://github.com/vllm-project/vllm) for serving and benchmark infrastructure.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for SM121 attention and NVFP4/B12X kernels.
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) for native Blackwell-family kernel infrastructure.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [llama-benchy](https://github.com/XiongjieDai/llama-benchy) for evaluation tooling.

## License

Repository code is MIT licensed. Model and upstream runtime components retain their own licenses.