# Benchmark evidence

Release run: `20260710T145810Z`

All public claims are generated from machine-readable artifacts under `benchmarks/runs/20260710T145810Z/`. Large raw sample payloads, telemetry streams, local virtual environments, and transient logs are intentionally excluded from Git.

## Quality

- Task: GSM8K test split, all 1,319 samples.
- Setting: 0-shot, chat-completions API, chat template applied, thinking disabled, temperature 0.
- Primary metric: `exact_match,flexible-extract`.
- Artifact: `gsm8k/results.json`.

## Throughput and latency

- Tool: installed vLLM `bench serve` command.
- Dataset: deterministic random prompts.
- Input/output: 2,048 prompt tokens and exactly 512 generated tokens (`--ignore-eos`).
- Concurrency: c1, c2, c4, c8, c16, c32.
- Repetitions: three at each concurrency; repetition 1 is warm-up and the final two are averaged.
- Artifacts: individual JSON files plus `concurrency/summary.json`.

## Context-depth sweep

- Tool: llama-benchy 0.4.0.
- Prompt/generation: 2,048 prompt tokens and exactly 128 generated tokens.
- Context depths: 0, 4,096, 8,192, and 16,384 tokens.
- Repetitions: three per depth.
- Artifact: `llama-benchy/results.json`.

## Telemetry and energy

A two-second sampler records board power, GPU utilization, temperature, clocks, throttle state, and host memory with benchmark-phase labels. The raw JSONL stream is excluded from Git; `telemetry-summary.json` preserves phase-level mean, p95, and maximum values. `energy-efficiency.json` joins mean active power from stable repetitions 2 and 3 to throughput from those same repetitions.

## Runtime proof

`runtime-manifest.json` records the model revision, image ID, runtime versions, resolved serving profile, effective CUDA graph mode, MTP counters and acceptance, and native-marker gates. `evidence/` contains the semantic smoke response and selected startup lines proving the active native paths.

`MANIFEST.sha256` hashes the public release artifacts. Run `sha256sum -c MANIFEST.sha256` from the run directory to verify them.
