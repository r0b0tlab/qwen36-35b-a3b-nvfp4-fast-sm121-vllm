#!/usr/bin/env python3
"""Capture a sanitized runtime manifest and native-path evidence for a release run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

REQUIRED_LOG_MARKERS = {
    "compressed_tensors": "quantization=compressed-tensors",
    "fp8_dense": "Selected CutlassFP8ScaledMMLinearKernel",
    "nvfp4_dense": "Using FlashInferCutlassNvFp4LinearKernel",
    "target_moe_b12x": "Using 'FLASHINFER_B12X' NvFp4 MoE backend",
    "draft_moe_triton": "Using TRITON Unquantized MoE backend",
    "flashinfer_attention": "Using FLASHINFER attention backend",
    "effective_piecewise": "setting cudagraph_mode=PIECEWISE",
}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode()


def metric_value(text: str, metric: str) -> float | None:
    values = []
    for line in text.splitlines():
        if line.startswith(metric + "{") or line.startswith(metric + " "):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    return sum(values) if values else None


def optional_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--container", default=os.getenv("CONTAINER", "qwen36-fast-vllm"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "18080")))
    parser.add_argument("--model-revision", default="11fb65d632976b5f00cb42f157562238cf0c71e0")
    args = parser.parse_args()

    run = Path(args.run_dir)
    evidence = run / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    env = {}
    for item in inspect["Config"].get("Env", []):
        key, _, value = item.partition("=")
        env[key] = value

    models = json.loads(fetch(f"http://127.0.0.1:{args.port}/v1/models"))
    metrics = fetch(f"http://127.0.0.1:{args.port}/metrics")
    logs = command("docker", "logs", args.container)
    versions = json.loads(
        command(
            "docker", "exec", args.container, "python3", "-c",
            "import json,torch,vllm,flashinfer; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':vllm.__version__,'flashinfer':flashinfer.__version__}))",
        ).strip().splitlines()[-1]
    )
    gpu = command(
        "nvidia-smi", "--query-gpu=name,compute_cap,power.limit", "--format=csv,noheader,nounits"
    ).strip().splitlines()[0].split(",")

    marker_status = {name: marker in logs for name, marker in REQUIRED_LOG_MARKERS.items()}
    missing = [name for name, present in marker_status.items() if not present]
    if missing:
        raise SystemExit(f"missing required native markers: {', '.join(missing)}")
    if env.get("KV_CACHE_DTYPE") != "fp8":
        raise SystemExit(f"release profile requires FP8 KV; got {env.get('KV_CACHE_DTYPE')!r}")

    draft = metric_value(metrics, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_value(metrics, "vllm:spec_decode_num_accepted_tokens_total")
    acceptance = accepted / draft if draft else None
    kv_match = re.search(r"GPU KV cache size:\s*([0-9,]+) tokens", logs)

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
        "served_model": models["data"][0]["id"],
        "model_revision": args.model_revision,
        "max_model_len": models["data"][0].get("max_model_len"),
        "hardware": {
            "gpu": gpu[0].strip(),
            "compute_capability": gpu[1].strip(),
            "power_limit_w": optional_float(gpu[2]),
            "architecture": "aarch64",
        },
        "runtime": versions,
        "image_id": inspect["Image"],
        "profile": {
            key: env.get(key)
            for key in (
                "KV_CACHE_DTYPE", "MOE_BACKEND", "LINEAR_BACKEND",
                "SPECULATIVE_CONFIG", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION",
                "ATTENTION_BACKEND", "LANGUAGE_MODEL_ONLY",
            )
        },
        "effective_cuda_graph_mode": "PIECEWISE",
        "kv_cache_tokens": int(kv_match.group(1).replace(",", "")) if kv_match else None,
        "mtp_draft_tokens_total": draft,
        "mtp_accepted_tokens_total": accepted,
        "mtp_acceptance_rate": acceptance,
        "native_marker_gate": marker_status,
        "semantic_profile": "FP8 KV + MTP K=2; NVFP4 KV rejected by semantic gate",
    }

    (run / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (evidence / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    (evidence / "native-markers.txt").write_text(
        "\n".join(marker for marker in REQUIRED_LOG_MARKERS.values()) + "\n"
    )
    selected = [
        line for line in logs.splitlines()
        if any(marker in line for marker in REQUIRED_LOG_MARKERS.values())
        or "SpeculativeConfig(method='mtp'" in line
        or "kv_cache_dtype=fp8" in line
    ]
    (evidence / "native-marker-log-lines.txt").write_text("\n".join(selected) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
