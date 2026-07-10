#!/usr/bin/env python3
"""Fail-fast audit for the GB10/SM121 native vLLM engine."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return name, bool(ok), detail


def main() -> int:
    import torch
    import vllm

    rows: list[tuple[str, bool, str]] = []
    version = getattr(vllm, "__version__", "unknown")
    rows.append(check("vllm_0.24", "0.24" in version, version))
    capability = torch.cuda.get_device_capability()
    rows.append(check("cuda_capability_sm121", capability == (12, 1), str(capability)))
    rows.append(check("cuda_runtime_13.0", str(torch.version.cuda).startswith("13.0"), str(torch.version.cuda)))

    for module in (
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        "vllm.v1.attention.ops.nvfp4_cache_quant",
        "vllm.v1.attention.backends.flashinfer",
    ):
        try:
            importlib.import_module(module)
            rows.append(check(f"import:{module}", True))
        except Exception as exc:
            rows.append(check(f"import:{module}", False, repr(exc)[:180]))

    quant_source = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/nvfp4_cache_quant.py"
    )
    source = quant_source.read_text(errors="replace") if quant_source.exists() else ""
    rows.append(check("nvfp4_kv_quantizer_present", "nvfp4_quantize_and_cache" in source))
    rows.append(check("nvfp4_kv_real_block_scales", "amax * (1.0 / 6.0)" in source))
    rows.append(check("nvfp4_kv_scale_factors_written", "sf_region" in source and "k_sf" in source))

    try:
        support = bool(torch.ops._C.cutlass_scaled_mm_supports_fp4(121))
    except Exception as exc:
        support = False
        rows.append(check("sm121_cutlass_fp4", False, repr(exc)[:180]))
    else:
        rows.append(check("sm121_cutlass_fp4", support, str(support)))

    failed = [row for row in rows if not row[1]]
    print("AUDIT PASS" if not failed else "AUDIT FAIL")
    for name, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
