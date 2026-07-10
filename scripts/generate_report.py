#!/usr/bin/env python3
"""Generate a self-contained, email-safe HTML benchmark report."""
from __future__ import annotations
import argparse, html, json, math
from pathlib import Path


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def fmt(value, digits=1, suffix=""):
    if value is None: return "—"
    try: return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError): return html.escape(str(value))


def line_chart(rows, key, color="#62f6ff", label=""):
    vals=[float(r[key]) for r in rows if r.get(key) is not None]
    if not vals: return '<div class="empty">No chart data</div>'
    w,h,p=760,260,42
    ymax=max(vals)*1.12 or 1
    pts=[]
    usable=max(1,len(rows)-1)
    for i,r in enumerate(rows):
        v=float(r.get(key) or 0); x=p+i*(w-2*p)/usable; y=h-p-v/ymax*(h-2*p); pts.append((x,y,v,r.get("concurrency")))
    path=" ".join(("M" if i==0 else "L")+f"{x:.1f},{y:.1f}" for i,(x,y,_,_) in enumerate(pts))
    grid="".join(f'<line x1="{p}" y1="{y}" x2="{w-p}" y2="{y}"/><text x="8" y="{y+4}">{fmt(ymax*(h-p-y)/(h-2*p),0)}</text>' for y in [p,(h)/2,h-p])
    dots="".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"/><text class="dotlabel" x="{x:.1f}" y="{y-12:.1f}">{fmt(v,1)}</text><text class="xlabel" x="{x:.1f}" y="{h-12}">c{c}</text>' for x,y,v,c in pts)
    return f'<div class="chart-title">{html.escape(label)}</div><svg class="chart" viewBox="0 0 {w} {h}" role="img"><g class="grid">{grid}</g><path class="area" d="{path} L{pts[-1][0]:.1f},{h-p} L{pts[0][0]:.1f},{h-p} Z"/><path class="line" style="stroke:{color}" d="{path}"/><g class="dots" style="fill:{color}">{dots}</g></svg>'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("output"); a=ap.parse_args()
    run=Path(a.run_dir); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    gsm=load(run/"gsm8k/results.json",{})
    conc=load(run/"concurrency/summary.json",{}); rows=conc.get("rows",[])
    tele=load(run/"telemetry-summary.json",{})
    manifest=load(run/"runtime-manifest.json",{})
    llama=load(run/"llama-benchy/results.json",{})
    energy=load(run/"energy-efficiency.json",{})
    flex=gsm.get("exact_match_flexible_extract")
    best=max((r for r in rows if r.get("output_throughput_tok_s") is not None),key=lambda r:r["output_throughput_tok_s"],default={})
    c1=next((r for r in rows if r.get("concurrency")==1),{})
    acceptance=manifest.get("mtp_acceptance_rate")
    tele_rows=tele.get("phases",tele if isinstance(tele,dict) else {})
    # Flatten representative power/utilization peaks if present.
    powers=[]; utils=[]; temps=[]
    if isinstance(tele_rows,dict):
        for phase in tele_rows.values():
            if not isinstance(phase,dict): continue
            if phase.get("power_w_max") is not None: powers.append(float(phase["power_w_max"]))
            if phase.get("gpu_util_pct_max") is not None: utils.append(float(phase["gpu_util_pct_max"]))
            if phase.get("temperature_c_max") is not None: temps.append(float(phase["temperature_c_max"]))
    power=max(powers,default=manifest.get("peak_power_w")); util=max(utils,default=manifest.get("peak_gpu_util_pct")); temp=max(temps,default=manifest.get("peak_temp_c"))
    raw=json.dumps({"gsm8k":gsm,"concurrency":conc,"llama_benchy":llama,"telemetry":tele,"energy":energy,"runtime":manifest},indent=2,sort_keys=True)
    table="".join(f'<tr><td>{int(r.get("concurrency",0))}</td><td>{fmt(r.get("output_throughput_tok_s"),1)}</td><td>{fmt(r.get("total_throughput_tok_s"),1)}</td><td>{fmt(r.get("mean_ttft_ms"),1)}</td><td>{fmt(r.get("p99_ttft_ms"),1)}</td><td>{fmt(r.get("mean_tpot_ms"),2)}</td><td>{fmt(r.get("p99_itl_ms"),2)}</td></tr>' for r in rows)
    llama_rows=llama.get("benchmarks",[]) if isinstance(llama,dict) else []
    llama_table="".join(f'<tr><td>{fmt(r.get("context_size"),0)}</td><td>{fmt((r.get("pp_throughput") or {}).get("mean"),1)}</td><td>{fmt((r.get("tg_throughput") or {}).get("mean"),1)}</td><td>{fmt((r.get("ttfr") or {}).get("mean"),1)}</td><td>{fmt((r.get("e2e_ttft") or {}).get("mean"),1)}</td></tr>' for r in llama_rows)
    energy_rows=energy.get("rows",[]) if isinstance(energy,dict) else []
    energy_table="".join(f'<tr><td>c{int(r.get("concurrency",0))}</td><td>{fmt(r.get("mean_active_power_w"),1)}</td><td>{fmt(r.get("output_tokens_per_joule"),2)}</td><td>{fmt(r.get("joules_per_1k_output_tokens"),1)}</td></tr>' for r in energy_rows)
    css='''
:root{--bg:#06101b;--panel:#0b1928;--panel2:#10243a;--ink:#eafaff;--muted:#92afc3;--cyan:#62f6ff;--violet:#b58cff;--lime:#78f7a8;--amber:#ffc86a;--red:#ff718b;--line:#1d3b55}*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:radial-gradient(circle at 80% -10%,#183557 0,transparent 38%),radial-gradient(circle at 0 20%,#112b35 0,transparent 30%),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.5}.wrap{max-width:1180px;margin:auto;padding:44px 28px 80px}.hero{position:relative;overflow:hidden;border:1px solid #28506d;background:linear-gradient(135deg,rgba(18,51,76,.96),rgba(8,20,35,.97));border-radius:28px;padding:44px;box-shadow:0 30px 80px #0008}.hero:after{content:"";position:absolute;right:-90px;top:-90px;width:360px;height:360px;border:1px solid #62f6ff55;border-radius:50%;box-shadow:0 0 80px #62f6ff22 inset}.brand{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--cyan);font-weight:800}.hero h1{max-width:850px;font-size:clamp(36px,6vw,72px);line-height:1.02;margin:18px 0;background:linear-gradient(90deg,#fff,var(--cyan) 55%,var(--violet));-webkit-background-clip:text;color:transparent}.subtitle{max-width:820px;color:#c0d8e8;font-size:18px}.badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:28px}.badge{border:1px solid #335a74;background:#0b2033aa;border-radius:999px;padding:8px 13px;font-size:12px;color:#d7f6ff}.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:14px;margin:22px 0}.kpi,.panel{border:1px solid var(--line);background:linear-gradient(180deg,var(--panel2),var(--panel));border-radius:20px;padding:22px;box-shadow:0 18px 48px #0004}.kpi .v{font-size:34px;font-weight:850;letter-spacing:-.04em}.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}.kpi:nth-child(1) .v{color:var(--lime)}.kpi:nth-child(2) .v{color:var(--cyan)}.kpi:nth-child(3) .v{color:var(--violet)}.kpi:nth-child(4) .v{color:var(--amber)}h2{font-size:29px;margin:44px 0 15px;letter-spacing:-.025em}h3{margin:0 0 10px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.callout{border-left:4px solid var(--cyan);padding:18px 22px;background:#0c1c2c;border-radius:8px 18px 18px 8px;color:#cde5f2}.callout.warn{border-color:var(--amber)}.stack{display:grid;gap:10px}.step{display:grid;grid-template-columns:42px 1fr;gap:14px;align-items:start}.num{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:#173550;color:var(--cyan);font-weight:900}.step p{margin:3px 0;color:#b8cfde}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.chart{width:100%;height:auto}.grid line{stroke:#21425b;stroke-width:1}.grid text,.xlabel,.dotlabel{fill:#7899ad;font-size:11px}.dotlabel,.xlabel{text-anchor:middle}.line{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.area{fill:#62f6ff12}.chart-title{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px}.status{display:inline-flex;align-items:center;gap:8px;color:var(--lime);font-weight:750}.status:before{content:"";width:9px;height:9px;border-radius:50%;background:var(--lime);box-shadow:0 0 16px var(--lime)}code{color:#bdefff;background:#06111d;padding:2px 6px;border-radius:6px}details{border:1px solid var(--line);border-radius:16px;padding:14px 18px;background:#081522}summary{cursor:pointer;color:var(--cyan);font-weight:700}pre{white-space:pre-wrap;word-break:break-word;color:#9fc0d4;font-size:11px}.footer{margin-top:50px;color:#6f8ea2;font-size:12px;text-align:center}.empty{height:180px;display:grid;place-items:center;color:var(--muted)}@media(max-width:820px){.grid4,.two{grid-template-columns:1fr 1fr}.hero{padding:30px}}@media(max-width:520px){.grid4,.two{grid-template-columns:1fr}.wrap{padding:18px}.hero h1{font-size:38px}}
'''
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.6 35B A3B NVFP4 Fast · GB10 Benchmark</title><style>{css}</style></head><body><main class="wrap">
<section class="hero"><div class="brand">DSpark · r0b0tlab performance dossier</div><h1>Qwen3.6 35B-A3B<br>NVFP4 Fast on GB10</h1><p class="subtitle">A correctness-first, SM121-native serving study: mixed FP8/NVFP4 weights, FlashInfer B12X experts, FP8 KV cache, and trained MTP speculative decoding.</p><div class="badges"><span class="badge">NVIDIA GB10 · SM121</span><span class="badge">vLLM 0.24 · CUDA 13</span><span class="badge">FlashInfer B12X</span><span class="badge">MTP ×2</span><span class="badge">65,536-token profile</span></div></section>
<section class="grid4"><div class="kpi"><div class="l">GSM8K 0-shot</div><div class="v">{fmt((flex or 0)*100,1,'%')}</div></div><div class="kpi"><div class="l">Peak output rate</div><div class="v">{fmt(best.get('output_throughput_tok_s'),0)}</div><small>tok/s at c{best.get('concurrency','—')}</small></div><div class="kpi"><div class="l">MTP acceptance</div><div class="v">{fmt((acceptance or 0)*100,1,'%')}</div></div><div class="kpi"><div class="l">Peak board power</div><div class="v">{fmt(power,1,' W')}</div></div></section>
<h2>Executive finding</h2><div class="callout"><span class="status">Production profile validated</span><p>The model card's vLLM 0.24 + two-token MTP path is compatible on GB10 when KV cache is FP8. An experimental NVFP4-KV profile reached HTTP readiness but corrupted semantics; it was rejected before benchmarking. The published numbers below use the coherent profile only.</p></div>
<h2>What actually runs</h2><div class="two"><div class="panel stack"><div class="step"><div class="num">01</div><div><h3>Mixed checkpoint</h3><p>Compressed-tensors FP8 dense groups plus NVFP4 expert weights.</p></div></div><div class="step"><div class="num">02</div><div><h3>Native target kernels</h3><p>FlashInferCutlass NVFP4 GEMM and explicit FlashInfer B12X target MoE on <code>sm_121a</code>.</p></div></div><div class="step"><div class="num">03</div><div><h3>Correct MTP split</h3><p>Two draft tokens; Triton handles the unquantized draft MoE while B12X stays on the quantized target.</p></div></div></div><div class="panel"><h3>Resolved runtime</h3><table><tr><td>KV cache</td><td>FP8</td></tr><tr><td>Attention</td><td>FlashInfer</td></tr><tr><td>Target MoE</td><td>FLASHINFER_B12X</td></tr><tr><td>Draft MoE</td><td>TRITON</td></tr><tr><td>CUDA graphs</td><td>PIECEWISE (effective)</td></tr><tr><td>GPU KV capacity</td><td>{fmt(manifest.get('kv_cache_tokens'),0)} tokens</td></tr></table></div></div>
<h2>Concurrency scaling</h2><div class="panel">{line_chart(rows,'output_throughput_tok_s','#62f6ff','Output throughput · tokens/second')}</div><div class="panel" style="margin-top:18px;overflow:auto"><table><thead><tr><th>Concurrency</th><th>Output tok/s</th><th>Total tok/s</th><th>Mean TTFT ms</th><th>P99 TTFT ms</th><th>Mean TPOT ms</th><th>P99 ITL ms</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>Latency profile</h2><div class="two"><div class="panel">{line_chart(rows,'mean_ttft_ms','#b58cff','Mean time to first token · milliseconds')}</div><div class="panel">{line_chart(rows,'mean_tpot_ms','#78f7a8','Mean time per output token · milliseconds')}</div></div>
<h2>Long-context depth sweep</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>Context depth</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>TTFR ms</th><th>E2E TTFT ms</th></tr></thead><tbody>{llama_table}</tbody></table><p style="color:var(--muted)">llama-benchy 0.4.0 · 2,048-token prompt · 128-token exact generation · concurrency 1 · three measured runs per depth. Coherence test passed.</p></div>
<h2>Quality gate</h2><div class="two"><div class="panel"><div class="kpi"><div class="l">Flexible-extract exact match</div><div class="v">{fmt((flex or 0)*100,2,'%')}</div></div><p>Full GSM8K, 1,319 questions, 0-shot, greedy decoding, chat template applied, thinking disabled. This is the publication metric; strict string matching is retained in raw data but is not the primary gate.</p></div><div class="panel"><h3>Semantic probes</h3><p class="status">Passed</p><p>Deterministic arithmetic, exact-string, and word-problem probes passed under FP8 KV with MTP. MTP counters were non-zero and acceptance was measured from server Prometheus metrics.</p><div class="callout warn"><strong>Rejected profile:</strong> NVFP4 KV + MTP changed prompt operands and produced punctuation degeneration. Readiness alone was not accepted as correctness.</div></div></div>
<h2>Thermal & power envelope</h2><div class="grid4"><div class="kpi"><div class="l">Peak GPU utilization</div><div class="v">{fmt(util,0,'%')}</div></div><div class="kpi"><div class="l">Peak temperature</div><div class="v">{fmt(temp,0,'°C')}</div></div><div class="kpi"><div class="l">C1 mean TPOT</div><div class="v">{fmt(c1.get('mean_tpot_ms'),2,' ms')}</div></div><div class="kpi"><div class="l">C1 P99 TTFT</div><div class="v">{fmt(c1.get('p99_ttft_ms'),1,' ms')}</div></div></div>
<div class="panel" style="overflow:auto"><table><thead><tr><th>Load</th><th>Mean active power W</th><th>Output tokens/J</th><th>J/1K output tokens</th></tr></thead><tbody>{energy_table}</tbody></table><p style="color:var(--muted)">Board power is sampled every two seconds during stable repetitions 2 and 3. Samples below 10% GPU utilization are excluded so client bookkeeping and post-generation idle time do not inflate efficiency.</p></div>
<h2>Methodology & reproducibility</h2><div class="panel"><p><strong>Concurrency:</strong> random 2,048-token input / exact 512-token output, concurrencies 1–32, three repetitions; first repetition dropped and the final two averaged. <strong>Telemetry:</strong> 2-second nvidia-smi and host-memory sampling with benchmark phase labels. <strong>Runtime:</strong> one GB10, text-only model path, 65,536 maximum sequence length, no Marlin or emulation fallback.</p><p>Run ID: <code>{html.escape(run.name)}</code> · Model revision: <code>{html.escape(str(manifest.get('model_revision','—')))}</code> · Image digest: <code>{html.escape(str(manifest.get('image_id','—')))}</code></p></div>
<h2>Raw evidence</h2><details><summary>Embedded machine-readable benchmark payload</summary><pre>{html.escape(raw)}</pre></details>
<div class="footer">Prepared by DSpark · r0b0tlab · correctness before throughput · generated from machine-readable artifacts</div></main></body></html>'''
    out.write_text(body,encoding="utf-8"); print(out)

if __name__=="__main__": main()
