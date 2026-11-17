"""Laptop check (no GPU, no torch): delta_split --grid analysis + report on synthetic data.
Usage: python check_delta_split_report.py   (prints the report, ends with SYNTHETIC TEST OK)
Not named test_*: it stubs torch in sys.modules, which would break the GPU tests under pytest."""
import sys, types, random
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
for m in ("torch", "harness", "loopprof"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["harness"].DTYPES = {"bf16": None}
sys.modules["torch"].nn = types.SimpleNamespace(Module=object)
import delta_split as ds

KVB, WB = 196608, 2466652162
random.seed(0)
rows = []
for C in (128, 256, 512, 1024, 2048, 4096, 8192):
    for B in (1, 2, 4, 8, 16, 32, 64):
        if B * (C + 1) * KVB / 1e9 > 55.6: continue
        t = 6.13 + 0.015 * B + 0.000265 * B * C + random.gauss(0, 0.02)
        rows.append(dict(impl="graph_static", B=B, C=C, layers=24, median_ms=t, achieved_gbps=500.0))
assert len(rows) == 48, len(rows)   # 80 GB GPU: 46 + B=64,C=4096 + B=32,C=8192
bd = lambda gemm, att, copy, other: dict(gemm_ms=gemm, attention_ms=att, copy_ms=copy, other_ms=other,
                                         busy_ms=gemm + att + copy + other, kernels=1381, top_kernels_ms={"k": gemm})
kb = {"eager_static 1x128": bd(3.79, 0.12, 0.62, 1.19), "graph_static 1x128": bd(3.77, 0.116, 0.56, 1.07)}
for C in (512, 2048, 8192):
    a = 0.000265 * 16 * C
    kb[f"eager_static 16x{C}"] = bd(4.02, a, 0.67, 1.6)
    kb[f"graph_static 16x{C}"] = bd(3.98, a, 0.6, 1.45)
kb["graph_static 16x512"]["attention_ms"] = 0  # a row without attention must be skipped, not crash
out = dict(gpu="NVIDIA H100 80GB HBM3", spec_peak_gbps=3350, measured_peak_gbps=635.0, measured_read_gbps=742.0,
           weight_bytes_looped_block=WB, kv_bytes_per_token_per_loop=KVB,
           predicted_weight_fetch_ms=dict(spec=0.74, measured_peak=3.89, read_peak=3.32), kernel_breakdown=kb,
           lm_head=dict(params=100_000_000, bytes=200_000_000, vocab_size=49152, hidden_size=2048,
                        tie_word_embeddings=False, shares_embedding_tensor=False, embedding_params=100_000_000,
                        rows=[dict(B=b, eager_ms=0.4, graph_ms=0.3, graph_gbps=666.0) for b in (1, 16, 64)]),
           errors={})
ds.analyze_grid(out, rows)
ci = out["analysis"]["calibration_inputs"]
assert abs(ci["delta_ms"] - 6.13) < 0.05 and abs(ci["alpha_prime_us"] - 0.265) < 0.002, ci
assert abs(ci["x1_crossover_tokens"] - 3.77 / 0.000265) / 14226 < 0.01
assert abs(ci["small_kernels_ms"] - 1.63) < 1e-9
t11 = next(r["median_ms"] for r in rows if (r["B"], r["C"]) == (1, 128))
assert abs(ci["gap_ms"] - (t11 - (3.77 + 0.116 + 0.56 + 1.07))) < 1e-9
at = {f"{r['impl']} {r['B']}x{r['C']}": r for r in out["analysis"]["attention_per_token"]}
assert "graph_static 16x512" not in at
assert abs(at["graph_static 16x8192"]["us_per_cached_token"] - 0.265) < 1e-9
assert abs(at["graph_static 16x8192"]["read_gbps"] - KVB / 0.265e-6 / 1e9) < 1e-6
md = ds.report_grid(out)
assert md.index("## Calibration inputs") < md.index("## GPU kernel time") < md.index("## Attention kernels")
assert "% of measured read peak" in md and "Once per token" in md
# missing pieces (failed breakdown / lm_head / no read peak) must still render
out2 = {**out, "kernel_breakdown": {}, "lm_head": None, "measured_read_gbps": None}
ds.analyze_grid(out2, rows[:3]); ds.report_grid(out2)
# v0 report still works with the new predicted_weight_fetch_ms key
print(ds.report(dict(gpu="x", weight_bytes_looped_block=WB, spec_peak_gbps=None, measured_peak_gbps=635.0,
                     measured_read_gbps=742.0, predicted_weight_fetch_ms=dict(spec=None, measured_peak=3.89, read_peak=3.32),
                     errors={})).splitlines()[2])
print(md)
# --compile mode: rows/breakdowns under compiled_graph; calibration must use them, not the uncompiled reference
crow = [{**r, "impl": "compiled_graph", "median_ms": r["median_ms"] - 1.5} for r in rows]
ckb = {k.replace("graph_static", "compiled_graph").replace("eager_static", "compiled"): dict(v) for k, v in kb.items()}
ckb["compiled_graph 1x128"].update(other_ms=0.2, copy_ms=0.1, kernels=400)
ckb["graph_static 1x128"] = kb["graph_static 1x128"]           # uncompiled reference breakdown
outc = {**out, "kernel_breakdown": ckb, "compile_mode": "default", "grid_impl": "compiled_graph",
        "compiled_vs_eager_check": dict(B=2, C=64, mode="default", rel_err_hidden=3e-3, max_abs_err_gate=1e-3,
                                        ok=True, compile_s=42.0, graph_breaks=0),
        "uncompiled_reference": [dict(B=1, C=128, graph_static_ms=5.9, compiled_graph_ms=4.4, speedup=5.9 / 4.4,
                                      saved_ms=1.5)],
        "compile_variants": [dict(mode="max-autotune-no-cudagraphs", B=1, C=128, graph_ms=4.2, compile_s=300.0)]}
ds.analyze_grid(outc, crow)
cc = outc["analysis"]["calibration_inputs"]
assert abs(cc["small_kernels_ms"] - 0.3) < 1e-9, cc["small_kernels_ms"]
assert abs(cc["delta_ms"] - 4.63) < 0.05, cc["delta_ms"]
assert outc["analysis"]["onchip_weights_upper_bound"]["rows"], "bound must use compiled_graph rows"
mdc = ds.report_grid(outc)
assert "torch.compile (default) + CUDA-graph" in mdc and "## torch.compile vs uncompiled" in mdc
assert "| compiled graph | 400 |" in mdc and "1.34×" in mdc and "max-autotune" in mdc
bad = {**outc, "compiled_vs_eager_check": {**outc["compiled_vs_eager_check"], "ok": False}}
assert "MISMATCH" in ds.report_grid(bad)
assert "unverified" in ds.report_grid({**outc, "compiled_vs_eager_check": None})
print(mdc[mdc.index("## torch.compile vs"):mdc.index("## Calibration inputs")])
# --int8 mode: weight time comes from the step-time estimate, not the (unreliable) Triton kernel names
oi = {**outc, "grid_impl": "int8_graph", "weight_bytes_looped_block": 1_240_000_000,
      "kernel_breakdown": {"int8_graph 1x128": dict(gemm_ms=0.0, attention_ms=0.12, copy_ms=0.0, other_ms=2.3,
                                                     busy_ms=2.42, kernels=342, top_kernels_ms={}),
                           "compiled_graph 1x128": dict(gemm_ms=3.77, attention_ms=0.12, copy_ms=0.0, other_ms=0.17,
                                                         busy_ms=4.06, kernels=342, top_kernels_ms={})},
      "int8_vs_bf16_check": dict(B=16, C=64, linears_quantized=168, bf16_weight_bytes=WB, rel_err_hidden=0.03,
                                 max_abs_err_gate=0.05, top1_agreement=0.94, note="n"),
      "int8_reference": [dict(B=1, C=128, bf16_ms=4.18, int8_ms=2.5, speedup=4.18 / 2.5, saved_ms=1.68)],
      "int8_weight_estimate": dict(w_bf16_ms=3.77, t_bf16_ms=4.18, t_int8_ms=2.5, w_int8_est_ms=2.09,
                                   int8_weight_bytes=1_240_000_000, bf16_weight_bytes=WB, int8_gbps=593.0,
                                   bf16_gbps=654.0, weight_time_ratio=2.09 / 3.77, byte_ratio=0.503)}
irows = [{**r, "impl": "int8_graph"} for r in crow]
ds.analyze_grid(oi, irows)
ci8 = oi["analysis"]["calibration_inputs"]
assert ci8["weight_ms"] == 2.09 and "int8 estimate" in ci8["weight_ms_source"], ci8["weight_ms"]
assert oi["analysis"]["onchip_weights_upper_bound"]["rows"][0]["weight_term_ms"] == 2.09
mdi = ds.report_grid(oi)
assert "int8 weight-only + torch.compile" in mdi and "## int8 weight-only vs bf16" in mdi and "1.67×" in mdi
print(mdi[mdi.index("## int8 weight-only vs"):mdi.index("## Calibration inputs")])
print("SYNTHETIC TEST OK")
