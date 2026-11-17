"""Laptop check (no GPU, no torch): delta_split --grid analysis + report on synthetic data.
Usage: python check_delta_split_report.py   (prints the report, ends with SYNTHETIC TEST OK)
Not named test_*: it stubs torch in sys.modules, which would break the GPU tests under pytest."""
import sys, types, random
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
for m in ("torch", "harness", "loopprof"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["harness"].DTYPES = {"bf16": None}
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
print("SYNTHETIC TEST OK")
