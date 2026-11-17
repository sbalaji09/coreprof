# Week 5 addendum: a fused (torch.compile) GPU baseline

**Question.** The best-case GPU step used so far (static KV cache + CUDA graph) still runs ~1,380
small unfused kernels per loop step: ≈ 1.6 ms on L40S and ≈ 1.9 ms on H100, more than the H100's
weight time. How much faster is the GPU when those kernels are fused, and what does that do to the
LoopCore comparison?

**Answer in one line.** Fusion removes almost all non-weight overhead at small batch × context
(**1.39× faster on L40S, 2.10× on H100** at B=1, C=128), barely helps once KV reads dominate
(1.03–1.20× at B=16, C=8192), and leaves weight time unchanged. The fused GPU step at B=1 is now
≈ 90 % (L40S) and ≈ 70 % (H100) weight streaming.

## Setup

- Same step, cache, grid and timing as `delta_split.py --grid` (WEEK5.md), with the step wrapped in
  `torch.compile(mode="default", dynamic=False)` and then captured as a CUDA graph (`compiled_graph`).
  One compile per (B, C) after `torch._dynamo.reset()`: ≈ 20–30 s each, **0 graph breaks**.
  Default mode keeps cuBLAS for the GEMMs, so the kernel breakdown still classifies them.
- Same-container reference: each run also times the uncompiled CUDA graph at 3 points (δ varies by
  ≈ ±0.25 ms between containers, so cross-run comparisons are not enough).
- Runs: `results/modal/delta_compile-l40s-20261116-214809/` (46 points) and
  `results/modal/delta_compile-h100-20261116-214809/` (48 points). torch 2.5.1, commit `5984120` (numerics check: `ca1f5b2`).

## Numerics: the compiled step is valid

The runs' own check (compiled vs eager bf16, same cache) shows ≈ 3 % relative error on the hidden
state, above the 2 % threshold, so their reports say "MISMATCH". That threshold was too strict: an
fp32 reference (`delta_split.py --compile-check`, B=16, C=64) shows the compiled step is *closer*
to fp32 than the eager bf16 step on both GPUs, i.e. the difference is bf16 rounding (fused kernels
round less often), not a bug.

| vs fp32 | L40S eager bf16 | L40S compiled | H100 eager bf16 | H100 compiled |
|---|---:|---:|---:|---:|
| hidden-state relative error | 2.4 % | **1.8 %** | 2.4 % | **1.8 %** |
| exit-gate max error | 0.063 | **0.059** | 0.078 | **0.035** |
| top-1 next token = fp32 (of 16) | 14 | **15** | 15 | **16** |

Files: `results/modal/compile_check-{l40s,h100}-20261116-215010/compile_numerics.json`.

## Results

### Same container: uncompiled vs compiled CUDA graph

| GPU | B × C | uncompiled ms | compiled ms | speedup |
|---|---|---:|---:|---:|
| L40S | 1 × 128 | 5.80 | 4.18 | **1.39×** |
| L40S | 16 × 2048 | 15.05 | 14.06 | 1.07× |
| L40S | 16 × 8192 | 40.67 | 39.58 | 1.03× |
| H100 | 1 × 128 | 4.51 | 2.14 | **2.10×** |
| H100 | 16 × 2048 | 7.98 | 5.27 | 1.51× |
| H100 | 16 × 8192 | 15.12 | 12.61 | 1.20× |

### Where a B=1, C=128 step goes (kernel time, ms)

| | kernels | weights (GEMV) | attention | small kernels | gaps | step |
|---|---:|---:|---:|---:|---:|---:|
| L40S uncompiled | 1,381 | 3.77 | 0.16 | 1.61 | 0.26 | 5.80 |
| L40S compiled | 342 | 3.77 | 0.15 | **0.17** | 0.08 | 4.18 |
| H100 uncompiled | 1,381 | 1.51 | 0.19 | 1.89 | 0.93 | 4.51 |
| H100 compiled | 342 | 1.52 | 0.18 | **0.21** | 0.23 | 2.14 |

### Fit t = δ + γB + α′·BC (M2) over the grid

| | δ ms | γ ms/seq | α′ µs/token | R² | max resid ms |
|---|---:|---:|---:|---:|---:|
| L40S uncompiled (2026-11-16) | 6.38 | 0.016 | 0.266 | 0.9995 | 0.61 |
| **L40S compiled** | **4.21 ± 0.02** | **0.070 ± 0.001** | **0.263 ± 0.001** | 0.9999 | 0.27 |
| H100 uncompiled | 4.62 | 0.021 | 0.0758 | 0.9976 | 0.55 |
| **H100 compiled** | **2.13 ± 0.04** | **0.051 ± 0.002** | **0.0767 ± 0.001** | 0.9984 | 0.75 |

- **δ drops by 2.2–2.5 ms; α′ (KV cost) is unchanged.** Fusion removes fixed overhead, not memory
  traffic.
- **γ rises 3–4×.** At larger batches the compiled step's weight GEMMs are slower (L40S 16×2048:
  4.88 vs 3.98 ms; H100: 2.13 vs 1.77 ms), so the compiled advantage fades with B and on L40S
  reverses at B=64 (0.87× vs the earlier uncompiled run). `max-autotune` recovers part of this at
  16×2048 (L40S 13.08 vs 14.06 ms; H100 6.80 vs 5.27 ms, i.e. worse on H100) and does not help at
  B=1; it costs 75–95 s of compile per shape. Not used as the baseline.

## What this means for LoopCore

- **The fused GPU is the stronger baseline** and should replace the uncompiled CUDA graph wherever
  LoopCore is compared to a GPU (decision 4, "best-case GPU step"). At small B it is up to 2× faster.
- **Weights are now nearly the whole small-batch step.** The "if weights were free" bound at B=1,
  C=128 becomes ≈ 10× (L40S, 4.18 / 0.41 ms) and ≈ 3.4× (H100, 2.14 / 0.63 ms). **Do not quote
  these:** with ≈ 0.4–0.6 ms left, the bound is dominated by the assumption that on-chip weight reads
  are free. LoopCore's real small-batch speedup is set almost entirely by its SRAM read time for
  1.2–2.5 GB per loop, which is what Vaibhav's model must supply.
- At long context nothing changes: KV reads dominate on both GPUs, compiled or not.

## Caveats

- One run per GPU. The cross-container comparison (`watch_runs.py`) agrees with the same-container
  one at B=1 within the known ±0.25 ms container noise (H100 uncompiled 4.23 vs 4.51 ms).
- torch 2.5.1 inductor; newer versions or hand-written fused kernels (vLLM/SGLang-style) could be
  faster still, especially the batched GEMMs that make γ worse.
- Loop 0 only, decode only, as in WEEK5.md.
