# Week 5: splitting δ into weight transfer vs. software overhead

> **Note (2026-11-16):** every GPU number and speedup bound here is against the *uncompiled*
> CUDA-graph step. A fused `torch.compile` step is up to 1.4× (L40S) / 2.1× (H100) faster at small
> batch and is now the baseline (CALIBRATION.md, decision 4); see `results/WEEK5_COMPILE.md`.

**Question.** Week 4 fit one recurrent step of Ouro-1.4B as t = δ + α′·(B·C) with δ ≈ 8.0 ms.
How much of δ is moving the looped block's weights from DRAM, and how much is software
overhead? Does the simple prediction *weight bytes ÷ memory bandwidth* get the weight part
right? And what is a first, bounded estimate of what LoopCore (weights on-chip) could save?

**Answer in one line.** On the GPU itself, δ ≈ 5.8 ms = **≈ 3.8 ms of weight streaming** (which
matches weight bytes ÷ measured bandwidth to within ~3 %) **+ ≈ 2.0 ms of non-weight
overhead** (≈ 1,160 small unfused kernels plus gaps). Week 4's larger δ also included host
(CPU/Python/launch) time, and its KV slope was inflated ≈ 3× by the HF cache. With
weights free, a step gets **≈ 2.9× faster at small batch × context but only ≈ 1.1–1.3× past
~30k cached tokens**, where KV reads dominate.

## Setup

- Ouro-1.4B (rev 574fa66), bf16, one recurrent step (24 layers + norm + exit gate, loop 0),
  decode (one new token per sequence). L40S on Modal, torch 2.5.1, transformers 4.57.1, SDPA.
  Looped-block weights: 2.467 GB. KV: 196.6 kB per cached token per loop.
- Three implementations of the *same* step, all in one container:
  - **eager_hf**: LoopProf's path (Week 4): HF `UniversalTransformerCache`, which appends each
    new token with `torch.cat`, copying that loop's cache every step.
  - **eager_static**: same layers, but a preallocated KV buffer written in place.
  - **graph_static**: eager_static captured as a CUDA graph and replayed: no Python, no
    per-kernel launches. This is the best-case GPU time for this code.
- Correctness: with the same cache contents, the static step reproduces the HF step exactly
  (relative error 0, gate error 0, B = 2, C = 64). `selfcheck.py` passed on both runs.
- Timing: CUDA events, 10 warmup + 50 reps, median reported (IQR ≤ 0.07 ms for graph runs).
  Kernel breakdown: `torch.profiler` kernel durations (GPU time only, no host gaps).
- Bandwidth references: L40S spec 864 GB/s; measured device-to-device copy 635–659 GB/s
  (varies per container).
- Runs: `results/modal/delta_split-l40s-20261112-170226/` (three implementations, launch
  microbenchmarks, layer sweep) and `results/modal/delta_grid-l40s-20261112-172748/`
  (kernel breakdown + graph_static over B ∈ {1..64} × C ∈ {128..8192}, 46 points).

## Results

### 1. Where one step's time goes (B = 1, C = 128, KV traffic negligible)

| | ms | evidence |
|---|---:|---|
| LoopProf step (eager_hf) | 11.7 | this run; Week 4 fit δ = 8.0 [7.5–8.5] on a different container |
| best-case GPU step (graph_static) | **5.82** | grid run (5.95 in the first run) |
| … weight streaming (GEMV kernels) | **3.77** | profiled: cuBLAS `gemvx`, 218 kernels |
| … other GPU kernels | **1.64** | 1,163 kernels: norms, SiLU/mul, rotary, residual adds, cache writes, copies |
| … gaps between kernels | 0.25 | graph time − kernel-busy time (5.82 − 5.57) |
| host overhead (eager − graph) | **≈ 6–9** | 1,379 kernels × ≈ 11 µs of Python + launch each; host-dependent |

- **Kernels per step: 1,379** (≈ 57 per layer). Launch microbenchmark: 4.8 µs of CPU per eager
  launch, 0.84 µs per op inside a graph.
- **The eager step is CPU-bound.** eager_static sits at ≈ 15 ms for every config up to ~32k
  cached tokens, regardless of B·C, then meets the graph curve (B·C = 32,768: 16.0 vs 15.5 ms).
  This is the real cause of Week 4's flat-latency region and of its host-to-host variation in δ
  (Week 4: 20.6 vs 9.8 ms at B = 1 on two containers).

### 2. Prediction check: weight bytes ÷ bandwidth

| prediction | ms | vs measured 3.77–3.98 ms |
|---|---:|---|
| 2.467 GB ÷ 864 GB/s (spec) | 2.85 | **too low by ≈ 25 %** |
| 2.467 GB ÷ 635 GB/s (measured copy peak, same container) | **3.89** | **within ≈ 3 %** |

The weight-streaming kernels run at essentially full achievable bandwidth (≈ 650 GB/s at
B = 1). At B = 16, C = 2048 the weight GEMMs switch to tensor-core CUTLASS kernels and still
take 3.98 ms. **Weight transfer is ≈ 3.8–4.0 ms per step, independent of B in this range.**
For LoopCore's model: use measured, not datasheet, bandwidth for the GPU baseline.

### 3. Best-case GPU step over B × C (LoopProf v1)

| model | δ (ms) | γ (ms per sequence) | α′ (µs per cached token) | R² | max residual |
|---|---:|---:|---:|---:|---:|
| Week 4, eager_hf (M2) | 7.99 [7.50–8.48] | −0.009 ± 0.011 | 0.925 [0.888–0.962] | 0.988 | — |
| **Week 5, graph_static (M2)** | **6.13 ± 0.03** | **0.015 ± 0.001** | **0.265 ± 0.001** | **0.9997** | 0.37 ms |

- M2 (δ + γB + α′·BC) is again the right form. M1 (δ + γB + αC) fails (R² 0.54); M3 adds
  nothing (α = 0.000001 ± 0.000013, ΔAIC = +2).
- **The KV term was inflated ≈ 3× in Week 4** (0.80–0.93 vs 0.265 µs per token). HF's cache
  copies the whole loop cache every step; a static cache only reads it.
- 0.265 µs per token = **742 GB/s of KV read**, 86 % of the 864 GB/s spec. It exceeds the
  copy "peak" because a copy reads *and* writes; pure reads run faster. Achieved bandwidth
  on the grid rises from ≈ 430 GB/s at small B·C (weights + overhead) to ≈ 690 GB/s at large
  B·C (KV-dominated).
- γ is small but now resolved: ≈ 1 ms extra at B = 64.

### 4. First LoopCore estimate: step time if weight streaming cost nothing

Weight term = 3.8 ms (profiled), removed from the best-case GPU step; everything else
(KV from DRAM, small kernels, gaps) unchanged.

| cached tokens B·C | example | GPU step (ms) | without weights (ms) | speedup bound |
|---:|---|---:|---:|---:|
| 128 | B=1, C=128 | 5.82 | 2.02 | **2.9×** |
| 1,024 | B=8, C=128 | 6.56 | 2.76 | 2.4× |
| 4,096 | B=8, C=512 | 7.38 | 3.58 | 2.1× |
| 16,384 | B=8, C=2048 | 10.63 | 6.83 | 1.6× |
| 32,768 | B=16, C=2048 | 15.08 | 11.28 | 1.3× |
| 65,536 | B=32, C=2048 | 24.00 | 20.20 | 1.2× |
| 131,072 | B=16, C=8192 | 40.99 | 37.19 | 1.1× |

- **Crossover: KV time equals weight time at ≈ 14k cached tokens** (3.8 ms ÷ 0.265 µs).
  Below it, on-chip weights are the dominant win; above it, LoopCore must also attack KV
  (on-chip or compressed/quantized KV, or fewer KV bytes per loop) to matter.
- This is an **upper bound on the weight-only effect**: it treats on-chip weight reads as
  free and ignores LoopCore's own compute and SRAM limits. It is also not an upper bound on
  LoopCore overall: the 1.6 ms of small kernels is a software artifact that a fused design
  would largely remove. Vaibhav's analytical model replaces this table.
- Compared against LoopProf's eager path instead, the "speedup" would look 2–3× larger. That
  comparison is not fair to the GPU and should not be quoted.

## Comparison with Week 4

| Week 4 conclusion | Week 5 |
|---|---|
| δ ≈ 8 ms per step, weights ≥ ~4 ms of it is not explained | **Explained:** 3.8 ms weights + 2.0 ms GPU-side overhead + host time on top (eager only) |
| α′ = 0.925 µs per cached token (≈ 213 GB/s, 35–47 % utilization) | **Mostly an HF-cache artifact:** 0.265 µs per token (742 GB/s) with a static cache |
| Time depends on B·C, not B, with a long flat region | Holds (M2), but the flat region was **CPU-bound**; on the GPU alone time rises from the first token |
| Crossover δ/α′ ≈ 8,640 tokens | **≈ 23k tokens** (6.13 / 0.265) for the whole δ; **≈ 14k** for weights alone |

## Caveats

- **One GPU (L40S), one run of each experiment.** The graph numbers are tight (IQR ≤ 0.07 ms)
  and two containers agreed at B = 1 (5.95 vs 5.82 ms), but eager numbers depend on the host
  CPU and vary between containers.
- **The layer-count sweep does not isolate weights.** Removing layers removes their norm,
  rotary and attention kernels too, so its slope (0.26 ms per layer) is all per-layer cost;
  the "395 GB/s effective weight bandwidth" in `delta_split.md` is not a weight bandwidth.
  The profiled GEMV time replaces it.
- **Report errors fixed after the first run:** the first `delta_split.md` mixed layer-sweep
  rows into the KV and speedup tables (the ×10⁶ speedups) and showed a negative "HF cache
  overhead" (host noise between eager runs). The grid run's breakdown counted flash-attention
  kernels as GEMMs (their names contain "cutlass"); 16 × 2048's "12.8 ms gemm" is 4.0 ms of
  GEMM + 8.8 ms of attention. Fixed in `delta_split.py` (commit fc7ed97); the numbers above
  are corrected.
- **Static cache values are random.** Decode timing does not depend on KV values; the
  correctness check uses real prefilled caches.
- **Loop 0 only, decode only.** Other loops read their own cache of the same size, so per-step
  cost should match; not measured here. Points with > 30 GB of KV (B·C ≥ ~150k) were skipped.
- **The measured copy peak is a read + write reference.** It undercounts what pure reads
  (weights, KV) can reach, so "utilization > 100 % of copy peak" in the grid output is
  expected, not an error.

## Files

- `results/modal/delta_split-l40s-20261112-170226/`: `delta_split.md`, `delta_split.json`,
  `delta_split_steps.csv`, `log.txt`, `env.txt`
- `results/modal/delta_grid-l40s-20261112-172748/`: `delta_split_grid.md`, `delta_split.json`
  (kernel breakdown, fits), `graph_grid.csv`, `log.txt`, `env.txt`
- Code: `delta_split.py` (`--grid` for the v1 sweep); Modal tasks `delta_split`, `delta_grid`
  in `modal_app.py`
