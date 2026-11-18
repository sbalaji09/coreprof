# Calibration: analytical model vs. measured GPU

**Purpose.** Vaibhav's analytical model of the chip must reproduce measured GPU behavior before
its LoopCore predictions are trusted. This file pairs each measured quantity on two GPUs (L40S,
H100) with the model's prediction for the same quantity on the same hardware.

## How to calibrate

No model predictions are recorded yet, so the analytical model is **not calibrated** and its
LoopCore numbers should not be quoted. To check it:

1. Run the model for each GPU (bf16 Ouro-1.4B, one recurrent step, decode) and fill in
   `calibration_predictions.json` (L40S) and `calibration_predictions_h100.json` (H100): the six
   summary quantities **and the step time at every grid point** (`"grid": {"BxC": ms}`, 46 and 48
   points), same units as the tables; leave unknowns `null`.
2. `python calibrate.py --gpu l40s --pred calibration_predictions.json --write` and
   `python calibrate.py --gpu h100 --pred calibration_predictions_h100.json --write` recompute
   the measured values from the raw run files and fill in that GPU's block below.
3. **Pass** = every summary quantity within ±10 % (`--tol`) **and**, over the grid, median
   |error| ≤ 10 % and max |error| ≤ 20 %. For scale: the δ + γB + α′BC formula fitted to the
   data itself scores median 1.7 % / max 8.7 % (L40S) and 2.8 % / 14.2 % (H100).

**Target = best GPU step** (decision 4): at each (B, C), the faster of the fused
(`torch.compile` + CUDA graph) and uncompiled CUDA-graph step, both measured 2026-11-16. Fused
is faster at 41/46 L40S points (all but B ≥ 32 at short context) and 48/48 H100 points. Weight
time is the fastest weight-kernel time of either implementation (fused GEMMs are slower at
B=16), since that is what the hardware can do. Rationale: weight time and α′ are the same in
both implementations; only the overhead differs, and the uncompiled ≈ 2 ms of per-kernel launch
cost is a PyTorch artifact LoopCore will not have, so reproducing it would not test the chip model.

## Measured vs. predicted: best GPU step (Ouro-1.4B bf16, one recurrent step, decode)

### L40S

<!-- calibration:l40s:begin -->
**Status:** Calibrated on NVIDIA L40S: **no** (6/6 summary predictions, 46/46 grid points).

Measured: the faster of `results/modal/delta_compile-l40s-20261116-214809/` (fused) and `results/modal/delta_grid-l40s-20261116-184352/` (uncompiled) at each point (fused wins 41/46). Predictions: `calibration_predictions.json`, from loopcore_model.py v1: L40S (fitted). Error % is 0 inside a measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA L40S, best GPU) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 3.77–3.98 ms | fastest weight GEMV/GEMM kernels of either implementation, 1x128 and 16x2048 | 3.77 ms | +0.0 |
| KV cost per cached token | 0.265 µs | M2 fit over the envelope | 0.265 µs | +0.0 |
| weight/KV crossover | 14,244–15,020 cached tokens | weight_ms / kv_us_per_token | 14,244 cached tokens | +0.0 |
| best GPU step time, B=1, C=128 | 4.18 ms | envelope | 4.4 ms | +5.4 |
| fixed cost δ (M2 fit) | 4.32 ms | M2 fit over the envelope | 4.32 ms | +0.0 |
| non-weight GPU time (attention + small kernels + gaps), B=1, C=128 | 0.406 ms | step − weight kernels of the fastest implementation | 0.632 ms | +55.8 ✗ |

**Step time over the whole grid (46 points):** median |error| 1.8 %, max -8.7 % at B=32, C=128 (pass: needs median ≤ 10 %, max ≤ 20 %)

**Model inputs:**

- weight_bytes: 2466648064
- weight_bandwidth_GBps: 744.4628959735501
- kv_bytes_per_token: 196608.0
- kv_bandwidth_GBps: 742.5228827114364
- overhead_term: 170 weight kernels x 2.70 us + 0.545 ms per step + 45.9 us/seq
- fitted_on: fitted: l40s envelope (results/modal/delta_compile-l40s-20261116-214809, results/modal/delta_grid-l40s-20261116-184352) + results/modal/delta_compile-l40s-20261116-214809 breakdown
<!-- calibration:l40s:end -->

### H100 80GB HBM3

<!-- calibration:h100:begin -->
**Status:** Calibrated on NVIDIA H100 80GB HBM3: **no** (6/6 summary predictions, 48/48 grid points).

Measured: the faster of `results/modal/delta_compile-h100-20261116-214809/` (fused) and `results/modal/delta_grid-h100-20261116-184347/` (uncompiled) at each point (fused wins 48/48). Predictions: `calibration_predictions_h100.json`, from loopcore_model.py v1: H100 (predicted from L40S (fitted)). Error % is 0 inside a measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA H100 80GB HBM3, best GPU) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 1.49–1.77 ms | fastest weight GEMV/GEMM kernels of either implementation, 1x128 and 16x2048 | 1.4 ms | -5.8 |
| KV cost per cached token | 0.0767 µs | M2 fit over the envelope | 0.0656 µs | -14.5 ✗ |
| weight/KV crossover | 19,435–23,025 cached tokens | weight_ms / kv_us_per_token | 21,403 cached tokens | +0.0 |
| best GPU step time, B=1, C=128 | 2.14 ms | envelope | 2.16 ms | +0.5 |
| fixed cost δ (M2 fit) | 2.12 ms | M2 fit over the envelope | 2.1 ms | -1.2 |
| non-weight GPU time (attention + small kernels + gaps), B=1, C=128 | 0.628 ms | step − weight kernels of the fastest implementation | 0.751 ms | +19.5 ✗ |

**Step time over the whole grid (48 points):** median |error| 6.3 %, max -17.8 % at B=2, C=4096 (pass: needs median ≤ 10 %, max ≤ 20 %)

**Model inputs:**

- weight_bytes: 2466648064
- weight_bandwidth_GBps: 3004.919376477507
- kv_bytes_per_token: 196608.0
- kv_bandwidth_GBps: 2997.0887868356617
- overhead_term: 170 weight kernels x 3.43 us + 0.694 ms per step + 45.9 us/seq
- fitted_on: predicted from fitted: l40s envelope (results/modal/delta_compile-l40s-20261116-214809, results/modal/delta_grid-l40s-20261116-184352) + results/modal/delta_compile-l40s-20261116-214809 breakdown; h100 datasheet clock/compute; measured bandwidth
<!-- calibration:h100:end -->

## Reference: earlier uncompiled targets (superseded 2026-11-16)


### Uncompiled CUDA-graph step, L40S (`results/modal/delta_grid-l40s-20261112-172748/`, the run WEEK5.md uses)

| quantity | measured (NVIDIA L40S) | source |
|---|---|---|
| weight streaming time per loop step | 3.77–3.97 ms | delta_split.json kernel_breakdown (weight GEMV/GEMM kernels, 1x128 and 16x2048) |
| KV cost per cached token | 0.265 µs | graph_grid.csv, M2 fit |
| weight/KV crossover | 14,238–15,008 cached tokens | weight_ms / kv_us_per_token |
| best-case step time, B=1, C=128 (CUDA graph, static cache) | 5.82 ms | graph_grid.csv |
| fixed cost δ (M2 fit) | 6.13 ms | graph_grid.csv, M2 fit |
| non-weight GPU overhead (small kernels + gaps) | 2.05 ms | step_ms_b1_c128 − weight kernels at 1x128 |

### Uncompiled CUDA-graph step, H100 (`results/modal/delta_grid-h100-20261116-184347/`)

| quantity | measured (NVIDIA H100 80GB HBM3) | source |
|---|---|---|
| weight streaming time per loop step | 1.49–1.77 ms | delta_split.json kernel_breakdown (weight GEMV/GEMM kernels, 1x128 and 16x2048) |
| KV cost per cached token | 0.0758 µs | graph_grid.csv, M2 fit |
| weight/KV crossover | 19,666–23,299 cached tokens | weight_ms / kv_us_per_token |
| best-case step time, B=1, C=128 (CUDA graph, static cache) | 4.23 ms | graph_grid.csv |
| fixed cost δ (M2 fit) | 4.62 ms | graph_grid.csv, M2 fit |
| non-weight GPU overhead (small kernels + gaps) | 2.74 ms | step_ms_b1_c128 − weight kernels at 1x128 |

Notes:

- On both GPUs the M2 fit overpredicts the smallest step (L40S: δ 6.13 vs 5.82 ms measured;
  H100: δ 4.62 vs 4.23 ms). Compare the model to each separately.
- δ varies between containers: a second L40S run (`delta_grid-l40s-20261116-184352`) gave
  δ = 6.38 ms (+4 %) with the same weight (3.77 ms) and α′ (0.266 µs). Treat ±0.25 ms on δ as noise.

## Measured reference values

- Looped-block weights: 2,466,652,162 bytes (bf16). KV: 196,608 bytes per cached token per loop.
  LM head (once per token, not per loop): 201.3 MB, not tied to the embedding; 0.33 ms (L40S),
  0.094 ms (H100) per step, graph, B=1.

| | L40S | H100 80GB HBM3 |
|---|---:|---:|
| spec bandwidth | 864 GB/s | 3,350 GB/s |
| copy peak (read + write) | 635–659 GB/s | 3,018 GB/s |
| `torch.sum` read test | 691 GB/s | 2,874 GB/s |
| KV read (α′-implied) | ≈ 742 GB/s | ≈ 2,594 GB/s |
| weight kernels, effective | ≈ 654 GB/s (3.77 ms) | ≈ 1,655 GB/s (1.49 ms) |
| weight bytes ÷ copy peak | 3.75–3.89 ms (within ≈ 3 %) | 0.82 ms (**45 % too low**) |
| small kernels + gaps, B=1, C=128 | ≈ 2.0 ms | ≈ 2.6 ms |

- **L40S:** use measured bandwidth, not the 864 GB/s datasheet value (which underpredicts weight
  time by ≈ 25 %). Weight bytes ÷ measured bandwidth is then within ≈ 3 %.
- **H100:** weight bytes ÷ bandwidth does **not** work. The weight GEMVs reach only ≈ 55 % of the
  copy peak at B=1, so the model needs a per-kernel cost (≈ 170 weight GEMV/GEMM kernels per step), not just
  bandwidth. KV reads do scale with bandwidth (≈ 86 % of copy peak).
- The `torch.sum` read test reads below the copy peak on H100 and below the KV kernels on L40S:
  it underestimates the read peak. Use the copy peak or α′-implied KV bandwidth as the ceiling.

## Fused GPU baseline (torch.compile + CUDA graph), 2026-11-16

Details in `results/WEEK5_COMPILE.md`. Since 2026-11-16 the calibration targets above are the
best GPU step, which is this fused step at nearly every point. Its measured values:

| | L40S compiled | H100 compiled |
|---|---:|---:|
| step, B=1, C=128 | 4.18 ms (uncompiled 5.80, same container) | 2.14 ms (uncompiled 4.51) |
| weight kernels, B=1, C=128 | 3.77 ms (unchanged) | 1.52 ms (unchanged) |
| small kernels + gaps, B=1, C=128 | 0.26 ms (was ≈ 1.9) | 0.44 ms (was ≈ 2.8) |
| δ / γ (M2) | 4.21 ms / 0.070 ms | 2.13 ms / 0.051 ms |
| α′ (M2) | 0.263 µs per cached token (unchanged) | 0.0767 µs (unchanged) |

Fusion removes fixed overhead, not memory traffic; γ rises because the compiled batched GEMMs
are slower. Numerically valid: the compiled bf16 step is closer to fp32 than eager bf16.

## int8 weight-only GPU baseline, 2026-11-18

Details in `results/WEEK5_INT8.md`. Best of two int8 paths per point (cast + coordinate-descent tuning
at B=1, the Triton kernel `int8_kernel.py` at B ≥ 2), vs the best bf16 step:

| | L40S | H100 |
|---|---:|---:|
| int8 step, B=1, C=128 | 2.15 ms (bf16 4.18; bf16 + coord-descent 3.67) | 1.33 ms (bf16 2.14; + coord-descent 1.71) |
| int8 vs best bf16, median over the grid | 1.54× (1.07–2.09) | 1.08× (0.99–1.66) |
| int8 vs bf16 at B=2–16, C=128 | 1.67–1.84× | 0.99–1.14× (**kernel-limited: lower bound**) |

Not yet a calibration target. Two gaps before it is: a bf16 + coordinate-descent grid (it speeds up
bf16 by ≈ 12–20 % at B=1, so the bf16 envelope above is not yet the best bf16), and a better H100
int8 kernel at B=2–16.

## Decisions (confirmed 2026-11-15)

1. **Target workload:** interactive decode, batch 1–16, context up to ~8k as the primary regime;
   long context is handled via the KV strategy.
2. **Model and precision:** Ouro-1.4B; weights in int8 (≈ 1.2 GB looped block), int4 (≈ 0.6 GB)
   as a stretch, pending the Week 6 accuracy test.
3. **Architecture outline:** a weight-stationary pipeline of chips (each layer's weights in
   on-chip SRAM, tokens recirculate once per loop) plus a separate attention/KV unit next to
   HBM. KV compression across loops is under evaluation.
4. **Baselines:** best-case GPU step (CUDA graph + static KV cache) on L40S and H100. Never
   compare against the eager HF path.
   **Amended 2026-11-16:** "best-case" means the fastest measured implementation at each
   (B, C): currently the faster of the fused (`torch.compile` + CUDA graph) and uncompiled
   CUDA-graph steps; add better implementations as they are measured. The analytical model is
   calibrated against this envelope over the whole B × C grid on both GPUs. Before any final
   LoopCore claim, the GPU baseline must also include **int8 weight-only** GEMMs, matching
   LoopCore's int8 weights (decision 2); otherwise LoopCore gets ≈ 2× on weight bytes for free.

## Open questions

- Does sharing KV across loops keep GSM8K accuracy? (The Ouro paper summary and the Looped
  Latent Attention paper disagree.)
- How much of the weight win survives on H100? First measurement (2026-11-16): weights are 1.49 of
  4.23 ms at B=1, C=128, so removing them gives at most ≈ 1.5× (vs ≈ 2.9× on L40S); small kernels
  and gaps (≈ 2.6 ms) dominate the H100 step.
- Is ~1 GB of on-chip SRAM across several chips realistic?
- int8 GPU baseline (measured 2026-11-18, see above): L40S done (≈ 1.5× over bf16); H100 at B=2–16
  is limited by our kernel. Is a production int8 GEMV kernel worth adding there?
- Does int8 change early-exit depth? The exit-gate output moved by up to 0.16 (L40S) / 0.44 (H100)
  in the quick check; Week 6 should measure exit depth along with GSM8K accuracy.
