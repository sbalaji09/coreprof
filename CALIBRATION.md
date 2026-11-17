# Calibration: analytical model vs. measured GPU

**Purpose.** Vaibhav's analytical model of the chip must reproduce measured GPU behavior before
its LoopCore predictions are trusted. This file pairs each measured quantity on two GPUs (L40S,
H100) with the model's prediction for the same quantity on the same hardware.

## How to calibrate

No model predictions are recorded yet, so the analytical model is **not calibrated** and its
LoopCore numbers should not be quoted. To check it:

1. Run the model for each GPU (bf16 Ouro-1.4B, one recurrent step, decode) and fill in
   `calibration_predictions.json` (L40S) and `calibration_predictions_h100.json` (H100), same
   units as the tables; leave unknowns `null`.
2. `python calibrate.py --gpu l40s --pred calibration_predictions.json --write` and
   `python calibrate.py --gpu h100 --pred calibration_predictions_h100.json --write` recompute
   the measured values from the raw run files, fill in that GPU's table and status line below,
   and exit non-zero unless every quantity is within the tolerance (`--tol`, default 10 %).

## Measured vs. predicted: L40S (Ouro-1.4B bf16, one recurrent step, decode)

<!-- calibration:l40s:begin -->
**Status:** Calibrated on NVIDIA L40S: **no** (0/6 predictions recorded, tolerance ±10 %).

Measured: `results/modal/delta_grid-l40s-20261112-172748/`. Predictions: `calibration_predictions.json`. Error % is 0 inside the measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA L40S) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 3.77–3.97 ms | delta_split.json kernel_breakdown (weight GEMV/GEMM kernels, 1x128 and 16x2048) | pending | pending |
| KV cost per cached token | 0.265 µs | graph_grid.csv, M2 fit | pending | pending |
| weight/KV crossover | 14,238–15,008 cached tokens | weight_ms / kv_us_per_token | pending | pending |
| best-case step time, B=1, C=128 (CUDA graph, static cache) | 5.82 ms | graph_grid.csv | pending | pending |
| fixed cost δ (M2 fit) | 6.13 ms | graph_grid.csv, M2 fit | pending | pending |
| non-weight GPU overhead (small kernels + gaps) | 2.05 ms | step_ms_b1_c128 − weight kernels at 1x128 | pending | pending |

**Model inputs:**

- weight_bytes: pending
- weight_bandwidth_GBps: pending
- kv_bytes_per_token: pending
- kv_bandwidth_GBps: pending
- overhead_term: pending
<!-- calibration:l40s:end -->

## Measured vs. predicted: H100 80GB HBM3

<!-- calibration:h100:begin -->
**Status:** Calibrated on NVIDIA H100 80GB HBM3: **no** (0/6 predictions recorded, tolerance ±10 %).

Measured: `results/modal/delta_grid-h100-20261116-184347/`. Predictions: `calibration_predictions_h100.json`. Error % is 0 inside the measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA H100 80GB HBM3) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 1.49–1.77 ms | delta_split.json kernel_breakdown (weight GEMV/GEMM kernels, 1x128 and 16x2048) | pending | pending |
| KV cost per cached token | 0.0758 µs | graph_grid.csv, M2 fit | pending | pending |
| weight/KV crossover | 19,666–23,299 cached tokens | weight_ms / kv_us_per_token | pending | pending |
| best-case step time, B=1, C=128 (CUDA graph, static cache) | 4.23 ms | graph_grid.csv | pending | pending |
| fixed cost δ (M2 fit) | 4.62 ms | graph_grid.csv, M2 fit | pending | pending |
| non-weight GPU overhead (small kernels + gaps) | 2.74 ms | step_ms_b1_c128 − weight kernels at 1x128 | pending | pending |

**Model inputs:**

- weight_bytes: pending
- weight_bandwidth_GBps: pending
- kv_bytes_per_token: pending
- kv_bandwidth_GBps: pending
- overhead_term: pending
<!-- calibration:h100:end -->

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

The calibration targets above use the **uncompiled** CUDA-graph step. A fused step
(`torch.compile`, details in `results/WEEK5_COMPILE.md`) is faster at small batch and is the
stronger GPU baseline for LoopCore comparisons. Whether to also calibrate against it is open
(see "Open questions"); its measured values, for reference:

| | L40S compiled | H100 compiled |
|---|---:|---:|
| step, B=1, C=128 | 4.18 ms (uncompiled 5.80, same container) | 2.14 ms (uncompiled 4.51) |
| weight kernels, B=1, C=128 | 3.77 ms (unchanged) | 1.52 ms (unchanged) |
| small kernels + gaps, B=1, C=128 | 0.26 ms (was ≈ 1.9) | 0.44 ms (was ≈ 2.8) |
| δ / γ (M2) | 4.21 ms / 0.070 ms | 2.13 ms / 0.051 ms |
| α′ (M2) | 0.263 µs per cached token (unchanged) | 0.0767 µs (unchanged) |

Fusion removes fixed overhead, not memory traffic; γ rises because the compiled batched GEMMs
are slower. Numerically valid: the compiled bf16 step is closer to fp32 than eager bf16.

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

## Open questions

- Does sharing KV across loops keep GSM8K accuracy? (The Ouro paper summary and the Looped
  Latent Attention paper disagree.)
- How much of the weight win survives on H100? First measurement (2026-11-16): weights are 1.49 of
  4.23 ms at B=1, C=128, so removing them gives at most ≈ 1.5× (vs ≈ 2.9× on L40S); small kernels
  and gaps (≈ 2.6 ms) dominate the H100 step.
- Is ~1 GB of on-chip SRAM across several chips realistic?
- Should the calibration targets (and the GPU baseline in decision 4) switch to the fused
  `torch.compile` step? It is up to 2.1× faster at B=1, which makes weights ≈ 70–90 % of a
  small-batch step, so LoopCore's real gain there depends almost entirely on its SRAM read time.
