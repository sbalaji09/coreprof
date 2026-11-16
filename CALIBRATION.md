# Calibration: analytical model vs. measured GPU

**Purpose.** Vaibhav's analytical model of the chip must reproduce measured GPU behavior before
its LoopCore predictions are trusted. This file pairs each measured L40S quantity with the
model's prediction for the same quantity on the same hardware.

**Status:** Calibrated on L40S: pending. The model's predictions are not yet recorded here (the
model script is not in this repo). H100 calibration: pending (Week 7).

## Measured vs. predicted (L40S, Ouro-1.4B bf16, one recurrent step, decode)

| quantity | measured (L40S) | source file | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 3.77–3.98 ms (gemvx 3.77 at B=1, C=128; CUTLASS GEMM 3.98 at B=16, C=2048) | `results/modal/delta_grid-l40s-20261112-172748/delta_split.json` (`kernel_breakdown`) | pending | pending |
| KV cost per cached token | 0.265 µs (≈ 742 GB/s of KV read) | `results/modal/delta_grid-l40s-20261112-172748/graph_grid.csv` (M2 fit, n = 46, R² 0.9997) | pending | pending |
| weight/KV crossover | ≈ 14k cached tokens (3.77–3.98 ms ÷ 0.265 µs = 14.2k–15.0k) | derived from the two rows above | pending | pending |
| best-case step time, B=1, C=128 | 5.82 ms (CUDA graph, static cache) | `results/modal/delta_grid-l40s-20261112-172748/graph_grid.csv` | pending | pending |
| fixed cost δ | 6.13 ± 0.03 ms (M2 fit) | `results/modal/delta_grid-l40s-20261112-172748/graph_grid.csv` | pending | pending |
| non-weight GPU overhead (small kernels + gaps) | ≈ 2.0 ms (5.82 − 3.77) | `results/modal/delta_grid-l40s-20261112-172748/delta_split.json` | pending | pending |

Note: the fitted δ (6.13 ms) and the measured smallest step (5.82 ms) differ by 0.3 ms; the M2
fit overpredicts the B=1, C=128 point by its maximum residual (0.37 ms). The model should be
compared to each separately.

## Model inputs

Inputs used by Vaibhav's model (bandwidths, bytes, overhead term): **pending**.

Measured reference values on this hardware, for comparison:

- Looped-block weights: 2,466,652,162 bytes (bf16). KV: 196,608 bytes per cached token per loop.
- Bandwidth: use **measured** bandwidth as the GPU baseline: device-to-device copy peak
  635–659 GB/s (varies per container), KV read ≈ 742 GB/s. The 864 GB/s datasheet value
  predicts 2.85 ms of weight time, ≈ 25 % below the measured 3.77–3.98 ms.

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
- How much of the weight win survives on H100?
- Is ~1 GB of on-chip SRAM across several chips realistic?
