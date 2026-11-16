# Calibration: analytical model vs. measured GPU

**Purpose.** Vaibhav's analytical model of the chip must reproduce measured GPU behavior before
its LoopCore predictions are trusted. This file pairs each measured L40S quantity with the
model's prediction for the same quantity on the same hardware.

## How to calibrate

No model predictions are recorded yet, so the analytical model is **not calibrated** and its
LoopCore numbers should not be quoted. To check it:

1. Run the model for the L40S (bf16 Ouro-1.4B, one recurrent step, decode) and fill in
   `calibration_predictions.json` (same units as the table; leave unknowns `null`).
2. `python calibrate.py --pred calibration_predictions.json --write` recomputes the measured
   values from the raw run files, fills in the table below and the status line, and exits
   non-zero unless every quantity is within the tolerance (`--tol`, default 10 %).

## Measured vs. predicted (L40S, Ouro-1.4B bf16, one recurrent step, decode)

<!-- calibration:begin -->
**Status:** Calibrated on L40S: **no** (0/6 predictions recorded, tolerance ±10 %). H100 calibration: pending (Week 7).

Predictions: `calibration_predictions.json`. Error % is 0 inside the measured range, else the distance to its nearest end.

| quantity | measured (L40S) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 3.77–3.97 ms | delta_split.json kernel_breakdown (gemvx 1x128, CUTLASS GEMM 16x2048) | pending | pending |
| KV cost per cached token | 0.265 µs | graph_grid.csv, M2 fit | pending | pending |
| weight/KV crossover | 14,238–15,008 cached tokens | weight_ms / kv_us_per_token | pending | pending |
| best-case step time, B=1, C=128 (CUDA graph, static cache) | 5.82 ms | graph_grid.csv | pending | pending |
| fixed cost δ (M2 fit) | 6.13 ms | graph_grid.csv, M2 fit | pending | pending |
| non-weight GPU overhead (small kernels + gaps) | 2.05 ms | step_ms_b1_c128 − gemvx | pending | pending |

**Model inputs:**

- weight_bytes: pending
- weight_bandwidth_GBps: pending
- kv_bytes_per_token: pending
- kv_bandwidth_GBps: pending
- overhead_term: pending
<!-- calibration:end -->

Note: the fitted δ (6.13 ms) and the measured smallest step (5.82 ms) differ by 0.3 ms; the M2
fit overpredicts the B=1, C=128 point by its maximum residual (0.37 ms). The model should be
compared to each separately.

## Measured reference values

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
