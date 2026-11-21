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

**Target = best GPU step** (decision 4): at each (B, C), the fastest of the fused
(`torch.compile` + CUDA graph, 2026-11-16), fused + `coordinate_descent_tuning` (2026-11-20) and
uncompiled CUDA-graph steps. Fused + coordinate-descent wins 45/46 L40S and 41/48 H100 points: inductor's
fused GEMV reductions read the bf16 weights faster than cuBLAS (weight kernels at 1x128: 3.12 vs 3.77 ms
L40S, 1.12 vs 1.52 ms H100), so the best step at B=1, C=128 is 3.75 ms (L40S) and 1.67 ms (H100). Weight
time is the fastest weight-kernel time of either implementation (fused GEMMs are slower at
B=16), since that is what the hardware can do. Rationale: weight time and α′ are the same in
both implementations; only the overhead differs, and the uncompiled ≈ 2 ms of per-kernel launch
cost is a PyTorch artifact LoopCore will not have, so reproducing it would not test the chip model.

## Measured vs. predicted: best GPU step (Ouro-1.4B bf16, one recurrent step, decode)

### L40S

<!-- calibration:l40s:begin -->
**Status:** Calibrated on NVIDIA L40S: **no** (6/6 summary predictions, 46/46 grid points).

Measured: the fastest of `results/modal/delta_compile-l40s-20261116-214809/`, `results/modal/delta_grid-l40s-20261116-184352/`, `results/modal/delta_bf16_cd-l40s-20261120-010859/` at each point (wins: compiled_graph 1, compiled_graph+cd 45 of 46). Predictions: `calibration_predictions.json`, from loopcore_model.py v1.2 kernel-sum: results/modal/delta_kernel_bench-l40s-20261118-123411/kernel_bench.json. Error % is 0 inside a measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA L40S, best GPU) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 3.12–3.62 ms | fastest weight GEMV/GEMM kernels of either implementation, 1x128 and 16x2048 | 3.78 ms | +4.3 |
| KV cost per cached token | 0.262 µs | M2 fit over the envelope | 0.263 µs | +0.2 |
| weight/KV crossover | 11,902–13,820 cached tokens | weight_ms / kv_us_per_token | 14,386 cached tokens | +4.1 |
| best GPU step time, B=1, C=128 | 3.75 ms | envelope | 4.32 ms | +15.3 ✗ |
| fixed cost δ (M2 fit) | 4.07 ms | M2 fit over the envelope | 4.44 ms | +9.2 |
| non-weight GPU time (attention + small kernels + gaps), B=1, C=128 | 0.627 ms | step − weight kernels of the fastest implementation | 0.541 ms | -13.8 ✗ |

**Step time over the whole grid (46 points):** median |error| 3.1 %, max -23.0 % at B=64, C=128 (**fail**: needs median ≤ 10 %, max ≤ 20 %)

**Model inputs:**

- kernel_counts: {'gemm': {'2048x2048': 96, '5632x2048': 48, '2048x5632': 24}, 'small': 126, 'attention_calls': 24}
- microbenchmarks: results/modal/delta_kernel_bench-l40s-20261118-123411
- measured_copy_peak_GBps: 635.3080904340032
- measured_read_peak_GBps: 745.538045208966
<!-- calibration:l40s:end -->

### H100 80GB HBM3

<!-- calibration:h100:begin -->
**Status:** Calibrated on NVIDIA H100 80GB HBM3: **no** (6/6 summary predictions, 48/48 grid points).

Measured: the fastest of `results/modal/delta_compile-h100-20261116-214809/`, `results/modal/delta_grid-h100-20261116-184347/`, `results/modal/delta_bf16_cd-h100-20261120-010911/` at each point (wins: compiled_graph 7, compiled_graph+cd 41 of 48). Predictions: `calibration_predictions_h100.json`, from loopcore_model.py v1.2 kernel-sum: results/modal/delta_kernel_bench-h100-20261118-123424/kernel_bench.json. Error % is 0 inside a measured range, else the distance to its nearest end.

| quantity | measured (NVIDIA H100 80GB HBM3, best GPU) | source | model prediction | error % |
|---|---|---|---|---|
| weight streaming time per loop step | 1.12–1.54 ms | fastest weight GEMV/GEMM kernels of either implementation, 1x128 and 16x2048 | 1.47 ms | +0.0 |
| KV cost per cached token | 0.0785 µs | M2 fit over the envelope | 0.0775 µs | -1.2 |
| weight/KV crossover | 14,300–19,558 cached tokens | weight_ms / kv_us_per_token | 18,953 cached tokens | +0.0 |
| best GPU step time, B=1, C=128 | 1.67 ms | envelope | 2.01 ms | +20.2 ✗ |
| fixed cost δ (M2 fit) | 2.04 ms | M2 fit over the envelope | 2.18 ms | +7.0 |
| non-weight GPU time (attention + small kernels + gaps), B=1, C=128 | 0.548 ms | step − weight kernels of the fastest implementation | 0.538 ms | -1.9 |

**Step time over the whole grid (48 points):** median |error| 7.6 %, max -33.1 % at B=64, C=128 (**fail**: needs median ≤ 10 %, max ≤ 20 %)

**Model inputs:**

- kernel_counts: {'gemm': {'2048x2048': 96, '5632x2048': 48, '2048x5632': 24}, 'small': 126, 'attention_calls': 24}
- microbenchmarks: results/modal/delta_kernel_bench-h100-20261118-123424
- measured_copy_peak_GBps: 3038.93791511304
- measured_read_peak_GBps: 2982.4837194629968
<!-- calibration:h100:end -->

**Model status (2026-11-20, against the updated targets):** `loopcore_model.py` v1.2 (recorded predictions) fails
the B=1 step (+15 % L40S, +20 % H100), the L40S B=1 non-weight time (−14 %; H100 −1.9 %) and the grid max
(−23 % L40S, −33 % H100); medians pass (3.1 %, 7.6 %). Its weight term uses cuBLAS GEMV microbenchmarks, so it
does not know the faster coordinate-descent reductions; the next step would be to microbenchmark those. The v1
clock-transfer model now predicts H100 at median 11.1 % (fails; its test is marked xfail). Even the δ + γB + α′BC
formula fitted to the H100 envelope itself fails the max rule (+24.7 % at B=1): the coordinate-descent path only
fuses at B=1, so the envelope dips sharply there.

**Target noise (2026-11-20, `delta_noise-*` runs, fused step at B=1, C=128, one container each):** L40S
non-weight time 0.315, 0.315, 0.318 ms (and 0.32 in the probe run): tight; the 0.41 ms of the 2026-11-16 run is
the outlier. H100: 0.41, 0.50, 0.55 ms (plus 0.52–0.63 earlier): ≈ ±20 % between containers, comparable to the
tolerance.

**Earlier status (2026-11-18, `loopcore_model.py` v1.2 kernel-sum, branch `sid/loopcore-model-fixes`):** both
GPUs predicted from per-kernel microbenchmarks only (no step timings, no clock scaling). Fails as recorded above on
two items. (1) B=1 non-weight time: L40S +33 % (a real miss), H100 −14 %. Note this target is small (0.4–0.6 ms,
step minus weight kernels) and varies ±10–12 % between containers (fused 1x128: L40S 0.32–0.41 ms over 2 runs,
H100 0.52–0.63 ms over 3), so the H100 value is inside the measurement spread. (2) Grid max at B ≥ 32 (−26 % L40S,
−46 % H100): the compiled step's batched GEMMs are slower than isolated (or lone-compiled) cuBLAS, which no
microbenchmark tried reproduces. The v1 clock-transfer model passes the grid rule on both GPUs but misses (1).

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
| … vs best bf16 incl. coord-descent (2026-11-20) | **1.48×** (1.06–1.76; 1.74× at 1x128) | **1.05×** (0.97–1.41; 1.26× at 1x128) |
| int8 vs bf16 at B=2–16, C=128 | 1.67–1.84× | 0.99–1.14× (**kernel-limited: lower bound**) |

Not yet a calibration target. The bf16 + coordinate-descent grid is now measured (row above). The H100 int8
kernel at B=2–16 is still the limit: a third, small-M path without tl.dot (2026-11-20) was never faster than
the existing kernel for any shape, so H100 int8 at B=2–16 remains a lower bound.

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

## Week 6 plan: accuracy and exit depth of LoopCore's formats (added 2026-11-21)

**Why these variants.** `loopcore_model.py --kv-levers` (2026-11-21, vs measured H100 int8, q=0.5,
5 µs/layer) shows:
- With the KV cache in HBM, KV reads are 85–95 % of a token at large B·C. Fewer bytes per value is the
  lever: fp8 → int4 KV takes 16×8192 from 3.0× to 5.8×. Sharing caches gives no speed there, because
  each loop still reads C tokens.
- With the KV cache on-chip, sharing and int4 set the chip count: at 16×8192, 106 chips with 4 caches,
  55 with one per pair of loops, 29 with one shared, 16 with one shared in int4.
- Once the KV cache is on-chip, per-layer latency is 84–88 % of a B=1 token, so the next lever is the
  chip's per-layer latency, not the KV cache.
- The GPU can use fp8/int4 KV too: only the on-chip gain is LoopCore's own.

**What runs** (`quality_check.py --variants`, Modal tasks `week6_smoke` then `week6`): all 1319 GSM8K
test questions, 4-shot, greedy, CDB with 32 slots, q = 0.5, one L40S container, bf16 baseline in the
same container.

| Variant | Weights | KV values | KV caches |
|---|---|---|---|
| `bf16` (reference) | bf16 | bf16 | 4 (one per loop) |
| `int8w` | int8, per output channel (`Int8WeightOnlyLinear`, the timed path) | bf16 | 4 |
| `int4w` (stretch) | int4, group 128, rounded in bf16 | bf16 | 4 |
| `fp8kv` | bf16 | fp8 e4m3, no scale, clipped at ±448 (clips counted) | 4 |
| `int4kv` | bf16 | int4, one scale per 32 values | 4 |
| `kv2` | bf16 | bf16 | 2 (loops 1–2, loops 3–4) |
| `kv1` | bf16 | bf16 | 1 shared |
| `chip` | int8 | fp8 | 1 shared |

Shared caches keep, at each position, the KV of the last loop that wrote it: the token's exit loop,
or the group's last loop after copy-forward. The prompt loads each group's last loop. The prompt itself is
prefilled with all 4 caches, as before. This is one sharing rule, not the only one (the Ouro paper and
the Looped Latent Attention paper differ), so a failure here does not rule out sharing.

**Pre-registered rule (PROPOSED 2026-11-21, pending Sid's approval; fixed before any data):**
1. A variant keeps accuracy if the lower end of the 95 % interval on its paired accuracy change vs `bf16`
   is above −2 percentage points. Exact McNemar p and the question-by-question counts are reported too.
2. Exit depth: if a variant's mean exit depth differs from `bf16` by more than 0.10 loops, LoopCore
   claims for that format use its measured depth instead of 3.22. That's an input to the model, not a pass/fail.
3. All eight results are recorded whatever they score. A variant that fails is not re-tuned and re-run
   without a new pre-registered entry here.
