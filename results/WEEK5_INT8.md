# Week 5 addendum 2: an int8 weight-only GPU baseline

**Question.** LoopCore plans int8 weights (CALIBRATION.md, decision 2). Against a bf16 GPU it would get
≈ 2× on weight bytes for free, and with the fused step weights are 70–90 % of a small-batch step. How
fast is the GPU with int8 weights?

**Answer in one line.** With a working int8 path, the GPU is **≈ 1.5× faster than bf16 on L40S**
(median over 46 points; 1.7–2.1× at short context) but only **≈ 1.1× on H100** (1.6× at B=1, ≈ 1.0–1.1×
at B=2–16, where our kernel is the limit, so treat H100 int8 as a lower bound on what the hardware can do).

## The int8 paths tried

All quantize the 168 linears of the looped block to int8 with one bf16 scale per output channel
(2.47 → 1.23 GB); embedding, LM head, norms and exit gate stay bf16. All are torch.compile + CUDA graph.

| path | what happens | result |
|---|---|---|
| cast, inductor default | `F.linear(x, w_int8.to(bf16)) * s`; torch 2.5.1 does **not** fuse the cast: it writes bf16 weights every step (`_to_copy` kernels, 1.2–1.9 ms) | no faster than bf16 (L40S 0.98×, H100 0.76× median); `delta_int8-*-20261116-*` |
| cast + `mixed_mm_choice=triton` | no change | `int8_probe-*` |
| cast + `coordinate_descent_tuning` (gpt-fast) | at **B=1** the matmul becomes a fused reduction that reads int8 directly: weight time **0.48–0.49× bf16**; not at B ≥ 2 | best at B=1 |
| Triton int8 kernel (`int8_kernel.py`) | loads int8 tiles, converts in registers, fp32 accumulate; split-K variant chosen per shape if faster (it almost never is) | best at B ≥ 2 (L40S 39/46 points, H100 41/48) |

The int8 baseline is the faster of the last two at each (B, C), the same rule as decision 4.

## Results: best int8 vs best bf16 (fused or uncompiled), per point

| GPU | B × C | best bf16 ms | best int8 ms | speedup | int8 path |
|---|---|---:|---:|---:|---|
| L40S | 1 × 128 | 4.18 | 2.15 | 1.94× | cast + coord-descent |
| L40S | 2 × 128 | 4.44 | 2.61 | 1.70× | kernel |
| L40S | 16 × 128 | 5.92 | 3.22 | 1.84× | kernel |
| L40S | 64 × 128 | 9.26 | 5.41 | 1.71× | kernel |
| L40S | 1 × 2048 | 4.74 | 2.70 | 1.76× | cast + coord-descent |
| L40S | 16 × 2048 | 14.06 | 11.30 | 1.24× | kernel |
| L40S | 16 × 8192 | 39.58 | 36.92 | 1.07× | kernel |
| L40S | **all 46** | | | **median 1.54×** (1.07–2.09) | |
| H100 | 1 × 128 | 2.14 | 1.33 | 1.62× | cast + coord-descent |
| H100 | 2 × 128 | 2.41 | 2.43 | 0.99× | kernel |
| H100 | 16 × 128 | 2.92 | 2.57 | 1.14× | kernel |
| H100 | 64 × 128 | 6.09 | 4.65 | 1.31× | kernel |
| H100 | 1 × 2048 | 2.33 | 1.53 | 1.53× | cast + coord-descent |
| H100 | 16 × 2048 | 5.27 | 4.92 | 1.07× | kernel |
| H100 | 16 × 8192 | 12.61 | 12.40 | 1.02× | kernel |
| H100 | **all 48** | | | **median 1.08×** (0.99–1.66) | |

- **L40S:** int8 delivers close to the byte saving at short context, fading at long context where KV
  reads (unchanged by int8 weights) dominate.
- **H100, B=2–16:** our kernel reaches only ≈ 1.0–1.14×. With N = 2048 it has too few programs for the
  H100's memory system, and split-K's extra zero-fill/cast kernels cost more than they save. Production
  int8 GEMV kernels likely do better, so **H100 int8 at B=2–16 is a lower bound** on the GPU, and a
  LoopCore comparison there would be optimistic for LoopCore.

## Side finding: coordinate-descent tuning also speeds up bf16

The cast + coord-descent runs time their bf16 reference with the same setting (same container):

| bf16 compiled step | without coord-descent | with coord-descent |
|---|---:|---:|
| L40S 1 × 128 | 4.18 ms (4.18 in the kernel run) | **3.67 ms** |
| L40S 16 × 2048 | 13.99 | 13.37 |
| H100 1 × 128 | 2.07 (kernel run) – 2.14 | **1.71 ms** |
| H100 16 × 2048 | 5.19 | 5.07 |

So the best bf16 GPU is ≈ 12 % (L40S) / ≈ 20 % (H100) faster at B=1 than the current bf16 envelope,
and a few % at B=16. Against that bf16, int8 at B=1 is 1.71× (L40S) and 1.29× (H100). The bf16 envelope
(and the calibration targets) need a bf16 + coord-descent grid to be complete.

## Update 2026-11-20: against the best bf16 including coordinate-descent

The bf16 + coordinate-descent grids (`results/modal/delta_bf16_cd-*-20261120-*`) are now in the bf16 envelope.
int8 vs that best bf16: **L40S median 1.48×** (1.06–1.76; 1.74× at 1x128, 1.66× at 64x128, 1.06× at 16x8192),
**H100 median 1.05×** (0.97–1.41; 1.26× at 1x128, 1.00× at 2x128, 1.03× at 16x8192). A third int8 kernel path
for M ≤ 16 without tl.dot (BLOCK_N 4–16, up to 512 programs) was added for the H100 but was never faster than the
plain kernel for any shape (`delta_int8_triton-h100-20261120-010924`), so H100 int8 at B=2–16 remains a lower bound.

## Accuracy (quick check, not GSM8K)

int8 vs bf16 step, eager, B=16, C=64, same cache: hidden state **6.2–6.4 %** (bf16 vs fp32 is ≈ 2.4 %),
top-1 next token **88–94 %** agreement, exit-gate max error **0.16–0.44** (larger on H100). The exit gate
decides loop depth, so the Week 6 test should check exit depth as well as GSM8K accuracy.

## Caveats

- One run per configuration; same-container references at 3 points per run.
- The int8 weight-time estimate (t_int8 − (t_bf16 − w_bf16)) is valid for the kernel runs
  (L40S 0.58×, H100 1.23× of bf16 weight time at B=1); in the coord-descent runs the bf16 GEMMs also
  become Triton reductions, so the name-based breakdown cannot isolate weight time (its estimate there
  is meaningless).
- `int8_kernel.check()` reported max_rel_err ≈ 11.2 in every run. **Correction (2026-11-20):** the cause was the
  check itself, not the kernels: it compared the [M, N] output with a [M, 1, N] reference, which broadcasts to an
  [M, M, N] comparison of different rows (≈ √(2M) = 11.3 at M = 64). The earlier explanation (split-K's
  autotuning call) was wrong; the warm-up change in `77342d0` was harmless but not the fix. The kernels were
  never shown wrong: the model-path accuracy was 6.2 % in every run, same as the cast path.
- The kernel refused prefill (M > 64) in the static-vs-HF check of the kernel runs; it now falls back
  to the cast for M > 64. Decode timing was unaffected.

## Files

- `results/modal/delta_int8_triton-{l40s-20261117-233833,h100-20261117-233820}/` (full grids, kernel)
- `results/modal/delta_int8_cd_small-{l40s-20261117-233859,h100-20261117-233847}/` (B=1–2, coord-descent)
- `results/modal/int8_probe-*-20261117-182541/` (path comparison), `delta_int8-*-20261116-*` (default cast),
  `delta_int8_cd-*-20261117-183727` and `delta_int8_triton-*-20261117-183727` (cancelled, partial).
- Code: `int8_kernel.py`; `delta_split.py --int8 [--int8-impl triton] [--coord-descent]`.
