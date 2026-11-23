# Lab notebook: calibration of loopcore_model.py against measured GPUs

Append-only. Corrections are new entries pointing back. Started 2026-11-21.

# Frame

- **Question.** Can the analytical model reproduce the best measured GPU decode step (L40S, H100) under the
  CALIBRATION.md rule? If not, which part of each miss would also affect the LoopCore (chip) prediction?
- **Metric.** `calibrate.py` scorecard: 6 summary quantities within ±10 %; grid median |err| ≤ 10 %, max ≤ 20 %.
  Rule, results and write-up numbers are not changed without Sid's approval.
- **Guardrails.** No tuning on H100 data (already used several times). GPU constants from raw run files only.
  Every model change is pre-registered (commit message) before it is scored, and recorded whatever it scores.
- **Success.** (a) Both GPUs pass with a pre-registered model fitted/designed on L40S only and applied to H100
  once; or (b) each miss is attributed, with evidence, to GPU software vs. something the chip shares.
- **Budget.** ~10 experiments. GPU only on Modal, after Sid approves task/duration/cost.

# Baseline (reproduced 2026-11-21, git 151f833, recorded predictions = v1.2 kernel-sum)

`python3 calibrate.py --gpu l40s --pred calibration_predictions.json` (and h100): identical to CALIBRATION.md.

| | L40S | H100 |
|---|---:|---:|
| B=1 C=128 step | 3.75 meas / 4.32 pred, **+15.3 %** | 1.67 / 2.01, **+20.2 %** |
| non-weight at B=1 | 0.627 / 0.541, **−13.8 %** | 0.548 / 0.538, −1.9 % |
| grid | median 3.1 %, max **−23.0 %** (64x128) | median 7.6 %, max **−33.1 %** (64x128) |

# Noise floor (from existing repeats)

- Within a container: step IQR 0.002–0.010 ms at 1x128 (≤ 0.5 %). Negligible.
- Between containers, fused (non-cd) step 1x128: L40S 4.085, 4.086, 4.084 ms (2026-11-20) and 4.177 (2026-11-16):
  ≤ 2.3 %. H100 2.016, 1.913 (2026-11-20) and 2.143 (2026-11-16): range 11 %, ≈ ±6 %.
- **Correction to CALIBRATION.md's H100 noise statement:** `delta_noise-h100-20261120-011018` ran on an
  **NVIDIA H200** (noise.json `gpu`). The "0.41, 0.50, 0.55 ms" H100 non-weight spread includes it (0.50 is the H200).
  Genuine H100s: 0.554 and 0.408. All calibration-target and kernel-bench runs are genuine H100s (checked every run).
- No repeat exists of the coordinate-descent (cd) grid, which is the target at 45/46 (L40S) and 41/48 (H100) points.
  Assume the fused step's between-container spread applies (L40S ~2 %, H100 ~6 %).
- So: B=1 step misses (15 %, 20 %) and the B=64 misses (23 %, 33 %) are far outside noise.

# Observations from existing data (no new runs)

O1. **Kernel classification bug inflates the non-weight target.** In the cd step, o_proj is lowered as
`triton_red_fused_bmm_3` (24 calls/step). `delta_split.KERNEL_CATS` matches `_mm_` but not `_bmm_`, so it is counted as
"other", not weights. Evidence: if the four fused reductions read qkv/o/gate-up/down, they run at 694/685/746/730 GB/s
on L40S (read peak 745) and 2064/1420/2388/1528 GB/s on H100: consistent. Corrected cd split at 1x128:
weights L40S 3.120 + 0.296 = 3.416 ms, H100 1.123 + 0.144 = 1.267 ms; non-weight L40S 0.331 ms (matches the fused-step
noise runs' 0.315–0.318), H100 0.404 ms. Affects targets `overhead_ms` and the low end of `weight_ms` on both GPUs.
NOT applied: needs Sid's approval (changes a result).

O2. **B=1 step miss = weight-kernel choice.** Model weights (lone compiled F.linear = cuBLAS gemv) 3.97 ms slot
(L40S) vs in-step cd reductions 3.42 ms; H100 1.62 vs 1.27. Model small+attention 0.35 ms (L40S) vs measured
non-weight 0.33. The weight difference accounts for ~all of the +0.57 ms (L40S) and +0.34 ms (H100) miss.

O3. **L40S cd reductions reach ~the read peak** (bytes / 745 GB/s = 3.31 ms vs 3.42 measured, −3 %). H100 cd reductions
do not: bytes / 2981 GB/s = 0.83 ms vs 1.27. Per-kernel excess over bytes/peak: H100 ≈ 3–4 µs for qkv, o, gate-up,
≈ 7 µs for down; L40S 0–2.4 µs.

O4. **Large-B miss is a per-sequence cost, not attention.** L40S residual (meas − pred) is flat in C at each B:
B=32 +0.77…+0.87 ms, B=64 +2.07…+2.26 ms over C = 128…2048. H100 similar but noisier (implementation switches).

O5. **o_proj stays a Triton reduction (no tensor cores) at B ≥ 2 in the cd step**, while the other projections switch
to cutlass tensor-core GEMMs. Its time grows with B: L40S 0.294 → 0.717 ms (B 1 → 16), H100 0.142 → 0.434 ms.
The model prices o_proj as a lone compiled linear (flat in B: 0.38 ms L40S, 0.22 H100 at B=16). Excess at B=16:
L40S 0.34 ms vs residual 0.27; H100 0.21 vs 0.30. Linear extrapolation to B=64: L40S ~2.1 ms vs residual 2.07.
A lone compiled F.linear never takes this path (it is lowered as mm, not bmm), which is why v1.2 could not reproduce it.

O6. The chip prediction does not use any GPU-fitted constant except `per_token_extras` for the GPU baseline
(H100 LM head comes from the clock-transfer model, though 0.094 ms is measured). GPU calibration therefore tests the
modelling method (bytes/bandwidth + compute + per-step overhead, serial sum), not chip parameters.

# Hypotheses and cheap tests (written before any new experiment)

## Miss 1: B=1 step (+15 % L40S, +20 % H100)
| # | hypothesis | cheap test | status |
|---|---|---|---|
| H1a | weight term uses the wrong kernels (cuBLAS gemv vs in-step cd reductions) | sum of in-step kernel times vs model terms (existing profiles) | **supported by O2** |
| H1b | inter-kernel gaps mis-modelled | step − busy at 1x128: L40S 0.053 ms, H100 0.089 ms; model's gap budget similar | weak: < 0.1 ms of 0.57/0.34 |
| H1c | target noise | between-container spread ≤ 2 % / ±6 % | **refuted** |
| H1d | (different decomposition) per kernel = bytes/peak + c_kernel, c from memory latency × parallelism (Little's law), not microbenchmarks | fit c on L40S kernels, predict H100 c from latency/BW ratio: needs a published/measured DRAM latency per GPU | open; cheap (paper figures) |

## Miss 2: L40S non-weight at B=1 (−13.8 %)
| H2a | target mis-measured: o_proj counted as non-weight | O1 | **supported**; corrected target 0.331 ms |
| H2b | model's split charges gemv slot gaps (0.19 ms) to non-weight | compare model small+attn (0.35) vs measured non-weight (0.33): +5 % | supported; the summary metric is not apples-to-apples on either side |
| H2c | container noise | L40S tight (≤ 0.004 ms) | refuted |

## Miss 3: large-B grid (−23 % L40S, −33 % H100 at 64x128)
| H3a | o_proj lowered as non-tensor-core Triton reduction, cost ∝ B | O5 extrapolation; decisive test = profile cd step at B = 32, 64 | supported at B ≤ 16; untested at 32/64 |
| H3b | attention slower in-step than microbenchmark at large B | residual flat in C (O4) | **refuted (L40S)**; H100 not decisive |
| H3c | tensor-core GEMM tiles in-step differ from lone compiled linear at B ≥ 32 | needs profile at B = 32/64 (same run as H3a) | open |
| H3d | small elementwise kernels scale with B | 126 × ~3 µs at B=64 ≈ 0.4 ms max | cannot explain 2 ms alone |
| H3e | (new mechanism) power/clock throttling during sustained large-B steps, absent in short microbenchmarks | `nvidia-smi` clocks during the profile run (same run) | open, low prior (kernels are memory-bound) |

# Chip relevance (per miss, current evidence)
- Miss 1: GPU software (kernel choice). Chip-relevant residue: H100's ~3–7 µs per-kernel cost above bytes/peak is a
  latency-bandwidth (ramp/drain) effect; the chip analogue is per-layer pipeline fill, i.e. the per-layer-latency term.
- Miss 2: a measurement artifact. The corrected GPU non-weight time is 13.8 µs/layer (L40S) and 16.8 µs/layer (H100):
  an empirical upper anchor for chip per-layer latency, dominated by launch gaps and kernel ramp, not by FLOPs.
- Miss 3: GPU software (inductor lowers o_proj as bmm → reduction). It slows the GPU *baseline* at B ≥ 16
  (≈ 0.45 ms of 5.37 ms at L40S 16x128): LoopCore speedups there are partly credited for a GPU software artifact.
  Separately, at B ≥ 4 the chip's compute term (assumed 100 % of peak FLOPs) exceeds its weight-read term; no GPU data so
  far measures small-M compute efficiency (all GPU GEMMs here are memory-bound), so that assumption is untested.

# Approach queue (ranked by information per cost)
| # | approach | cost | prior | status |
|---|---|---|---|---|
| A1 | Ask Sid: fix `_bmm_` classification and recompute targets (O1) | free | high | pending approval |
| A2 | Pre-registered v1.3: weight term = per-shape in-step cd kernel times, o_proj with its B-dependence; L40S design, H100 applied once | free for B ≤ 16; profile run for B > 16 | med-high | not started |
| A3 | Profile cd step at B ∈ {2, 4, 8, 32, 64} × C=128 (+ clocks), L40S only | ~15 min L40S ≈ $0.5 | decisive for H3a/c/e | needs approval |
| A4 | Physics floor + software excess decomposition (bytes/peak, FLOPs/peak) reported per term on both GPUs | free | high (for the paper) | not started |
| A5 | Little's-law per-kernel cost (H1d), transferred by published latency, not fitted on H100 | free | low-med | not started |
| A6 | Confirm on a third GPU (A100) | ~$1–2 | — | ask Sid first |

# Experiments

## E1 - fix the o_proj classification, recompute targets (approach A1; approved by Sid 2026-11-22)
Change: `delta_split.KERNEL_CATS` counts `fused_bmm_` as gemm; `calibrate.weight_kernel_ms` re-sorts older
breakdowns from `top_kernels_ms`. Raw run files untouched. Pre-registered in f7e3b2a before running.
Prediction: weight low end L40S ~3.41 / H100 ~1.27 ms; non-weight ~0.33 / ~0.40 ms; v1.2 non-weight misses
~+60 % / ~+33 %; step, δ, KV, grid unchanged.
Result (`calibrate.py --write`, both GPUs): weight 3.41–3.98 ms (L40S), 1.26–1.77 ms (H100), model inside both.
Crossover 13,023–15,173 / 16,106–22,497 tokens, model inside both. Non-weight 0.333 ms (L40S, model +62.4 % ✗),
0.406 ms (H100, model +32.4 % ✗; was a pass at −1.9 %). Step, δ, KV, grid unchanged. Overall: FAIL on both, as before.
Guardrails: `calibrate.py --selftest`, `check_delta_split_report.py`, `pytest tests/test_loopcore_model.py` pass.
Verdict: supported (all predicted values matched).
Learned: the model's non-weight row was "passing" by accident. Two errors offset: the target counted o_proj as
non-weight, and the model counts ~0.19 ms of launch gaps around its matrix-vector kernels as non-weight.
Note: the prose paragraphs under the blocks in CALIBRATION.md ("Model status 2026-11-20", "Target noise") still
quote the old −13.8 % / −1.9 % and the H200-contaminated H100 noise. Not edited; asked Sid.

## E2 - v1.3: B=1 weight term from the in-step coordinate-descent kernels (approach A2, fix 1; pre-registered 5ff8a69)
Change (one thing): weight term at B=1 = in-step busy time of the cd weight kernels (1x128 profile) + one
microbenchmark launch gap per weight kernel. Everything else as v1.2.
Prediction: step L40S ~3.87 (+~3 %), H100 ~1.74 (+~4 %); non-weight ~0.46 (+~37 %) / ~0.48 (+~17 %); grid max unchanged.
Result: step **3.88 ms (+3.5 %, pass)** L40S, **1.74 ms (+4.4 %, pass)** H100. Non-weight 0.463 (+39.1 % ✗) / 0.48
(+18.1 % ✗). δ +6.4 % / +3.8 %. Grid median 3.1 → 2.7 % (L40S), 7.6 → 6.2 % (H100); max unchanged −23.0 % / −33.1 %
(B=64). B=1 row L40S +2.3 to +5.5 %; H100 +4.4, +2.0, +1.3, +1.5, −11.9, −5.8, −6.1 % (C=128 … 8192).
Overall: FAIL on both GPUs (non-weight row and grid max), as predicted.
Guardrails: pytest 23 passed (+1 xfail), new test checks only B=1 weights changed.
Verdict: supported. Every predicted number matched within ~0.01 ms.
Caveats: (1) The B=1 step is a decomposition check, not an independent prediction (kernel times and target are
from the same run). (2) The "weight streaming time" row now equals the measured low end by construction (3.41 /
1.26 ms): that row no longer tests anything for v1.3. (3) H100 B=1 at C ≥ 2048 swings from +1.5 % to −11.9 %: the
measured B=1 row jumps there (also seen in v1.2: +16.3 → +0.3 %); unexplained, within the ±20 % max rule.
Learned: once the right weight kernels are used, the remaining B=1 error is ~0.13 ms (L40S) / ~0.07 ms (H100), all of
it in the non-weight part, consistent with the ~0.1 ms of launch gaps charged per weight kernel (fix 2).
Chip relevance: none directly. This was GPU kernel choice. It does show that, with good kernels, L40S weight time is
within 3 % of bytes / measured read peak (the chip model's form); H100 is 52 % above it (per-kernel ramp, O3).

## E3 - v1.4: busy time + one in-graph gap per kernel (fix 2; pre-registered ca3ca0b)
Change (one thing): kernels cost busy time + g per kernel; g from the non-target delta_noise runs
(L40S −0.03 µs, H100 0.27 µs; H200 run excluded) instead of microbenchmark slot gaps (0.2–1.2 µs).
Prediction: step L40S ~3.70 (−1 %), H100 ~1.66 (−1 %); non-weight H100 ~0.40 (pass), L40S ~0.29 (~−13 %, fail);
B ≥ 2 lower by ~0.2 / ~0.15 ms; grid max ~−26 % / ~−36 %. Median not predicted.
Result: step **3.70 (−1.2 %)** / **1.66 (−0.6 %)**. Non-weight L40S **0.289 (−13.2 % ✗)**, H100 **0.396 (−2.6 %, pass)**.
δ +0.4 % / −4.8 %. Grid L40S median 3.0 %, max −26.0 %; H100 median **10.8 % (✗, was 6.2 %)**, max −34.9 %.
B=2–8 rows: L40S −0.7…−3.7 %; H100 −16.2…+11.6 % (mostly −10…−16 % at C ≤ 1024).
Verdict: supported on every predicted number; one unpredicted regression (H100 median now fails, 10.8 %).
Why: the microbenchmark gaps (~0.15 ms per step on H100) were hiding a real H100 deficit at B=2–8. Same pattern as E1:
two errors cancelling. Candidates for that deficit (not tested): o_proj's Triton reduction growing with B (O5:
~0.02 ms per sequence on H100) and the cublasLt splitKreduce kernels H100 adds at B ≥ 2 (0.17 ms at B=16), which the
model does not count. Both belong to fix 3.
L40S non-weight: the cd step has ~0.24 µs/kernel of gaps in its own profile vs ~0 in the non-cd noise runs. Not
tuned (the only source for the cd value is the target run). Open question: why do Triton reduction kernels leave
gaps that cuBLAS kernels do not?
Not changed after seeing the result. v1.4 is the recorded model; fix 3 builds on it.
Chip relevance: the in-graph gap per kernel is a GPU software cost (0–0.5 µs). It has no chip counterpart in a fused
dataflow design; it is excluded from the chip's per-layer latency anchor.

## E4 - v1.5: o_proj priced at its in-step, batch-dependent time (fix 3; pre-registered a4b6e21)
Change (one thing): at B ≥ 2, o_proj per call = line through its in-step time at B=1 and B=16 (cd run profiles).
Dropped before coding: "uncounted splitK kernels on H100" (in-step non-o_proj weights at B=16 within 4 % of the model).
Prediction: L40S grid max ~−7 %, grid passes, overall still fails on non-weight (−13.2 %). H100 max ~−18 % at B=2
(borderline pass), median ~10 % (borderline fail). B=1 unchanged.
Result: L40S grid median 1.7 %, max **−6.7 %** (64x128) → grid passes; overall **FAIL** (non-weight −13.2 %, only miss).
H100 grid median **8.2 %**, max **−18.7 %** (2x4096) → passes; every summary row within ±10 % → **H100 PASSES** the rule.
Verdict: supported; L40S as predicted; H100 median better than predicted (8.2 vs ~10 %).
Caveats (strong):
- The model now takes in-step kernel times from the target runs' own profiles (weights at B=1, o_proj at B=1 and 16).
  It shows the step is the sum of its parts, not that the model predicts a GPU it has not seen.
- The H100 pass is narrow: max −18.7 % vs the 20 % limit, with between-container noise about ±6 %. A repeat run could
  flip it.
- The H100 rows at B=2–8 are still −12…−15 %: the gaps between kernels grow at B ≥ 2 (step − busy 0.37 ms at 16x2048
  vs 0.09 at B=1). Not modelled.
- L40S B ≥ 32 was not a blind test (O5 had already shown the match).
Chip relevance: o_proj's slowdown is GPU software (compiler lowering). It inflates the GPU baseline at B ≥ 16 (L40S
16x128: 0.36 ms of 5.37 ms over a tensor-core o_proj). LoopCore speedups at B ≥ 16 should be reported both against
the measured baseline and against the baseline with that excess removed.
