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

## E5 - L40S step profile at B = 1..64, C=128 (`delta_cd_profile-l40s-20261122-202020`; predictions in a4b6e21)
Run: cd step, one container, NVIDIA L40S, selfcheck ALL OK, ~$0.5.
| B | step ms | gap (step − busy) | o_proj µs/call (v1.5 line) | other weights vs microbench | SM clock, power |
|---|---|---|---|---|---|
| 1 | 3.649 | 0.001 | 12.2 (12.2) | – | 2520, 284 W |
| 2 | 4.415 | 0.093 | 12.6 (13.4) | 3.632 vs 3.624 | 2520 |
| 4 | 4.498 | 0.099 | 12.9 (15.8) | 3.641 vs 3.627 | 2520 |
| 8 | 4.723 | 0.094 | 14.6 (20.5) | 3.660 vs 3.628 | 2520 |
| 16 | 5.385 | 0.084 | 29.6 (30.0) | 3.743 vs 3.630 | 2520 |
| 32 | 6.634 | 0.093 | 54.2 (48.9) | 3.810 vs 3.653 (+4 %) | 2520, 314 W |
| 64 | 8.994 | 0.079 | 101.4 (86.7) | 3.907 vs 3.707 (+5 %) | 2460–2520, 350 W |
Against the pre-registered criteria:
- o_proj at B=64 101.4 µs, inside 65–108 → mechanism **supported**; still `triton_red_fused_bmm_3` at B=32/64. It is
  the largest single kernel at B=64 (2.43 ms/step), more than all other weight GEMMs (2.07 + 1.83 ms).
- The straight line is the wrong shape: flat (~12–15 µs, memory-bound) up to B=8, then ~∝ B (≈1.6 µs per sequence,
  ~5 TFLOPS: CUDA cores, no tensor cores). The line overestimates B=2–8 by up to 6 µs/call (0.14 ms/step at B=8) and
  underestimates B=64 by 15 µs/call (0.35 ms/step). Not re-tuned.
- Clock throttling (H3e) **refuted** as a cause: −2.4 % at most at B=64 (at the 350 W power limit).
- Other weight kernels in-step vs microbenchmark (H3c): +4 % / +5 % at B=32/64 → **minor**, within the predicted 10 %.
- Container check: B=16 step +0.3 % ✓. B=1 step 3.649 vs 3.747 ms (−2.6 %, just outside the predicted 2 %).
  **Surprise:** at B=1 this container has ~0 gap (step − busy 0.001 ms) and non-weight 0.276 ms, vs 0.053 ms and
  0.333 ms in the target run. The L40S non-weight target varies ~17 % between containers for the cd step (the
  "tight" 0.315–0.318 ms noise runs were the non-cd step). v1.5's 0.289 ms is −13.2 % vs the target run and +4.7 % vs
  this run: the remaining L40S miss is within the cd step's container-to-container spread (n=2, so only indicative).
- New: L40S gaps between kernels rise from ~0 at B=1 to ~0.09 ms at B ≥ 2 (more kernels, cutlass + splitK). The same
  effect, larger, is the leading candidate for H100's B=2–8 shortfall (−12…−15 %).
Targets NOT changed with this run (would need Sid's approval).
Next candidates (not run): v1.6 o_proj = max(memory floor, per-sequence compute cost) (roofline shape, fit on L40S
only); per-kernel gap at B ≥ 2 from a non-target profile; repeat the cd B=1 point in 2–3 more containers to set the
cd-step noise floor properly.

## Clean-ups and E6 setup (2026-11-22; free tests done, paid run NOT yet launched)
Clean-ups: (1) `--validate` no longer labels v1.3+ "microbenchmarks only; both GPUs are predictions" (they use in-step
kernel times from the target runs). (2) GPU baseline in `--kv-levers` / `compare` uses the MEASURED LM head
(`gpu_extras`: H100 0.101 ms, L40S 0.301 ms at B=1) instead of the clock-transfer model (H100 0.066 ms). Speedup
table changes ≤ 0.1× (e.g. 1x8192 on-chip 14.9 → 15.0×, 16x2048 today 2.3 → 2.4×); write-ups not edited.
(3) `watch_runs.py` shows step-profile runs instead of crashing.
Run design: task `delta_cd_repeat`, per container 1x128 ×3 (normal, caches on), 2x128, 8x128, 1x128f ×2 (fresh builds:
FX-graph + autotune caches off, new cache dir). Step and breakdown from the same graph. 3 containers, L40S.
Free tests: pytest (model tests pass; 19 GPU-only scheduling/KV tests error on a laptop, untouched); calibrate
selftest; check_delta_split_report; laptop dry run of step_profile with GPU calls faked (labels, fresh cache dirs,
cache flags off only for fresh builds, cd on for every build). Torch 2.5.1 source checked: fx_graph_cache (default on),
autotune_local_cache (default on), fx_graph_remote_cache / autotune_remote_cache all exist; cache_dir() re-reads
TORCHINDUCTOR_CACHE_DIR on every call; cd best configs persist only through autotune_local_cache.
**Correction to my 2026-11-22 "two builds" concern:** with default caches, the target run's second 1x128 compile most
likely reused the first build, so build-to-build variation is a weaker explanation than I first said. The run tests it.
Pre-registered decision rule (non-weight time at B=1, C=128; target run 0.333 ms, profile run 0.276 ms):
- builds: spread of the 2 fresh vs 3 cached in a container > 10 % → build variation matters; propose measuring
  target step and breakdown from one build (needs Sid's approval).
- containers: if builds agree but the 3 new containers (+ profile run) spread > ±10 % → the row is too noisy for a
  ±10 % rule; report, do not force a pass.
- if most of the 5 container values (target, profile, 3 new) are within 0.27–0.30 ms → the target run was atypical;
  propose the mean as target (approval). If most are ≥ 0.32 → the miss is real; needs a model fix.
- My expectation, written now: builds agree within ~3 %; containers spread ~10–20 %; mean ~0.29–0.30 ms.
Secondary (for the next fix, non-target data): step − busy gaps at B=2 and 8 per container (profile run: ~0.09 ms).

## E6 - L40S repeat runs: what moves the B=1 non-weight time? (3 containers, `delta_cd_repeat-l40s-20261122-2052{26,41,56}`)
All three: NVIDIA L40S, selfcheck ALL OK, ~$1.2. Non-weight = step − weight kernels, same graph.
| position in container | c1 | c2 | c3 |
|---|---|---|---|
| 1st 1x128 (cached build... first compile) | 0.278 | 0.270 | 0.269 |
| 2nd 1x128 (cached build) | 0.355 | 0.409 | 0.401 |
| 3rd 1x128 (cached build) | 0.356 | 0.346 | 0.400 |
| 1st fresh build | 0.343 | 0.402 | 0.404 |
| 2nd fresh build | 0.403 | 0.403 | 0.403 |
Weight kernels 3.375–3.423 ms in every case; attention/copy busy identical; SM clock 2520 MHz everywhere; fresh builds
really were fresh (compile 17–22 s vs 8.6 s cached; cache flags recorded off).
Finding: **the only thing that changes is the gap between kernels** (step − busy): ≈ 0 for the first graph captured in a
process, then one of two levels, ≈ 0.07 ms or ≈ 0.127 ms, for every later graph, cached or fresh. Same at B ≥ 2: 0.11 or
0.20 ms (2x128 and 8x128). Yesterday's profile run (1x128 first: 0.276, gap 0.001) fits the pattern; the target run timed
its 1x128 grid point after four other graphs (gap 0.053).
Against the pre-registered rule:
- Build variation: **not the cause**. Fresh vs cached builds at the same position agree (c3: 0.401/0.400 vs 0.404/0.403).
  The literal rule ("fresh vs cached spread > 10 %") is tripped only because the first cached point is special; the rule
  did not anticipate a position effect. Reported both ways.
- Container variation: **small** at a given position (first graph 0.269–0.278, ±2 %; plus profile run 0.276).
- The row depends on **measurement order** (first graph vs later: 0.27 vs 0.34–0.41 ms, up to +50 %). With the 5 container
  values taken as each run's first graph, 4/5 are in 0.27–0.30 (target run is the exception, and it was not a first graph).
  Taking all repeats, the spread is ±20 %: too sensitive for a ±10 % rule as currently measured.
- My prediction (builds within ~3 %: yes; containers 10–20 %: no, ±2 %; mean ~0.29–0.30: depends on position) was
  partly wrong: I did not foresee the position effect.
Mechanism: unknown. Not clocks, not kernel time, not builds. Candidates (untested): GPU memory placement of later
graphs (allocator/graph mempool fragmentation), graph-launch work queue state. Quantized levels suggest a discrete state.
What it means:
- v1.5's L40S non-weight (0.289 ms) is +5 % vs first-graph measurements and −20…−29 % vs later ones. The calibration
  verdict on this row is decided by a GPU software state, not by the model. No target changed (needs Sid).
- Proposal for Sid: fix a measurement protocol for this row (e.g. same graph for step and breakdown, and either the first
  graph in a fresh process or the median over N graphs), and report the gap sensitivity alongside it.
- Chip relevance: none for the gaps (GPU scheduling). The genuine non-weight work at B=1 is the busy time of attention +
  small kernels: 0.276 ms = 11.5 µs per layer on L40S; gaps add 0–5 µs per layer depending on state. Use 11.5 µs (busy) as
  the GPU anchor for chip per-layer latency, not the step-minus-weights number.
- For the B ≥ 2 pause fix: non-target L40S gaps are 0.11–0.20 ms per step, state-dependent.

## Decision (Sid, 2026-11-22): report both measurement protocols
The L40S scorecard now also shows the B=1 non-weight time under both protocols, computed from raw files
(`calibrate.PROTOCOL_RUNS`): first graph 0.272 ms (median of 4 containers) → v1.5 +6.2 %; later graphs 0.402 ms (median
of 12) → v1.5 −28.0 %. Informational only: the official row and the pass/fail rule are unchanged until Sid picks which
protocol counts. H100 has no such data yet.

## Decision (Sid, 2026-11-22): the L40S non-weight target is the range first graph .. later graphs
Target 0.272–0.402 ms (error 0 inside, the scorecard's existing convention for ranges). v1.5 0.289 ms → inside →
**L40S now passes the rule** (all 6 summary rows; grid median 1.7 %, max −6.7 %). H100 unchanged (passes; no protocol
data, single target 0.406 ms).
Caveats recorded with the result: (1) the range is wide (±19 % around its midpoint 0.337 ms; vs the midpoint v1.5 would be
−14 %, a fail), so this row is a weaker test than the others; (2) v1.3+ use in-step kernel times from the target runs'
own profiles: both passes show the step is the sum of its measured parts, not an independent prediction of an unseen GPU;
(3) the H100 grid max (−18.7 %) is within noise of the 20 % limit. An independent confirmation needs a third GPU.

# Unified formula (started 2026-11-22, after Sid's brief of the same day)

## Frame
- **Question.** Can ONE formula, used for both GPUs and the chip, predict the best measured decode step on a GPU it
  has never seen, when its GPU inputs come only from the datasheet and a short generic fingerprint run (no step
  timings, no profiles of the target runs)?
- **Formula (proposed, to be frozen on L40S before scoring):**
  `t_step = Σ_s [ L_s + max(bytes_s / BW_s, flops_s / Thr_s) ] + N_stages · g`.
  GPU: a stage is a kernel; BW_s = DRAM read peak, or L2 bandwidth for weight re-reads per sequence (o_proj);
  L_s = per-kernel latency from the fingerprint; g = in-graph gap. Chip: stages = `layer_stages` (11 per layer) x 24 +
  hops; L_s = cycles/clock; BW = SRAM / KV bandwidth; g = 0.
- **Kernel list (Sid, 2026-11-22):** K1 = taken from L40S profiles (counts and kinds only), used for every GPU; primary
  prediction. K2 = counts-only profile on the target GPU (`fingerprint.py --kernel-list`, durations discarded in the
  container), a pre-registered secondary prediction.
- **Metric / rule.** Unchanged `calibrate.py` rule. Design and freeze on L40S; apply once to H100 (semi-blind: H100 data
  was looked at again in F0 below); real test = A100-80GB, predictions committed before its targets are downloaded.
  Inputs and targets from separate containers. GPU name and memory size checked on every run.
- **Chip relevance per term:** bytes/BW and the stage latency chain carry over (form, not values); o_proj's L2 re-reads,
  g, and the kernel list are GPU-only.

## F0 - diagnostics on existing data (free; seen data, NOT model inputs)
Command: `python3 fingerprint.py --existing-checks` (cd target runs' 1x128 profiles, v1.2 microbenchmarks).
- Weight kernels, least squares `t = t0 + bytes/BW` over qkv, o, gate+up: **L40S BW 763 GB/s, t0 1.97 µs**, all 4 kernels
  within −4…+6 % (down +2 %). **H100 BW 2819 GB/s, t0 3.05 µs**; qkv/o/gate+up within 2 %, **down −26 %** (line 11.2 vs
  15.1 µs). Fitted BW ≈ measured read peak (745 / 2982 GB/s). Correction to my report to Sid: I had said L40S t0 ≈ 1.2 µs
  (a 2-point estimate); the 3-point fit gives 2.0 µs.
  → per-kernel latency ≠ tiny-kernel busy time (1.0–1.4 µs on both GPUs): refuted for streaming kernels on both GPUs.
- o_proj (Triton reduction, no tensor cores) per call: L40S 12.2 → 30.0 µs, H100 5.9 → 18.0 µs (B = 1 → 16). H100 is
  1.7x faster per call at B=16; datasheet FP32 (from memory, not checked: L40S 91.6, H100 67 TFLOPS) predicts 1.4x
  SLOWER → CUDA-core FLOPs refuted as the o_proj input; SM count x clock also points the wrong way. Implied weight re-read
  rate at B=16: L40S 4.8 TB/s, H100 9.0 TB/s (plausible L2 bandwidths; not checked against a source).
- Lone 2048x2048 GEMM is flat in B (L40S 14.7 → 14.5 µs, B = 2 → 64; H100 7.9 → 8.5): adding flops/datasheet would add
  1.5 µs (+10 %) at B=64 on L40S → per-stage **max**, not sum, supported.
- Attention 1x128 per layer in-step (3 flash kernels): L40S 7.5 µs, H100 8.6 µs; KV bytes / read peak only 1.4 / 0.35 µs.
  → attention is latency-dominated, more than 3 tiny kernels.
- Kernel counts: 222 on both GPUs at 1x128; at 16x2048 L40S 317 vs H100 413 (gemm 145 vs 241: cuBLAS split-K). The kernel
  list is GPU-dependent at B ≥ 2 (K1 will miss these on H100).

## Hypotheses per GPU input (cheap kill tests; F = fingerprint measurement)
| input | hypotheses | kill test | status after F0 |
|---|---|---|---|
| BW_read | (a) datasheet (b) copy peak (c) read peak (d) F1 slope | weight-kernel fit, α′ | (a) out (L40S −25 %, CALIBRATION.md); (b) out for L40S weights; (c)/(d) fit weights; open: H100 KV at 87 % of read peak |
| L (streaming kernel) | (a) tiny-kernel busy (b) F1 intercept t0 (c) launch + k x DRAM latency (F4) (d) clock-scaled constant | fitted t0 2.0 / 3.05 µs | (a) out; (d) failed in v1; (b) leading; (c) open |
| o_proj per sequence | (a) datasheet FP32 (b) SMs x clock (c) W / BW_L2 (F2) (d) benchmark of the same kernel | H100/L40S ratio; L40S knee at B≈7-8 (E5) | (a), (b) out; (c) leading; (d) fallback (not a fingerprint) |
| compute term | (a) sum (b) per-stage max | GEMM flat in B | (b) supported |
| attention latency | (a) kernels x L_tiny + KV/BW (b) + DRAM round trips x F4 latency (c) attention microbenchmark | in-step 1x128 | (a) out; (b) open; (c) fallback |
| g | (a) 0 (b) F3 first graph / later graphs (c) delta_noise runs | E6 range 0.272-0.402 ms | (c) not allowed (step timings); (b) to test |
| kernel list | K1 / K2 | H100 B ≥ 2 | Sid chose K1 primary, K2 secondary |

## Fingerprint task (written 2026-11-22, not run; needs Sid's OK + billing check)
`fingerprint.py` + Modal task `fingerprint` (~10-15 min): F3 tiny-kernel chain (first graph of the process, then later
graphs), F1 DRAM read sweep 64 KB-512 MB (≥ 384 MB of distinct buffers per point), F2 L2 re-read 1-16 MB, F4 pointer
chase (1 MB, 1 GB; L1 bypassed), F5 bf16 / fp32 GEMM peaks, nvidia-smi name/memory/clocks/power; then K2 kernel list.
Pre-registered fits: BW_read, t0 = least squares of busy time on bytes over 4-256 MB; BW_L2 = slope over 2-16 MB.
Expected (written now, before any run): L40S BW ~740-765 GB/s, t0 ~1-2.5 µs; H100 BW ~2.8-3.0 TB/s, t0 ~2-4 µs (if
t0(H100) < 2 µs, F1's intercept does not explain the weight-kernel latency and (b) is refuted); BW_L2 L40S ~4-6 TB/s,
H100 ~7-11 TB/s (if BW_L2 is > 1.5x off the implied 4.8 / 9.0 TB/s on either GPU, o_proj hypothesis (c) is refuted);
g first graph ~0, later > 0 (if the tiny chain shows no first/later difference, F3 cannot reproduce E6's gap states).
Laptop checks only: `py_compile`, `--existing-checks`. The GPU code has not run yet.

## Decision 4 check: self-speculative decoding for looped LMs (desk only, 2026-11-22)
Read by a search subagent; key numbers re-checked by a second fetch (some table rows from page summaries, not verbatim).
- **LoopSpec** (arXiv 2609.17184, code https://github.com/kaist-flexml-lab/loopspec, SGLang): loop 1 drafts, full depth
  verifies; claimed lossless. Ouro-1.4B greedy 2.64-3.17x (Table 1), batch 1 only, RTX PRO 6000 Blackwell, bf16.
  Baseline = SGLang autoregressive; whether it uses CUDA graphs is not stated.
- **WaveFront** (arXiv 2609.23033): diagonal wavefront of (position, depth) states. Ouro-2.6B only (not 1.4B), RTX A6000,
  B=1; vs a CUDA-graph baseline 16.72 → 38.51 tok/s (2.30x, Table 8, from summary). Speedup falls monotonically with
  batch (App. B.1). bf16 outputs differ slightly from normal decoding.
- **LoopAhead** (arXiv 2610.21594): separately trained 11.1M drafter; Ouro-1.4B 2.82-7.82x vs a PyTorch backend of
  unstated type; batch 1; no code found.
- All three run Ouro at a fixed 4 loops (exit gate unused). Our baseline runs mean depth 3.22 (q=0.5), so their
  speedups overstate the gain over our baseline; none is measured against early exit. No data at B=16.
- **Implication:** at B=1 a lossless ~2-3x GPU speedup is plausible and has public code. Under decision 4 the B=1 GPU
  baseline may have to include LoopSpec (or the LoopCore B=1 speedups must state they are vs non-speculative decoding).
  Decision for Sid. A fair test would run LoopSpec (SGLang) on Ouro-1.4B at B=1 and 16 on L40S/H100, with and without the
  exit gate, plus a check that outputs match. Not costed yet.

## FP1 - fingerprint v1 on L40S and H100 (`fingerprint-{l40s,h100}-20261122-232848`; pre-registered in 68b3939)
GPUs checked: NVIDIA L40S 46068 MiB; NVIDIA H100 80GB HBM3 81559 MiB (not an H200). Both ALL OK. Inputs only.
| measurement | L40S | H100 | pre-registered expectation | verdict |
|---|---|---|---|---|
| F1 BW_read (slope 4-256 MB) | 754 GB/s | 2977 GB/s | 740-765 / 2800-3000 | **as predicted** |
| F1 t0 (intercept) | 4.52 µs | 6.28 µs | 1-2.5 / 2-4 | **invalid probe** (below) |
| F2 BW_L2 (slope 2-16 MB) | 4.71 TB/s | 4.82 TB/s | 4-6 / 7-11; > 1.5x off 4.8 / 9.0 refutes o_proj (c) | L40S as predicted; **H100 1.87x off → (c) refuted as measured** (caveat below) |
| F3 gap per tiny kernel, first graph / later graphs | −0.09 / 0.24-0.27 µs | −0.05 / 0.34-0.60 µs | ~0 / > 0 | **as predicted** (E6's first-graph effect reproduced without the model) |
| F3 tiny-kernel busy | 0.96-1.05 µs | 1.05-1.07 µs | – | – |
| F4 pointer chase 1 MB / 1 GB | 117 / 121 ns | 144 / 149 ns | DRAM clearly slower than L2 | **invalid** (1 GB ≈ 1 MB is not plausible; cause unknown) |
| F5 bf16 GEMM / fp32 GEMM (no TF32) | 235 / 43.8 TFLOPS | 808 / 52.2 TFLOPS | – | measured fp32 ratio H100/L40S 1.19; o_proj per call is 1.66x faster on H100, so measured fp32 does not explain o_proj either |
Why F1's t0 is invalid: `torch.sum` launches 2 kernels per call from 256 KB up (kernels_per_op 2.0-2.25), and below 256 KB
it is one kernel that runs nearly serially (128 KB: 10.8 µs L40S, 14.4 µs H100). The intercept is therefore the cost of a
2-kernel reduction, not the ramp of one streaming kernel. Hypothesis "L = F1 intercept" is neither supported nor refuted.
Caveat on F2: the same `torch.sum` probe gives 4.7-4.8 TB/s on two very different GPUs, which suggests the probe, not L2,
sets that number. The pre-registered rule says refuted; recorded as refuted-as-measured, retest only with a new
pre-registered probe.
K2 kernel lists (names and counts, no durations): totals at 1x128 = 222 on both. B=2-8: L40S 341, **H100 437-438**;
B=16-64 and 16x2048/8192: L40S 317 (B=32: 413), H100 413. So H100 runs ~96-97 more GEMM-category kernels than L40S at
B ≥ 2 (cuBLAS split-K), except at B=32 where L40S switches to the same count. New hypothesis (untested): these extra kernels explain v1.5's
H100 B=2-8 shortfall (−12…−15 %): ~100 x 1-3 µs ≈ 0.1-0.3 ms of ~2 ms. K1 (L40S list) will miss them by construction.
(Category splits in kernel_list.json differ from the old breakdowns at 1x128 (gemm 98 vs 74, other 2 vs 26) with equal
totals: kernel naming per compile; to be resolved from the stored names before the list is used.)
Cost: two runs of ~10-15 min each (exact durations not recorded).
Next (proposed, not run): fingerprint v2 with a single-kernel Triton streaming read for F1/F2 (one program per chunk,
partial sums out) and a pointer-chase size sweep (1 MB-4 GB) to diagnose F4. Needs Sid's OK (~$1.2).

## FP2 - fingerprint v2: pre-registered before running (Sid approved 2026-11-22)
Change: F1 and F2 use `read_sum` (one Triton kernel per read; each program reads 8 KB once, writes one fp32 partial)
instead of `torch.sum`; F4 sweeps 1 MB-4 GB and checks the kernel against the same chain followed on the host;
kernel names stored per probe. F3, F5 and the K2 kernel list unchanged (the list doubles as a repeat).
Predictions and decision rules (written before the run):
- F1: kernels_per_op = 1 at every size (else the probe is invalid again). BW L40S 740-765 GB/s, H100 2.8-3.0 TB/s.
  Hypothesis "per-kernel latency L = F1 intercept t0" is **supported** if t0 is within 1 µs of the weight-kernel fit
  (F0: L40S 1.97, H100 3.05 µs) on both GPUs, **refuted** otherwise.
- F2: o_proj = W / BW_L2 is **revived** if BW_L2 is within 1.5x of the implied rate on both GPUs (L40S 4.8, H100
  9.0 TB/s, i.e. H100 ≥ 6.0 TB/s), and **refuted for good** if H100 stays below 6.0 TB/s with a 1-kernel probe.
  My expectation: H100 6-9 TB/s (i.e. v1's 4.8 was probe-limited), confidence low.
- F4: valid only if `kernel_matches_host` is true and latency at ≥ 256 MB is ≥ 2x the 1 MB latency on both GPUs.
  Expectation: L2 ~150-300 ns, DRAM ~400-700 ns.
- F3: repeat of v1 (first graph ~0, later 0.2-0.6 µs per kernel).

## FP2 - results (`fingerprint-l40s-20261122-233916`, `fingerprint-h100-20261122-233915`)
GPUs: NVIDIA L40S 46068 MiB; NVIDIA H100 80GB HBM3 81559 MiB. Both ALL OK.
| measurement | L40S | H100 | rule (FP2) | verdict |
|---|---|---|---|---|
| F1 kernels per read | 1 + 2/copies | same | = 1 | **literally tripped**: each graph replay adds 2 `FillFunctor<long>` kernels (not the probe). Bias on per-read busy ≤ ~0.4 µs at 8 copies; it can only lower the true t0. Verdicts below hold either way. |
| F1 BW_read | 746 GB/s | **3179 GB/s** | 740-765 / 2800-3000 | L40S as predicted; H100 above range (the 1-kernel probe reads faster than torch.sum's 2982) |
| F1 t0 vs weight-kernel fit (1.97 / 3.05 µs) | 1.51 µs (Δ 0.46) | 1.77 µs (Δ 1.28) | within 1 µs on both | **refuted** (holds on L40S, fails on H100) |
| F2 BW_L2 vs implied (4.8 / 9.0 TB/s) | 5.54 TB/s (1.15x) | 7.32 TB/s (1.23x) | within 1.5x on both, H100 ≥ 6.0 | **o_proj = W / BW_L2 revived**; v1's 4.7-4.8 TB/s was the torch.sum probe's limit |
| F4 pointer chase | 118-122 ns, 1 MB-4 GB | 141-146 ns | kernel = host, and ≥ 256 MB ≥ 2x 1 MB | **invalid again**. Kernel follows the chain correctly (matches host at every size), but each timed call repeats the same 20,000-step path (~2.5 MB of lines), which stays in L2 after the first call. Same flaw in v1. Fix: a path longer than L2 or a new start per call. |
| F3 gap first / later graphs | −0.08 / 0.17-0.18 µs | −0.03 / 0.34-0.59 µs | repeat of v1 | as predicted; L40S later-graph level differs between runs (v1 0.24-0.27) |
| F5 bf16 / fp32 TFLOPS | 246 / 43.6 | 798 / 52.2 | repeat | within 5 % of v1 |
| K2 kernel lists | identical to v1 | identical to v1 | – | repeatable |
Diagnostic on seen data (not a scored prediction; in-step cd kernels at 1x128, F0 numbers):
fingerprint-only weight kernels `t = t0 + bytes / BW_read`: **L40S +1 % per layer** (each kernel −3…+4 %);
**H100 −25 % per layer** (qkv −21, o −25, gate+up −16, down −40 %). The H100 cd reductions read at ~2.8 TB/s and pay
~3 µs each, while a plain streaming kernel reads at 3.2 TB/s and pays 1.8 µs. So on H100 the weight kernels are limited by
the kernels inductor generates, not by the hardware the fingerprint measures. That is a GPU-software term the generic
fingerprint cannot see; at B=1 it is ~0.31 ms (~19 % of H100's 1.67 ms step).
o_proj with `L + max(W / BW_read, B·W / BW_L2)`: L40S knee at B ≈ 7.4 (E5: flat to 8, then linear), B=16 −14 %, B=64
−3 %; H100 B=16 +12 %.
Decision needed (Sid): how the formula treats kernel efficiency below the hardware (H100 weight kernels). Options in the
report of 2026-11-22. Any option chosen now is informed by H100 data, so H100 can no longer be called semi-blind for
that term; the A100 stays the real test.

## Decision (Sid, 2026-11-22): both predictions
- **Prediction (a), primary:** hardware fingerprint only. Weight kernel = t0 (F1 read intercept) + bytes / BW_read (F1).
- **Prediction (b), secondary:** weight kernel = t0_eff + bytes / BW_eff from F6, a compiled matrix-vector kernel with
  coordinate-descent tuning on generic shapes. Designed after seeing H100's gap, so H100 is not semi-blind for this term.
  The A100 is blind for both.
Both are combined with K1 (primary kernel list) and K2 (secondary), and reported separately; none is chosen after the
A100 data.

## FP3 - fingerprint v3: pre-registered before running
Changes: F6 added (lone `torch.compile` F.linear, coordinate_descent_tuning, B=1, shapes 1024x4096, 4096x4096,
8192x4096, 4096x16384 fitted; 1024x16384 held out; distinct weights, ≥ 384 MB per graph). F4: a new random start
per timed call. Everything else as v2.
Predictions and rules:
- F6 is valid only if the stored kernel names are Triton reductions (`triton_red...`), not cuBLAS/cutlass, on both GPUs.
- F6 fit: L40S BW_eff 730-770 GB/s, t0_eff 1.5-2.5 µs; H100 BW_eff 2.6-3.0 TB/s, t0_eff 2.5-4 µs (in-step F0 fit:
  763 / 1.97 and 2819 / 3.05).
- (b) explains the gap if `t0_eff + bytes / BW_eff`, applied to Ouro's 4 weight kernels at 1x128, gives the in-step weight
  time per layer within ±10 % on both GPUs (L40S 142.0 µs, H100 52.5 µs; prediction (a) gave +1 % / −25 %).
  Held-out long reduction 1024x16384: no number predicted; reported to judge H100's down_proj (−26 % vs the line).
- F4 valid if kernel = host and latency at 1 GB ≥ 2x latency at 1 MB on both GPUs. Expected DRAM 400-700 ns.

## FP3 - results (`fingerprint-l40s-20261122-234738`, `fingerprint-h100-20261122-234738`)
GPUs: NVIDIA L40S 46068 MiB; NVIDIA H100 80GB HBM3 81559 MiB. Both ALL OK.
| item | L40S | H100 | rule (FP3) | verdict |
|---|---|---|---|---|
| F6 kernels | `triton_red_fused_mm_0` | same | Triton reductions | **valid** |
| F6 fit BW_eff / t0_eff | 756 GB/s / 0.27 µs | 3082 GB/s / 2.82 µs | 730-770 / 1.5-2.5; 2600-3000 / 2.5-4 | L40S BW yes, t0 below range; H100 t0 yes, BW slightly above range |
| Ouro weight time per layer, 1x128, prediction (a) | 143.7 vs 142.0 µs (**+1.2 %**) | 39.4 vs 52.5 µs (**−25.0 %**) | – | as in FP2 |
| same, prediction (b) | 137.0 (**−3.5 %**) | 44.6 (**−15.0 %**) | ±10 % on both | **refuted**: (b) closes 10 of H100's 25 points, not all |
| held-out long reduction 1024x16384 | 46.1 meas vs 44.6 fit | 13.1 vs 13.7 | reported only | the generic fit holds for long K, so H100's in-step down_proj excess (−26 %) is not a generic long-reduction effect |
| F4 pointer chase | 118 ns (≤ 16 MB) → 249-260 ns (≥ 256 MB), 2.2x | 142 ns → 340-351 ns, 2.5x | kernel = host, ≥ 2x | **valid**. DRAM latency ~260 / ~350 ns, below my 400-700 expectation |
| F3 later-graph gap | – | 0.26-0.27 µs (v1 0.34-0.60, v2 0.34-0.59) | – | varies run to run on both GPUs |
Measurement caveat (all versions): each graph replay includes 2 `FillFunctor<long>` kernels of unknown origin, so per-op
busy times carry + (fill time x 2 / copies). Largest at few copies (F6 4096x16384: 4 copies). This lowers fitted
intercepts somewhat (likely part of L40S's t0_eff 0.27 µs). Not corrected; the fill time was not stored separately.
What remains of the H100 gap after (b): in the step, weight kernels read at ~2.8 TB/s; the same kind of kernel alone on
generic shapes reads at 3.08 TB/s. The in-step kernels are fused with other work (copy, bmm) and get their own tuned
configs; that is specific to Ouro's compiled step and cannot come from a generic fingerprint. L40S shows no such gap.
Status of the formula's GPU inputs: BW_read (F1), BW_L2 (F2), L_tiny and g (F3), DRAM/L2 latency (F4), peaks (F5),
compiled-GEMV efficiency (F6, for (b)) all measured and valid. Remaining run-to-run spread: g (later graphs).
Next: freeze the formula on L40S (free), apply once to H100, then A100.

## U1 - unified formula v2.0 in `loopcore_model.py`: pre-registered before computing any prediction (2026-11-23)
Code: `kernel_class`, `fingerprint_inputs`, `kernel_list`, `gpu_step_unified`, `unified_predictions`, `chip_step_unified`;
`python3 loopcore_model.py --unified <gpu>` writes `results/unified/<gpu>_{a,b}_{K1,K2}.json`, scored with
`calibrate.py --pred` (no --write: CALIBRATION.md is not changed without Sid).
Stage terms per kernel class (header comment in the code). Per-GPU inputs: fingerprint only. Grid and step use the
later-graph gap g (target grids run many graphs per process); the non-weight row is also reported with the first-graph g.
Simplifications fixed now: kernel list at the exact (B, C) if listed, else the listed C=128 config with the largest
B ≤ B; pointwise kernels' activation bytes ignored (≤ ~1 % at B=64); F4 (DRAM latency) not used, since no count of
dependent DRAM trips per kernel could be justified without fitting it.
Order: L40S scored first (design GPU); any change after that is a new pre-registered version; then frozen and applied
once to H100 (semi-blind; H100 fingerprint already seen; variant (b) designed after seeing H100's gap).
Expected on L40S (hand estimate from the fingerprint, written before running), variant (a), K1 (= L40S's own list):
- weight 3.45 ms (inside 3.41-3.98 → 0 %); step 1x128 ~3.70 ms (~−1.5 %); KV per token 0.263 µs (+0.5 %);
  crossover ~13,100 (inside); δ within ±5 %.
- non-weight ~0.25 ms with later-graph g (−5…−10 % vs the range's low end 0.272: borderline), ~0.19 ms with first-graph g.
- grid: median ~3-6 %, max ~−10…−15 % at B=64 (64x128 ~8.0 vs 8.99 ms: in-step tensor-core GEMMs read slower than F1).
- overall: pass likely, non-weight row borderline.
Variant (b): weight ~3.33 ms (~−2 %), step ~−4…−5 %; rest as (a).

## U1 - results on L40S (design GPU; `results/unified/l40s_*.json`, scored with `calibrate.py --pred`, not written)
| row | (a) K1 | (b) K1 | expected (a) |
|---|---|---|---|
| weight | 3.45 ms (inside 3.41-3.98) | 3.29 (−3.7 %) | 3.45, inside |
| KV per token | 0.263 µs (+0.5 %) | same | +0.5 % |
| crossover | 13,095 (inside) | 12,479 (−4.2 %) | ~13,100 |
| step 1x128 | 3.72 ms (**−0.7 %**) | 3.56 (−5.0 %) | ~−1.5 % |
| δ | 3.70 (**−9.0 %**) | 3.63 (−10.7 % ✗) | ±5 % (missed, inside the rule) |
| non-weight 1x128 | 0.271 ms (−0.3 % vs range 0.272-0.402) | same | −5…−10 % |
| grid | median **7.2 %**, max **−12.8 %** (16x128) | 7.5 % / −13.4 % | median 3-6 %, max −10…−15 % at B=64 |
| verdict | **PASS** | FAIL (δ) | pass |
K1 = K2 on L40S (its own list); the K2 files are identical in content.
Grid pattern (a): B=1 within ±1 % at every C. B ≥ 2: −9…−13 % at C ≤ 512, shrinking to −1…−6 % at C = 8192 (the KV term
is right; the fixed part is short by ~0.5 ms). Cause, from E5's in-step data: tensor-core GEMMs at small M (B=2-64) read
weights at ~83 % of the F1 read rate (in-step other-weights at B=16: 3.74 ms vs formula 3.25 ms). A GPU-software
efficiency term again, as with H100's B=1 reductions.
Caveats: the non-weight row passes at the range's lower edge only because of the later-graph gap; with the first-graph
gap the formula gives 0.192 ms (−29 % vs the first-graph value 0.272). Several expectations were off (δ, grid median),
all within the rule. Not changed after seeing this: **v2.0 frozen** as committed in a6514b1.

## U2 - H100, applied once: pre-registered before running
Same code, same frozen v2.0, H100 fingerprint (`fingerprint-h100-20261122-234738`). Semi-blind (H100 fingerprint and
F0/FP2/FP3 diagnostics already seen). All four combinations reported: (a)/(b) x K1/K2.
Expected (from FP2/FP3 and the kernel lists, written before running):
- (a): weight at B=1 ~−25 % → weight ~0.95 vs 1.26-1.77 ms (✗); step 1x128 ~−20 % (✗); FAIL.
- (b): weight ~−15 % (✗); step 1x128 ~−12 % (✗); FAIL.
- KV per token: BW_read 3100 GB/s vs α′-implied 2594 → ~−16 % (✗) in all variants (H100 KV reads at 87 % of the read rate).
- K2 vs K1 at B ≥ 2: + ~96 split-K kernels x (L_tiny + g ≈ 1.3 µs) ≈ +0.13 ms per step, i.e. K2 ~+5-8 % at small C.
- grid with K1: B ≥ 2 ~−15…−30 %; max beyond −20 % likely (✗).

## U2 - results on H100 (frozen v2.0, applied once; `results/unified/h100_*.json`)
| row | (a) K1 | (a) K2 | (b) K1 | (b) K2 | expected |
|---|---|---|---|---|---|
| weight | 0.945 (−25.3 % ✗) | same | 1.07 (−15.3 % ✗) | same | −25 / −15 % |
| KV per token | 0.0634 µs (−19.2 % ✗) | same | same | same | ~−16 % |
| step 1x128 | 1.19 ms (**−28.6 %** ✗) | same | 1.32 (−21.1 % ✗) | same | ~−20 / ~−12 % |
| δ | −37.3 % ✗ | −32.9 % ✗ | −34.7 % ✗ | −30.3 % ✗ | – |
| non-weight 1x128 | 0.248 ms (**−39.1 %** ✗) | −38.9 % | −39.1 % | −38.9 % | not predicted |
| grid median / max | 30.1 % / −43.0 % | 27.6 / −39.2 | 26.7 / −42.1 | 24.8 / −38.3 | max beyond −20 % |
| verdict | FAIL | FAIL | FAIL | FAIL | FAIL |
Worse than predicted: I did not foresee the non-weight miss (−39 %), and the step miss is larger than estimated.
K2 helps by ~2-3 points of median (split-K kernels), as predicted in direction.
Where the time goes (diagnostic, seen data: cd target runs' 1x128 profiles, top-12 kernels, per step):
| class | L40S measured / formula | H100 measured / formula |
|---|---|---|
| weight reductions + o_proj | 3.41 / 3.45 ms | 1.26 / 0.95 ms |
| attention (24 flash kernels) | 115 / 70 µs (4.8 µs per kernel vs L_read 1.51 + KV) | 125 / 45 µs (5.2 vs 1.56 + KV) |
| reductions (norms, combine, ...) | 137 / 112 µs (1.84 vs 1.51 per kernel) | 163 / 115 µs (2.20 vs 1.56) |
| pointwise | 29 / 28 µs | 28 / 28 µs |
So: pointwise kernels match the tiny-kernel probe on both GPUs; everything that reads memory or reduces costs more per
kernel in the real step than the generic probes say, slightly on L40S and a lot on H100. On L40S the non-weight row
passed partly because the formula adds later-graph gaps (62 µs) that offset a 25 % shortfall in busy time.
Attention as a latency chain (not in v2.0): (in-step − L_read − KV/BW) / F4 DRAM latency = ~7 trips (L40S), ~9 (H100).
Conclusion (evidence level: measured on 2 GPUs, the second semi-blind): **the unified formula with generic fingerprint
inputs reproduces L40S but not H100.** On H100 the real kernels of the compiled step are 25-40 % slower than the
hardware the fingerprint measures. That gap is GPU software (kernel generation and tuning), so it does not by itself
say the chip formula is wrong; it does mean the GPU check validates the formula's form only where the software
reaches the hardware (L40S).

## Chip side: same formula vs the current chip model (config D, sized; not applied to defaults or write-ups)
`chip_step_unified` (layer_stages latencies, per-stage max) vs `step_time` with overhead = 24 x bottom-up t_L:
- Ouro as shipped (bf16 KV in HBM): −0.4 … −1.7 % at 1x2048, 16x2048, 16x8192, all three cycle levels.
- One shared fp8 KV cache on-chip: −5 % (1x2048, central), −12 % (16x2048), −14 % (16x8192); up to −16 % at the low
  level. The difference is the per-stage max: the current model adds weight-read time and compute time; when both are
  small and comparable (on-chip KV, B=16) that double-counts.

## U3 - A100 (third GPU, blind): frozen predictions, committed BEFORE any target data is downloaded (2026-11-23)
Inputs: `fingerprint-a100-80gb-20261123-000828` (NVIDIA A100-SXM4-80GB, 81920 MiB, ALL OK; its own container).
F1 BW_read 1757 GB/s, t0 3.67 µs; F2 BW_L2 4.02 TB/s; F3 L_tiny 1.71-1.81 µs, gap first −0.24 / later ~0.00 µs;
F4 L2 ~202 ns, DRAM ~386 ns (valid, 1.9x... ≥ 1 GB / 1 MB = 1.92: **just below the 2x validity rule**; not used by v2.0);
F5 bf16 288 / fp32 19.2 TFLOPS; F6 BW_eff 1753 GB/s, t0_eff 4.27 µs (Triton reductions, valid).
K2 list totals: 1x128 222; B=2-4 438; B=8-16 414; B=32-64 342 (differs from both L40S and H100).
Targets launched (spawned, separate containers, NOT downloaded yet): `delta_bf16_cd-a100-80gb-20261123-000836`,
`delta_compile-a100-80gb-20261123-000904`, `delta_cd_repeat-a100-80gb-20261123-{000933,001001,001029}`.
Frozen v2.0 code (a6514b1 + the grid-points argument, which changes no L40S/H100 output); predictions in
`results/unified/a100_{a,b}_{K1,K2}.json`, all 49 B x C points:
| | (a) K1 primary | (a) K2 | (b) K1 | (b) K2 |
|---|---|---|---|---|
| weight 1x128 | 1.757 ms | 1.757 | 1.817 | 1.817 |
| KV per token | 0.112 µs | same | same | same |
| step 1x128 | 2.180 ms | 2.182 | 2.241 | 2.243 |
| non-weight 1x128, first / later graphs | 0.369 / 0.424 ms | 0.371 / 0.426 | 0.369 / 0.424 | 0.371 / 0.426 |
| δ (fit over 49 points) | 2.336 ms | 2.445 | 2.361 | 2.470 |
| 64x128 / 16x2048 | 6.44 / 6.78 ms | 6.49 / 6.95 | 6.46 / 6.80 | 6.50 / 6.97 |
Scoring plan: `calibrate.py --gpu a100 --pred results/unified/a100_*.json` after adding the A100 to `calibrate.RUNS`
(gpu name NVIDIA A100-SXM4-80GB; target runs above) and its protocol runs to `PROTOCOL_RUNS`. The non-weight target is the
first-graph .. later-graphs range from the three repeat runs, as for L40S. Same pass rule. No re-run of `--unified a100`
after the data is downloaded (the files above are the predictions).
My expectation (written now): A100 lands between L40S and H100; step 1x128 −5…−20 %; overall FAIL more likely than PASS
(~60/40).
