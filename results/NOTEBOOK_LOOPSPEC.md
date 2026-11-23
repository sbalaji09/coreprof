# Lab notebook: is LoopSpec a faster GPU baseline? (decision 4)

Append-only. Started 2026-11-23 (Sid: "go ahead and build LoopSpec, cost is fine").

# Frame
- **Question.** Does LoopSpec (self-speculative decoding for looped LMs; arXiv 2609.17184; code
  github.com/kaist-flexml-lab/loopspec at commit 41aa710ed7c9ce908cffa03597be5d1df19d7b56) decode Ouro-1.4B faster at
  B=1 on our GPUs than (i) normal decoding in the same stack and (ii) our best measured GPU step, and are its outputs
  identical to normal decoding?
- **Why.** Decision 4: the GPU baseline must be the fastest GPU implementation. If LoopSpec wins at B=1, LoopCore's B=1
  speedups shrink by about the same factor.
- **What the code allows (read 2026-11-23, not run).** Batch 1 only (`--loopspec` refuses `--max-running-requests` > 1),
  so it cannot change the B ≥ 2 baseline. Fixed depth 4: the exit gate is loaded but never evaluated
  (`modeling/ouro.py`, OuroRecurrentExecutor). Stack: CUDA 13.3, torch 2.11.0, sglang 0.5.14, transformers 5.8.1,
  Triton attention backend. Speed metric logged by the server per request (`generation_metrics`: decode tokens after
  the first, host-side seconds), identical for baseline and LoopSpec.
- **Metric.** Decode ms per output token at B=1 (sum of decode seconds / sum of decode tokens over the prompts, and the
  per-request median). Guardrail: greedy output text identical to SGLang normal decoding (fraction of prompts).
- **Workload.** 40 GSM8K test questions (first 40), 3-shot (first 3 train examples, "Question: / Answer:" format),
  greedy, max 256 new tokens, stop at "Question:". 2 warm-up prompts discarded per mode. Ouro-1.4B bf16 at our pinned
  revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. Context during decode ≈ prompt (~600-800 tokens) + generated.
- **Comparisons.**
  - (i) LoopSpec "1" and "1:2" vs SGLang normal decoding (4 loops), same container: same outputs expected, pure speed.
  - (ii) vs our best bf16 step x mean depth 3.22 (early exit, q=0.5) + LM head, from existing grids at the measured
    context: different outputs (early exit), the decision-4 comparison. Also vs H100 int8 (LoopCore's baseline).
  - SGLang's own normal decoding is itself a GPU implementation: if it beats our compiled step, it changes the baseline
    too, independently of LoopSpec.
- **Budget.** L40S + H100, one container each (~20-40 min each incl. image build and CUDA-graph capture), plus a short
  L40S smoke run. Estimated $3-8 including debugging.

# Pre-registered expectations (written before any run)
- LoopSpec "1" vs SGLang normal decoding at B=1: **2.0-3.0x** faster on both GPUs (paper: 2.64-3.17x on an RTX PRO 6000).
  "1:2" within ±10 % of "1".
- Outputs: identical text for ≥ 90 % of prompts (bf16 batch-shape effects can flip a greedy token; the paper claims
  lossless).
- SGLang normal decoding vs our compiled step x 4 loops: within ±25 % (both CUDA-graphed; ours uses fused
  coordinate-descent kernels, theirs SGLang kernels).
- vs our early-exit baseline (3.22 loops): LoopSpec "1" faster by **1.5-2.5x** (the early-exit baseline already saves
  ~20 % of the loops).
- Decision rule: if LoopSpec is faster than our best B=1 implementation by more than the run-to-run noise (we use ±10 %),
  it becomes the B=1 GPU baseline for LoopCore claims (bf16), and the int8 baseline question is reopened (LoopSpec is
  bf16 only). Otherwise the paper states that LoopSpec was measured and did not beat the baseline on our setup.

# Setup log (2026-11-23)
- Image: `uv sync --locked` fails (the anonymized repo's pyproject no longer matches uv.lock); switched to `--frozen`,
  which installs exactly the lockfile's versions. Verified in the container: torch 2.11.0+cu130, transformers 5.8.1.
- Bench fix: SGLang sends its own warm-up request at start-up, which logs one extra metrics line; metrics lines present
  when "The server is fired up" appears are now skipped.
- Smoke (`loopspec_smoke-l40s-20261123-112505`, L40S, 3 prompts x 64 tokens, mean prompt 436 tokens; NOT the measurement):
  normal decoding 16.8 ms/token; LoopSpec "1" 7.9 (2.12x), "1:2" 6.8 (2.48x). Text identical for 2/3 prompts: prompt 1
  diverges at one token ("2/2" vs "2 / 2"), identically in "1" and "1:2", then drifts. Consistent with a bf16 near-tie
  flipped by different batch shapes; not proven.
- Prompt length is ~430 tokens (not ~600-800 as guessed in the frame); our comparison uses the measured mean context.

# H100 failures and the TinyGEMM workaround
- `loopspec-h100-20261123-113017`: the normal-decoding server hung at decode CUDA-graph capture (bs=1) for 30 min.
- `loopspec-h100-20261123-120219` (retry): the same capture fails with "CUDA error: unspecified launch failure".
- Cause (from the code, not proven by a kernel trace): `install_linear_backend` switches SGLang's linear layers to
  FlashInfer TinyGEMM on GPUs it supports (it passes a 2x64 probe), i.e. on H100 but presumably not on L40S (sm89).
- Workaround (pre-registered as a change of kernels only, applied to all three modes alike): `--no-tinygemm` keeps
  SGLang's own linear kernels, as on L40S; a sitecustomize wrapper logs the chosen backend in every server process
  (H100 run: `active: torch` in all 3 modes). Consequence: on H100, LoopSpec as the authors ship it does not run at
  all on our stack; the numbers below are LoopSpec with SGLang's kernels. TinyGEMM, if it worked, could make both normal
  decoding and LoopSpec faster on H100 (not measured).

# Results (40 GSM8K prompts, greedy, ≤ 256 tokens, mean prompt 444 tokens, mean completion ~118; B=1)
Runs: `loopspec-l40s-20261123-112959` (NVIDIA L40S; env reports 49140 MiB, vs 46068 MiB on earlier L40S runs: a different
board/driver variant, noted), `loopspec_notinygemm-h100-20261123-120532` (NVIDIA H100 80GB HBM3).
Speed = server-logged decode ms per output token (sum over prompts); median per prompt within 1-2 % of it.
| | L40S | H100 |
|---|---|---|
| SGLang normal decoding (4 loops) | 14.91 ms | 7.28 ms |
| LoopSpec "1" | 6.17 ms (2.42x) | 3.16 ms (2.31x) |
| LoopSpec "1:2" | **5.69 ms (2.62x)** | **2.89 ms (2.52x)** |
| ours bf16, 4 loops (C≈512, GPU time only) | 15.39 ms | 7.11 ms |
| ours bf16, early exit 3.22 loops | 12.45 ms | 5.74 ms |
| ours int8, early exit (LoopCore's baseline) | 7.50 ms | 4.54 ms |
| best LoopSpec vs ours bf16 early exit | 2.19x | 1.98x |
| best LoopSpec vs ours int8 early exit | **1.32x** | **1.57x** |
Ours = depth x measured best step at (1, 512) + measured LM head (`gpu_extras`); it counts GPU time only, so it slightly
favours our baseline (the LoopSpec numbers include server overheads).
Output agreement with normal decoding (same container):
| | L40S "1" / "1:2" | H100 "1" / "1:2" |
|---|---|---|
| identical text | 65 % / 65 % | 67.5 % / 57.5 % |
| same final answer | 37 / 35 of 40 | 37 / 37 of 40 |
| correct (GSM8K, 40 questions) | 29 / 30 (normal: 29) | 29 / 29 (normal: 29) |
Context for the text differences: normal decoding itself gives identical text on L40S and H100 for only 29/40 prompts.
Greedy bf16 text at this length is fragile across kernels and batch shapes; LoopSpec's 57-68 % is of the same order.
Differences seen are formatting (dropped `<<a+b=c>>` calculator notes, spacing). Not proven to be rounding only.

# Against the pre-registration
- LoopSpec 2.0-3.0x over normal decoding: **as predicted** (2.31-2.62x).
- Identical text ≥ 90 %: **refuted** (57.5-67.5 %); accuracy on 40 questions unchanged (29-30 vs 29). n=40 cannot rule out
  a small accuracy change.
- SGLang normal decoding within ±25 % of our compiled step x 4: **as predicted** (L40S −3 %, H100 +2 %).
- LoopSpec 1.5-2.5x over our early-exit bf16 baseline: **as predicted** (2.19x, 1.98x).
- Not predicted: LoopSpec (bf16) also beats our int8 early-exit baseline, by 1.32x (L40S) and 1.57x (H100).
- Decision rule (written before running): LoopSpec beats our best B=1 implementation by more than ±10 % on both GPUs →
  **it becomes the B=1 GPU baseline for LoopCore claims**, and the int8 question reopens (LoopSpec has no int8 path;
  int8 + LoopSpec would likely be faster still, not measured).
Implication for LoopCore (estimate, not computed in the model yet): B=1 speedups vs H100 shrink by ~1.6x (vs the int8
baseline); e.g. the as-trained 1x2048 band 5.8-14.6x would become roughly 3.7-9.3x, if LoopSpec's gain at C=2048 is
similar to its gain at C≈500 (unmeasured). B ≥ 2 unchanged: LoopSpec runs at B=1 only.
Caveats: one container per GPU (no repeat); C≈500 only; H100 without TinyGEMM; GSM8K only (the paper reports similar
gains on code and BBH).
