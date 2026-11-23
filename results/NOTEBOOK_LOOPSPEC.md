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
