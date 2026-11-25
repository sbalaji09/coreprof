# Novelty check before submission (2026-11-24)

Read-only check of the paper's four intended contributions against 2023-2026 work. Four parallel search agents
(arXiv listing/abs/html/pdf, web search, GitHub API; Semantic Scholar and Google Scholar were rate-limited or refused
on most queries, so coverage is arXiv-heavy and conference proceedings pages were not browsed directly). Every quote
below is from a page or PDF that was opened and saved as text; rows marked **[re-checked]** I re-read myself in the
saved text. Anything seen only in a search snippet or listing is marked "not verified". Ratings: **same result** /
**overlapping** / **adjacent**. Saved texts: scratchpad `lit/` and `pdfs/` of this session (not in the repo).

## Summary

| contribution | verdict | biggest prior work |
|---|---|---|
| 1. GPU characterisation of Ouro-1.4B incl. LoopSpec | **partly novel** | 2611.05150 (Ouro serving profile, GH200 / RTX 5090), PackLoop 2611.30205 (Ouro on H100 NVL, RTX 3090, A100, A6000; vs LoopSpec and CDB) |
| 2. Analytical model needs a measured efficiency factor; spec-only transfer fails a blind test | **partly novel** | GenZ 2406.01698, KernelSight-LM 2606.28565, Chen 2605.30571 |
| 3. SRAM dataflow chip requirements for looped LMs | **partly novel** (the looped-LM chip study itself appears new) | SHIP (MLSys'26), LIMINAL 2507.14397, Chiplet Cloud 2307.02666, MixerLoop 2608.18230 |
| 4a. Ouro can't share KV; Table 14 doesn't reproduce (−9.8) | **already done** | Fold the Loop 2611.25369 (−9.86, same n=1319), CDB 2608.09444, LoopKV 2611.04137, MELT 2605.07721 |
| 4b. Trained first-loop sharing + 64-token window matches per-loop | **already done** | Monea et al. 2610.02383 (same design, ω=64, 150M-1B), PLT 2510.24824 (w=64, 2025), MoR 2507.10524 |
| (CDB, DESIGN_CDB.md; not a listed contribution) | **someone else's published method** | Schwethelm, Rückert, Kaissis 2608.09444 (v1 10 Aug 2026); vllm-rlt (public 10 Sep 2026) |

---

## Specific question 1: arXiv 2608.09444 "Continuous Depth Batching" — is it Sid's? **No.**
**[re-checked]** "Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching", Kristian
Schwethelm (TUM), Daniel Rückert (TUM; Imperial College London; MCML), Georgios Kaissis (Hasso Plattner Institute,
Potsdam). v1 10 Aug 2026, v2 25 Sep 2026. https://arxiv.org/abs/2608.09444 . No author is Siddharth Balaji or Vaibhav.

Timeline: their v1 (10 Aug 2026) and their public engine github.com/InferMatrixLab/vllm-rlt (GitHub `created_at`
2026-09-10, checked with `gh api`) both predate this repo's DESIGN_CDB.md (commit 67a753a, 2026-10-23). The name
itself comes from Bae et al. (RRT, ICLR 2025), which both they and DESIGN_CDB.md cite.

Overlap with DESIGN_CDB.md — **same result on the core mechanism**:
- Scheduling: "CDB removes tokens from the loop as they exit and refills the freed slots with new tokens to keep batch
  size large" (§1, v2); refill mode lets "successor tokens from unfinished sequences ... fill the freed recurrent slots
  before the next loop batch" (§A.4). Same as our "same sequence's next token enters at loop 1".
- Skipped-loop KV: **[re-checked]** "A last-exited cache copies a token's final KV state into skipped slots upon exit.
  This generally preserves accuracy and avoids computing the missing states" (§4.2, v2). Same as our copy-forward (D1, A).
- Claim of priority: **[re-checked]** "we include all of these costs and provide the first end-to-end implementation
  and evaluation of CDB" (§3).
- Same model (Ouro-1.4B, plus Huginn-3.5B) and an H100 ("All experiments run on a single 80 GB H100 GPU", §6.1).

Differences that remain ours (small): same-step Ouro gate with CDF ≥ q (theirs: a distilled one-step-lookahead gate,
r_min = 2); a token-for-token q = 1 equivalence test against HF generate (they assert CDB "does not change the model's
predictions" but no such test was found); L40S and A100 measurements. PackLoop 2611.30205 also benchmarks CDB on
Ouro and reports **[re-checked]** "PackLoop is 1.89–2.60× faster than CDB in vllm-rlt on the RTX 3090 and 2.73–3.59×
on the H100". **Framing:** CDB is our reimplementation of Schwethelm et al.'s method (cite it; do not claim it).

## Specific question 2: how much of contribution 4 do Monea et al. and MoR cover?
- **Monea et al., "The Surprising Effectiveness of Shared Memory in Looped Transformers"**, arXiv 2610.02383, v1
  1 Oct 2026 (Monea, Ramji, El-Kurdi, Lastras, Artzi, Godey, Fernandez Astudillo; Cornell / IBM). **Same result as 4b,
  at larger scale and with a stronger outcome.** Design: first recursion writes the cache; later recursions read it
  plus their own window "using a single softmax" — the same as sharefirst_w64. Default window: **[re-checked]** "we
  adopt ω=64 as the default for our memory sharing models" (§6). Scale: ~150M, 450M, 1B params; 20-50B FineWeb-Edu
  tokens (ours 81M / 2B). Result: "lpt lowers validation perplexity by 0.38–1.09 ... over lt, with 2.0–4.7× less KV
  memory" (§5) — they *beat* per-loop caches; we tie. **Disagreement worth reporting:** **[re-checked]** "lpt outperforms
  lt at every window size ... This includes ω=0" (§6), whereas our plain first-loop sharing is +0.0108 nats (worse).
  No MQA / GQA baseline in their text (0 matches, re-checked).
- **MoR, Bae et al., arXiv 2507.10524** (v1 14 Jul 2025, NeurIPS 2025). **Overlapping** for plain first-loop sharing:
  "recursive KV sharing" = "caching KV pairs exclusively at this initial step and reusing them across all subsequent
  recursions" (§2.2.2); "KV sharing results in only a slight performance drop" (§4); Table 15 NLL +0.009 to +0.016
  (on the train set) — consistent with our +0.011. No own-loop window, no MQA baseline.
- **Parallel Loop Transformer (PLT), Wu et al. (ByteDance), arXiv 2510.24824**, 28 Oct 2025. **Same design a year
  earlier**: **[re-checked]** "we share the first loop's KV cache with all later loops" and non-first loops use
  sliding-window attention, "we set w=64" (gated fusion rather than one softmax).
- Together: **4b is already done.** What is left: (i) the iso-KV-bytes MQA comparison (not found in Monea, PLT, MoR,
  MELT or BeNice!; CLA 2405.12981 is the non-looped precedent); (ii) a small-scale data point where ω=0 is slightly
  worse, disagreeing in sign with Monea.

## Specific question 3: has anyone else reported that Ouro's Table 14 fails to reproduce? **Yes, several.**
| work | setup | quote / numbers | rating vs 4a |
|---|---|---|---|
| **Fold the Loop**, Zhou, Yang, Wang, Li, Jia. arXiv 2611.25369, 19 Nov 2026 **[re-checked]** | Ouro-1.4B, GSM8K n=1319, **3-shot** (Ouro's protocol), greedy, ≤256 tokens, App. H Table S7 | FullKV 78.77; "Generated tokens (last-gen) 68.92 (−9.86)"; prompt-only 6.60 (−72.18); both 0.15 (−78.62). "Ouro does not fully specify which KV states are shared during decoding, and its description is closest to last-all." | **same result** (decode-only −9.86 vs our −9.8, our 4-shot) |
| **CDB**, Schwethelm et al. 2608.09444 **[re-checked]** | Ouro-1.4B GSM8K, App. E.1 Table 4 | "Ouro loses accuracy under both layouts, from about 6 points under first-then-shared to a complete collapse under single-slot sharing. We cannot reproduce the results from the Ouro paper ... This might be because they share only during decoding" | overlapping (prompt+decode; they did not test decode-only, which is exactly our E2) |
| **LoopKV**, Zhou, Zhang, Wu, Wang, Nixon, Yang. arXiv 2611.04137, 3 Nov 2026 | Ouro Thinking, MATH500 / AIME, long outputs | "Decode-time last-step reuse ... this reuse loses 44.4 points at an 8K cap" | overlapping |
| **MELT**, Vendrell, Masdemont, Grillo, Ros-Giralt, Behboodi, Massoli (Qualcomm). arXiv 2605.07721 | Ouro-1.4B-Thinking, AIME / AMC / MATH-500 | "we were unable to fully reproduce the reported results of Ouro" (App. E.1); trains KV sharing in (shared cache + W=50 per-loop window) | overlapping (4a and 4b) |
| Looped Latent Attention, O'Neill & Reid, 2607.15456 | Ouro-1.4B GSM8K 5-shot | final-loop reuse 4×: GSM8K 0.000 (all tokens, not decode-only) | overlapping |
| BeNice!, Prairie et al., 2610.31633 | Ouro 1.4B/2.6B, prompt included | Core 54.57 → 25.17 | overlapping |
| Closing the Loop, Marchenko et al., 2610.00673 | own 1.4B LoopLM | decode last-step reuse "drifted ... KL ... grew from 0.03 to 3.6 over 4,096 decoded tokens" | adjacent |
Where nothing was found: HF ByteDance/Ouro-1.4B discussions (13 threads; #8 reproduces the full-cache GSM8K ~79.4),
GitHub issue search, no OpenReview page for Ouro found.
Ouro's own text (2510.25741 v6, §5.4.2, Table 14): full 78.92, last-step 78.85, averaged 78.73, first-step 18.73;
evaluation "strict match, 3-shot CoT, lm-eval-harness" — our runs are 4-shot; say so.

What is left of 4a: E3 (decode-only sharing **with early exit**, q = 0.5: −2.6 points, depth 3.23 → 3.02) — not found
in any paper opened; and the observation that early exit makes sharing less harmful. Small, but ours.

---

## Contribution 1: measured GPU characterisation incl. LoopSpec — **partly novel**
| paper | what overlaps (quoted, opened) | rating |
|---|---|---|
| **What Sets the Serving Speed of Looped LMs?** Lu, Li, Ni, Tang, Kong, Yang, Guo, Gao. arXiv 2611.05150, 3 Nov 2026 | Ouro-1.4B (fixed depth 4), Huginn, Ouro-2.6B on GH200 and RTX 5090, concurrency 1-64: "At batch 1, fusing all loop submissions leaves Ouro throughput at 0.99×; the 5689 small kernels per token are the real cost"; self-speculation "1.52× ... at 1 ... 0.75× at 32" | overlapping |
| **PackLoop**, Runguo Li, Dezhen Lu. arXiv 2611.30205, 23 Nov 2026 **[re-checked]** | "Ouro-1.4B, Ouro-2.6B ... on an NVIDIA RTX 3090 (24 GB), an H100 NVL (94 GB), and ... an RTX 4090" (A6000, A100 in App. E), 1-192 concurrent requests; baselines vllm-rlt, CDB, CopyLoop, WaveFront, LoopSpec | overlapping (the biggest risk for 1) |
| LoopSpec, Cho, Cui, Kim, Han, Han. arXiv 2609.17184, 15 Sep 2026 | RTX PRO 6000, single batch; Ouro-1.4B 2.64-3.17× | adjacent (our baseline) |
| LoopAhead 2610.21594 (A40, Ouro-1.4B up to 4.3×, re-runs LoopSpec 2.7-3.3×); WaveFront 2609.23033 (A6000, Ouro-2.6B 2.42×) | faster self-speculative baselines than LoopSpec exist | adjacent, but a **baseline-strength risk**: LoopAhead and PackLoop (1.47-2.29× over LoopSpec's best mode) report beating LoopSpec |
| Chen, "Memory-Bound but Not Bandwidth-Limited", arXiv 2605.30571, 28 May 2026 | batch-1 decode on H100, A100, L40S, L4 (generic LLMs): "The achieved fraction of peak HBM bandwidth falls as peak bandwidth rises" | overlapping on the same GPUs (not looped) |
Still ours: L40S and A100 for a looped LM; early exit at q < 1 across GPUs (2611.05150 runs fixed depth); LoopSpec
measured on L40S/H100 at context ~2048 with the TinyGEMM workaround; per-kernel decomposition tied to a model.
**Risk:** the paper's "strongest batch-1 baseline" is no longer LoopSpec; a reviewer will ask for LoopAhead or
PackLoop (PackLoop is 1 day old).

## Contribution 2: analytical model — **partly novel**
| paper | overlap (quoted, opened) | rating |
|---|---|---|
| **KernelSight-LM**, Yao, Kim, Pei, ..., Karypis, Herbordt. arXiv 2606.28565, 26 Jun 2026 | form "t=max(t_roof·η, t_0)"; spec-only 12.1 % vs 3.8 % with a target microbenchmark sweep; "the families for which the learned efficiency factor matters most are precisely those for which it is least transferable" | same result (kernel level) |
| **GenZ**, Bambhaniya et al. arXiv 2406.01698 | "GenZ uses the runtime of real systems ... to calculate the efficiency factor"; per-platform efficiency (A100 0.4, MI300X 0.25) | same result ("needs a measured efficiency factor") |
| LLMCompass, Zhang, Ning, Prabhakar, Wentzlaff, arXiv 2312.03134 | decode error 7.5 %; "Lack of software knowledge" as error source | overlapping |
| LIMINAL, Davies, Crago, Sankaralingam, Kozyrakis, arXiv 2507.14397 | analytical decode model, MAE 7.6 % on 8×H100; "17-30 kernel launches per transformer layer" | overlapping |
| NeuSight (ASPLOS'25, 2407.13853), PipeWeave (2601.14910), WattGPU (2607.02391) | cross-GPU transfer with efficiency terms learned from other GPUs: 2-8.5 % on unseen GPUs | overlapping (a reviewer will ask why we didn't do this) |
| Vidur (2405.05465), LLMServingSim, LLM-Viewer, LIFE, HYMELL, AIConfigurator | profile-based or spec-only; adjacent | adjacent |
Not checked: AMALI (not found), Calculon (SC23 page 403).
Still ours: a **pre-registered, blind** whole-step decode prediction on a held-out GPU, reported as a failure with
sign and size (−21…−29 % H100, −22…−24 % A100); the cause — torch.compile kernels on A100/H100 slower than generic
kernels moving the same bytes, absent on L40S. Not found in any paper opened. Do not claim the model form or "an
efficiency factor is needed".

## Contribution 3: chip design requirements — **partly novel**
| paper | overlap (quoted, opened) | rating |
|---|---|---|
| **SHIP**, Bitar et al. (Groq), MLSys 2026 **[re-checked]** | pipeline parallelism "offers limited improvement in KV cache storage"; "In SHIP, a key bottleneck during decode [is] limited KV-cache capacity"; KV on HBM devices discussed | overlapping with (i), (ii) |
| NVIDIA Groq 3 LPX blogs (Aubrey & Ghodsian, Mar 2026; Weidman et al., Aug 2026) | "Vera Rubin NVL72 computes attention and holds the KV cache in DRAM, while Groq 3 LPX executes the FFN layers"; "Only intermediate tokens are sent between racks" | overlapping with (i), (iv): industry already keeps KV off the SRAM chips and does not stream it |
| **LIMINAL** 2507.14397 **[re-checked]** | "Key Finding 5. When an order of magnitude more memory bandwidth compared to HBM3 is made available, synchronization latencies dominate performance"; SRAM systems "capacity constrained" | overlapping with (iii) (method), no looping |
| Chiplet Cloud 2307.02666; Ouroboros (ASPLOS'26, 2603.02737); Theseus 2407.02079 | larger KV → more chips / pipeline under-use in SRAM-only designs; TCO / per-area normalisation | overlapping with (ii) in general and with the per-mm² / per-cost tables |
| **MixerLoop**, Lin, Guo, Zhu, Ye, Eshraghian. arXiv 2608.18230 **[re-checked]** | "We implement a custom FPGA accelerator on a KV260 that keeps these weights on chip"; looped-LM locality "remains an open systems problem" | adjacent (13M model, edge FPGA), but it makes "first hardware for looped LMs" false |
| EdgeBERT (MICRO'21), IBM analog ALBERT chip (Nat. Comms 2025) | weight-shared encoder chips | adjacent |
| WaferLLM, SN40L ("The router and KV cache is always in HBM"), FastAFD 2611.07149, HeteroPanacea 2608.03741, LLMCompass | adjacent |
Not found anywhere: a hardware study of looped *decoder* LLMs on SRAM-only multi-chip silicon; KV placement or
cross-loop KV sharing expressed as chip count; a per-layer µs threshold vs H100; an explicit analysis that KV streaming
over chip links does not pay. **Caveat for (ii):** its 10-12 chip configuration needs a model that tolerates one shared
cache; Ouro does not (4a), so state (ii) as a requirement on the model and cite Monea / PLT as models that meet it.

---

## Strongest honest framing
"A design-requirements study of SRAM-only, multi-chip dataflow hardware for looped decoder LLMs, calibrated against
measurements on three GPUs." Specifically:
1. GPU: Ouro-1.4B on L40S, H100 and A100 with early exit, consistent with and extending 2611.05150 and PackLoop;
   LoopSpec as a baseline, acknowledging LoopAhead / PackLoop as faster.
2. Model: the form follows GenZ / LIMINAL / KernelSight-LM; the contribution is a pre-registered blind cross-GPU test
   and its cause (compiled vs generic kernels).
3. Chip: to our knowledge the first analysis of KV placement and cross-loop KV sharing as latency and chip-count drivers
   for looped LMs on SRAM-only hardware; quantified (~6 µs/layer threshold; streaming never pays), complementing SHIP
   and LIMINAL. Not "the first looped-LM hardware" (MixerLoop).
4. Model-side: an independent replication — Ouro's decode-only sharing loses ~10 GSM8K points (as Fold the Loop and
   others found), early exit reduces the loss to −2.6; trained-in first-loop + window sharing matches per-loop caches
   at 81M (replicating Monea and PLT), and beats MQA at equal KV bytes (new). Cite these as requirements met by
   existing training recipes, not as discoveries.
5. CDB: Schwethelm et al.'s method, reimplemented; mention only as the batching used, with the q = 1 equivalence test.

## Must cite
Schwethelm et al. 2608.09444 (CDB); Fold the Loop 2611.25369; Monea et al. 2610.02383; PLT 2510.24824; MoR 2507.10524;
MELT 2605.07721; LoopKV 2611.04137; 2611.05150; PackLoop 2611.30205; LoopSpec 2609.17184; LoopAhead 2610.21594;
SHIP (MLSys'26); LIMINAL 2507.14397; GenZ 2406.01698; KernelSight-LM 2606.28565; Chen 2605.30571; MixerLoop 2608.18230;
Chiplet Cloud 2307.02666; RRT (Bae et al., ICLR 2025); Ouro 2510.25741. Should cite: Looped Latent Attention
2607.15456, BeNice! 2610.31633, CLA 2405.12981, LLMCompass 2312.03134, NeuSight 2407.13853, Theseus, Ouroboros,
WaferLLM, SN40L, NVIDIA LPX blogs, EdgeBERT.

## Limits of this check
Semantic Scholar (HTTP 429) and Google Scholar (403) were mostly unavailable; ISCA/MICRO/HPCA/ASPLOS proceedings
pages were not browsed directly, so non-arXiv conference papers may be missed. The field is moving fast: two of the
closest papers (Fold the Loop, PackLoop) appeared in the last six days; re-run this check before submission.
Not verified (listing or snippet only): LoopVQ 2611.05304, Loopy 2610.05265, LoopQuant 2610.16368, Late Entry
2611.24576, Dual-Slot 2610.25086, LoopSight 2611.28575, DREX 2512.15705, AMALI, Calculon.
