# Lab notebook: can Ouro-1.4B share its KV cache across loops without training?

Append-only. Started 2026-11-23.

# Frame
- **Question.** Is there a form of cross-loop KV sharing that Ouro-1.4B (rev 574fa66, as trained) tolerates on GSM8K
  under the Week 6 rule, and which form? Specifically: does the Ouro paper's "decode-only, last-step reuse" (§5.4.2,
  Table 14: 78.85 vs 78.92) reproduce in our stack, where our Week 6 sharing (kv1 3.9 %, kv2 39.3 % vs bf16 77.8 %)
  collapsed?
- **Why.** Every route to 10x at B=1 in the current tables uses "trained shared KV, on-chip". If Ouro shares for free in
  some form, that form is available on the real 1.4B model without Week 6b training.
- **Metric / rule.** Week 6 rule (CALIBRATION.md, approved 2026-11-21): pass if the lower end of the 95 % interval on the
  paired accuracy change vs bf16 (same container, same q) is above −2 points; exact McNemar p and exit depth reported.
- **Budget.** Step 1 (this file, until Sid approves anything): no GPU. Step 2: one 1319-question L40S run per variant,
  at most 3 variants, estimates before launch.

# Step 1.1 - what the papers did (text opened 2026-11-23 from arxiv.org/pdf, pdftotext)

## Ouro, arXiv 2510.25741 v6 (11 Oct 2026), §5.4.2 "KV Cache Sharing for Inference Efficiency", Table 14
Quotes:
- Prefill: "we find that all four recurrent steps require their own KV caches, as each step transforms the
  representations in ways that cannot be approximated by earlier steps. Attempting to reuse KV caches during
  prefilling leads to performance degradation (>10 points on GSM8K)."
- Decode: "during the decoding phase (auto-regressive generation), we discover that KV cache reuse becomes viable.
  [...] 1. Last-step reuse: Only maintain KV cache from the final (4th) recurrent step 2. First-step reuse: Only
  maintain KV cache from the first (1st) recurrent step. 3. Averaged reuse: Maintain an averaged KV cache across all
  four steps"
- Table 14 (GSM8K / MATH-500): Full (4x cache) 78.92 / 82.40; First-step only 18.73 / 8.43; Last-step only
  78.85 / 80.40; Averaged 78.73 / 78.52.

| item | what the paper says | status |
|---|---|---|
| which loop's KV is kept | the final (4th) step's, for decode tokens | stated |
| early exit on or off | not stated in §5.4.2 | **inferred: off (all 4 loops).** Table 14's "Full" = 78.92 / 82.40 is exactly Table 7's "Ouro 1.4B R4" column (R4 = 4 recurrent steps). The released config has `early_exit_threshold: 1.0`, and the released `modeling_ouro.py` (lines 592-608) runs all 4 loops for every token and writes every loop's KV whatever the exit; exit only picks which loop's logits are used. So in an HF-based eval, the "final step" KV always exists. |
| prompt / prefill KV | "all four recurrent steps require their own KV caches" during prefill | stated: the prompt keeps 4 caches; sharing it costs > 10 GSM8K points (no exact number, no table) |
| which positions read the shared cache | not stated beyond "only maintain KV cache from the final (4th) recurrent step" in decoding | inferred: generated positions hold loop-4 KV and every loop reads it; the token in flight necessarily uses its own current-loop K/V (loop 4 does not exist yet when loop 1 runs) |
| model | "our 4-step model"; numbers match Table 7's 1.4B column | inferred: Ouro-1.4B base |
| eval setup | not stated in §5.4.2. Table 16 (App. C.1) for the base models: GSM8K "strict match, 3-shot CoT", lm-eval-harness | inferred to apply because Full = Table 7 value. Decoding and max length for GSM8K are not given (the "greedy decoding with max_new_tokens=128" sentence in §7.1 is for the HEx-PHI safety eval, not GSM8K). lm-eval's gsm8k defaults are greedy with stop strings (from memory, not checked). |
| implementation | not given; the released HF code has no sharing option (`UniversalTransformerCache`, lines 129-170: one slot per loop x layer) | stated by absence |
| n | not stated; no error bars | – |

Note the text says "We explore two strategies" and then lists three.

## Looped Latent Attention, arXiv 2607.15456 v2 (24 Jul 2026)
Quotes:
- §3 "Matched-axis baselines": "We also include a zero-parameter control that caches only the final-loop K/V and
  reuses it for every loop. That control has the same 4x cache footprint as LLA for T = 4".
- App. A: "final-loop reuse: a zero-parameter control that caches only the last loop's K/V and reuses it for all loops."
- Table 1: teacher GSM8K 0.794; final-loop reuse (4x) 0.000 (MC-avg 0.443 vs 0.646, MMLU 0.537 vs 0.682).
- §5: "Its reconstruction relative error is 0.35 for K and 0.60 for V."
- App. A protocols: "GSM8K: 5-shot at 1.4B [...] strict-match EM, max_gen_toks=256. The 1.4B teacher is scored on the
  full 1319-item test set; the iso-cache and broad-suite codecs use a 100-item subset with the faithful re-forward
  decoder". Models "run at T = 4 recurrence steps".
- App. A decoder: "pass 1 captures the teacher's pre-RoPE per-loop K/V, the codec reconstructs them, and pass 2
  re-forwards forcing the reconstruction so the model's own RoPE and cache store the post-RoPE reconstruction."

| item | LLA | status |
|---|---|---|
| which loop's KV | final loop | stated |
| early exit | T = 4 fixed; no early exit mentioned | stated (T = 4); exit not mentioned |
| prompt | the codec replaces what the cache stores; nothing says the prompt is exempt | **inferred: prompt shared too** (the control is a codec swap over the whole cache) |
| positions | "reuses it for all loops" | stated; whether the in-flight token also reads its own final-loop K/V in the re-forward pass is not stated |
| eval | 5-shot, strict EM, max 256 tokens, 100-item subset for the control, teacher on 1319 | stated |

# Step 1.2 - differences: Ouro Table 14 vs LLA vs our Week 6 vs week6_share_check

| dimension | Ouro "last-step only" (78.85) | LLA final-loop reuse (0.000) | Week 6 `kv1` (3.9 %) | week6_share_check "shared" (collapses, 4 questions) |
|---|---|---|---|---|
| prompt KV | **4 caches (not shared)** | shared (inferred) | **shared**: loop-4 KV for all loops | **shared**: loop-4 KV for all loops |
| generated tokens' KV | loop 4 only | loop 4 only | last writer = the token's exit loop | loop 4 (all loops run) |
| early exit | off (inferred) | off | **on, q = 0.5**, depth 2.59 after sharing | off (q = 1.0; log: "100.0% of decode tokens ran all loops") |
| copy-forward | n/a | n/a | none needed (1 cache) | n/a |
| loops actually run per token | 4 (HF always runs 4) | 4 | 1-4 (skipped loops never computed) | 4 |
| shots / max tokens | 3-shot CoT (lm-eval), max not stated | 5-shot, 256 | 4-shot, 256 | 4-shot, 64 |
| n | not stated | 100 | 1319 | 4 |
| implementation | not public | two-pass re-forward | our KVStore (CDB) | HF `UniversalTransformerCache`, slots overwritten |

The one difference that separates "works" from "collapses" in this table is the prompt. Both collapses share the
prompt; the only reported success keeps 4 prompt caches. And the paper itself reports that sharing at prefill costs
> 10 points. **Week 6 never tested decode-only sharing.** The Week 6 plan already said "a failure here does not rule
out sharing"; HANDOFF and CALIBRATION.md's "consistent with ... the literature (Ouro §5.4.2)" reads Table 14 as
agreeing with us, but Table 14's success is a different form.

# Step 1.3 - competing hypotheses for the collapse

| # | hypothesis | predicts for decode-only last-step reuse, q = 1.0 | cheapest killing test | status |
|---|---|---|---|---|
| H1 | **Prompt sharing** is what breaks Ouro: loops 1-3 reading loop-4 K/V over the few-shot prompt corrupts every step; sharing only generated tokens is near free (the paper's claim). | ≈ bf16 accuracy | E1 below (decode-only on a few questions) | alive, favoured |
| H2 | **Early-exit stale KV**: a token that exits at loop k leaves loop-k KV that later tokens' loops read as if final. | (q = 1.0 has no early exit, so H2 predicts nothing here; it predicts a q = 0.5 decode-only drop) | existing data | **dead as the cause of the collapse**: week6_share_check ran at q = 1.0 (100 % of tokens ran all loops) and collapsed the same way as kv1 at q = 0.5. Survives as a possible *extra* cost of decode-only sharing at q = 0.5 (the chip-relevant case): E3. |
| H3 | **Any reuse of loop-4 K/V by loops 1-3 breaks Ouro**, decode too; Table 14 does not reproduce (different implementation, a bug in theirs, or a setting we can't see). LLA's 0.000 is consistent with this. | collapse, like the full-sharing control | E1 | alive |
| H4 | **Eval settings** (3- vs 4-shot, stop strings, max length, strict vs flexible match) explain the gap. | – | existing data | **dead**: unshared bf16 gives 77.8 % (ours, 4-shot) vs 78.92 (paper, 3-shot), so the pipelines agree within ~1 point without sharing; and share_check's shared outputs degenerate within the first 1-16 generated tokens ("16 - 3 - e = 4 eggs", repeated "Question:"), which no shot count, stop string or max length can rescue. |
| H5 | **A bug in our KVStore/CDB sharing path.** | – | existing data | **dead**: share_check's independent HF-cache reference produces the same degenerate text as the engine (q3 identical for all 64 tokens; q0, q1 for the first 13-16; q2 diverges at token 1, but both engine and HF outputs
  are wrong and degenerate). |

So after existing data: H2 (as sole cause), H4, H5 are dead. H1 vs H3 is the open question, and one test separates
them. If H1 survives, H2 decides whether the chip (which skips loops, so has no loop-4 KV for early exits) gets it.

# Step 1.4 - chip relevance (scratch script, loopcore_model.py functions unchanged)

Script: `decode_only.py` in this session's scratchpad (not in the repo yet; ask Sid whether to add it; calls `kv_footprint`, `fit_check`,
`token_time`, `chip_token_time_loopspec`, `layer_latency_bottom_up`, `loopspec_stats` as they are). Config D, int8
weights, fp8 KV, q = 0.5, depth 3.22, C = P + G = 2048, H100 baselines as in `--loopspec-table` (int8 B=1 5.01 ms,
B=16 15.87 ms; LoopSpec B=1 3.74 ms measured). Decode-only footprint = 4 caches x P + 1 cache x G; chips = fewest
500 MB chips holding weights + LM head (1334 MB) + KV. Validity: the "per-loop, HBM" rows reproduce `--loopspec-table`
"as trained" (7.0x / 8.9-9.9x / 4.3x at install-low) and the "full, on-chip" rows reproduce "trained shared KV,
on-chip" (10.7x / 24.9-25.7x / 22.2x).

B=1, 1x2048 (b = chip without LoopSpec vs GPU + LoopSpec; c = both with LoopSpec; install int8 low / high t_L):

| P / G | design | KV MB | chips | (b) low / high | (c) low / high |
|---|---|---:|---:|---|---|
| any | per-loop, HBM (as trained) | 805 | 3 | 7.0 / 5.0x | 8.9-9.9 / 7.5-8.2x |
| any | per-loop, **on-chip** | 805 | 5 | 10.6 / 6.6x | 24.7-25.4 / 16.3-16.6x |
| 1792 / 256 | decode-only, on-chip | 730 | 5 | 10.6 / 6.6x | 24.7-25.4 / 16.3-16.6x |
| 1024 / 1024 | decode-only, on-chip | 503 | 4 | 10.7 / 6.6x | 24.9-25.7 / 16.4-16.7x |
| 256 / 1792 | decode-only, on-chip | 277 | 4 | 10.7 / 6.6x | 24.9-25.7 / 16.4-16.7x |
| any | full sharing, on-chip | 201 | 4 | 10.7 / 6.6x | 24.9-25.7 / 16.4-16.7x |

B=16, 16x2048, vs H100 int8:

| P / G | per-loop HBM | per-loop on-chip | decode-only on-chip | full on-chip |
|---|---|---|---|---|
| 1920 / 128 | 3 chips, 4.3 / 4.0x | 29 chips, 20.4 / 15.9x | 28 chips (12.3 GB KV), 20.5 / 15.9x | 10 chips (3.2 GB), 22.2 / 16.9x |
| 1792 / 256 | " | " | 27 chips (11.7 GB), 20.5 / 16.0x | " |
| 1024 / 1024 | " | " | 19 chips (8.1 GB), 21.3 / 16.4x | " |
| 256 / 1792 | " | " | 12 chips (4.4 GB), 21.9 / 16.8x | " |

Findings (model-derived, not measured):
1. **In this model, sharing does not set the speed; placement does.** Every loop step reads one cache of C tokens
   whether caches are shared or not (`step_terms`), so per-loop caches *on-chip* give the same speedup as full sharing
   (B=1: 10.6 vs 10.7x row b; B=16: 20.4 vs 22.2x), only with more chips. Sharing is a chip-count (SRAM, cost) lever.
   The "every route to 10x needs shared KV on-chip" framing should read "needs KV on-chip; sharing decides how many
   chips that takes."
2. At 1x2048, keeping all 4 per-loop caches on-chip costs 5 chips instead of 4 (805 MB of KV). Sharing is close to
   irrelevant at B=1 at this context.
3. Decode-only sharing helps in proportion to the generated fraction. GSM8K-like workloads are prompt-heavy (our 3-4-shot
   prompts are ~600-900 tokens vs ~100-250 generated; NOTEBOOK_LOOPSPEC.md frame), so decode-only saves ~10-25 % of
   KV: at 16x2048 with G = 256 it needs 27 chips vs 29 per-loop vs 10 full sharing. It approaches full sharing only for
   generation-heavy work (long chain-of-thought, chat with short prompts): G = 1792 → 12 chips.
4. Not modelled: a split placement (prompt's 4 caches in HBM, shared generated cache on-chip). Per loop step it would
   still read the prompt part from HBM, so it would sit between the HBM and on-chip rows.

# Pre-registration: E1 (cheap screen, H1 vs H3) - NOT RUN, waiting for Sid

- **Change (one thing vs share_check's control):** in the HF-cache reference of `week6_share_check.py`, keep the
  prompt's 4 per-loop caches (skip the prompt `share_last_loop`); after each decode step overwrite loops 1-3's newest
  position with loop 4's K/V (as now). All 4 loops run (q = 1.0). Arms on the same questions: unshared; full sharing
  (existing rule, positive control); decode-only last-step.
- **Questions:** first 8 GSM8K test questions, the Week 6 4-shot prompt, greedy, 256 new tokens, graded with
  `quality_check`'s grader.
- **Prediction (H1):** decode-only gets ≥ 6/8 of the answers the unshared arm gets right, and no degenerate loops;
  full sharing ≤ 2/8 (reproducing the collapse).
- **Meaning:** decode-only ≥ 6/8 → H3 dead, go to E2. Decode-only ≤ 2/8 → H1 dead, H3 survives: Table 14 does not
  reproduce in our stack; Ouro cannot share without training; E2 not run. 3-5/8 → inconclusive at this n, go to E2
  only with Sid's OK.
- **Cost:** laptop (free, but needs the ~2.9 GB weights and a transformers 4.57 venv; the laptop has 5.17, which the
  remote code may not load) or Modal L40S ~10 min ≈ $0.35.

# Planned (Step 2, to be pre-registered in full before launch)
- **E2:** decode-only last-step, q = 1.0, vs bf16 at q = 1.0 in the same container, all 1319 questions (closest to the
  paper's own setup). Needs a KVStore mode: prompt in 4 caches, generated positions in 1.
- **E3:** decode-only, q = 0.5 (generated positions hold the exit loop's KV: the chip-relevant form), vs bf16 q = 0.5.
  Separates H2 as an extra cost.
- Optional E4: averaged reuse, only if E2 passes and E3 fails.
