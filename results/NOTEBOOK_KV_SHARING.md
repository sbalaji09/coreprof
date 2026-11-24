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

# E1 - result (run `kvshare_e1-l40s-20261124-001132`, L40S checked in env.txt, git fc51386-dirty: the dirty files are the other session's CALIBRATION.md / train_looped.py, not used here; ~10 min, ≈ $0.35)
Command: `./run_modal.sh --task kvshare_e1` = `python week6_share_check.py --e1 --n 8 --max-new-tokens 256`. HF reference,
all 4 loops (q = 1.0), first 8 GSM8K test questions, 4-shot (prompts 496-580 tokens), greedy, 256 new tokens.

| arm | correct | degenerate outputs | first token that differs from unshared |
|---|---:|---:|---|
| unshared | 4/8 | 0/8 | – |
| full sharing (prompt + decode; Week 6 rule) | 0/8 | 8/8 (garbled numbers, "e c a c", repeated "#### e") | tokens 1-12 |
| decode-only (prompt keeps 4 caches) | 5/8 | 0/8 | tokens 6-159 |

- Decode-only is right on all 4 questions unshared gets right (4/4), plus q4 (unshared answered 2, decode-only 20 = gold).
  Its 3 misses are questions unshared also misses (q2 same wrong answer 120000; q5 80 vs 60; q7 120 vs 100).
- Its text drifts from unshared within 6-159 tokens, but every output stays coherent, ends with "#### <n>", and follows the
  same reasoning steps. Full sharing reproduces the old collapse on all 8.

**Verdict against the pre-registered rule: inconclusive (5/8 is in the 3-5/8 band).** The rule's thresholds were absolute
and assumed the unshared arm would get ~6/8 (77.8 % overall); on these first 8 questions it got only 4/8, so "≥ 6/8"
could not be reached even by a perfect copy of unshared. I wrote the rule badly; I am not reclassifying the result.
What the data does show, without the rule: decode-only matched unshared on every question it got right and never
degenerated; full sharing degenerated on every question. That is strong qualitative evidence for H1 and against H3
(H3 predicted decode-only would collapse like full sharing). n = 8, so it is not an accuracy estimate: per the
pre-registration, E2 runs only with Sid's OK.

# Per-loop caches on-chip vs sharing, sourced latency range (kv_decode_only.py `sweep()`, model only, 2026-11-24)
Config D, int8, fp8 KV, q = 0.5, C = 2048 (all prompt; sharing form doesn't change speed, only chips). t_L from
`install_int8` with 20/25/30/35/40 cycles per install (3.83 / 4.54 / 5.25 / 5.97 / 6.68 µs at B=1). (b) chip vs H100 +
LoopSpec 3.74 ms; (c) both LoopSpec; B=16 vs H100 int8 15.87 ms. Endpoints reproduce `--loopspec-table`.

| design | chips B=1 / B=16 | (b) B=1, eff 1.0 → 1.3 over 3.83-6.68 µs | /chip | (c) B=1 | /chip | B=16 | /chip |
|---|---|---|---|---|---|---|---|
| per-loop, on-chip | 5 / 29 | 10.6 … 5.0x | 2.13 … 1.01 | 25.4 … 12.5x | 5.1 … 2.5 | 20.4 … 12.2x | 0.70 … 0.42 |
| per-loop, HBM (as trained) | 3 / 3 | 7.0 … 3.8x | 2.34 … 1.27 | 9.9 … 5.7x | 3.3 … 1.9 | 4.3 … 3.1x | 1.42 … 1.03 |
| full sharing, on-chip | 4 / 10 | 10.7 … 5.1x | 2.69 … 1.27 | 25.7 … 12.6x | 6.4 … 3.2 | 22.2 … 13.0x | 2.22 … 1.30 |

Reading: at B=1, per-loop on-chip (no sharing, no training) costs one extra chip and gives within ~0.3x of full sharing;
row (c) ≥ 12.5x across the whole sourced range. At B=16 it needs 29 chips and is the worst design per chip (0.42-0.70,
below HBM's 1.0-1.4). Model caveats: extra chips cost only hop latency (1 µs each); cost, power and links not modelled.
Proposed numbers only; no write-up changed.

# Pre-registration: E2 and E3 (approved by Sid 2026-11-24, ~$4-5; written and committed before any data)
- **Code:** `KVStore(share_decode=True)` (quality_check variant `kvdec`). The prompt keeps its 4 per-loop caches
  (`load_prefill` unchanged). Every decode write goes to all 4 caches, so a generated position holds the KV of the last
  loop that wrote it and every loop of later tokens reads that (one shared cache for generated tokens); the token in
  flight sees its own current-loop KV. Storage stays 4x (accuracy only). CPU tests: `tests/test_kv_store.py`
  (`test_share_decode_*`; the multi-row test fails on a first version that broadcast wrongly, checked).
- **Run:** Modal task `kvshare_e23`, one L40S container (stops unless `nvidia-smi` says L40S; I also check env.txt):
  selfcheck, KV-store tests on GPU, 8-question smoke, then all 1319 GSM8K test questions, 4-shot, greedy, 256 new tokens,
  CDB 32 slots, in this order: bf16 q=1.0, bf16 q=0.5, kvdec q=1.0, kvdec q=0.5. Each kvdec is paired with bf16 at
  the same q.
- **Rule (Week 6 rule, unchanged):** pass if the lower end of the 95 % interval on the paired accuracy change vs bf16
  is above −2 points; exact McNemar p, question-by-question counts and exit depth reported; depth change > 0.10 loops
  means LoopCore claims for that form use the measured depth.
- **E2 (kvdec, q=1.0; the paper's setup: last-step KV, all loops).** Prediction (H1): pass, change within −1.5 … +1
  points (paper: −0.07). Pass → Ouro-1.4B as trained can share generated tokens' KV in the paper's form. Fail → H1's
  "near free" is refuted at scale in our stack, whatever E1 suggested; the paper's Table 14 does not reproduce.
- **E3 (kvdec, q=0.5; the chip's form: generated positions hold the exit loop's KV).** Prediction: pass but with a
  larger cost than E2, change −2.5 … +0.5 points (I am ~60 % on pass); exit depth within ±0.10 of bf16. Meaning:
  pass → the chip can use decode-only sharing with early exit. E2 pass + E3 fail → H2 survives as an extra cost:
  non-final-loop KV read as final hurts, and decode-only sharing on the chip would need all 4 loops for KV (losing early
  exit's savings) or a different rule. Both fail → decode-only sharing is not usable on Ouro as trained.
- No variant is re-tuned or re-run on failure without a new entry here.

# P1 - KV placement term (approved by Sid 2026-11-24; pre-registered, committed before any table is run)
**Why.** The model reads on-chip KV at per-chip SRAM bandwidth wherever it sits (`step_terms`), so per-loop caches on 5
chips looked as fast as one shared cache on 4. But with the weights packed on the fewest chips (3 chips, 1334 MB, 166 MB
spare), ~80 % of per-loop KV at 1x2048 sits on chips holding no layer. The old per-loop on-chip numbers (Step 1.4 and the
sweep above) are withdrawn until this term is in; they are not to be quoted.

**Change (new option, defaults unchanged; `--loopspec-table`, `--kv-levers` and the default report print identical output
before and after).** `Hardware.kv_placement`: "aggregate" (default, old behaviour), "stream", "near".
`kv_remote_fraction` r = KV not fitting in the weight chips' spare SRAM / total KV.
- stream: + r x (this step's cache bytes) / link_bw per loop step (prefetched: bandwidth-bound, latency hidden).
- near: + ceil(r x 24) remote layers x 2 hops x (hop latency + B x 2048 x 2 B / link_bw) per loop step.

**Link parameters (results/LATENCY_SOURCES.md, Groq rows), applied to every design in the new table, including the
existing per-chip pipeline hops:** hop latency 0.30 µs (F, LPU v1, vendor) .. 0.722 µs (B, TSP v1); 1 µs (current
assumption) as a sensitivity row. Link bandwidth 235 GB/s (F, LPU v1, 11 ports aggregate) .. 1.344 TB/s (G, Groq 3,
96 x 112 Gbps; Derived, raw line rate, all links to one peer: optimistic for a point-to-point stream). Pairing for the
range: (0.30 µs, 1.344 TB/s) = optimistic, (0.722 µs, 235 GB/s) = pessimistic.

**Predictions (hand arithmetic, t_L 3.83 µs, eff 1.0, row b = chip without LoopSpec vs GPU + LoopSpec).**
- r: per-loop 1x2048 0.79 (20 of 24 layers remote); shared 1x2048 0.17 (5 layers); 16x2048 per-loop 0.99, shared 0.95
  (23-24 layers).
- per-loop, stream, B=1: +119 .. 681 µs per loop step (on ~108) → row b ~4.5x .. ~1.5x, **below per-loop in HBM (7.0x)**
  at every sourced bandwidth (no link here beats the assumed 3.35 TB/s HBM).
- per-loop, near, B=1: +12 .. 29 µs per step (+40 µs at 1 µs hops) → row b ~9-10x optimistic, ~8x pessimistic, ~7.5-8x at
  1 µs: a 10-30 % loss vs shared.
- shared, near, B=1: +3 .. 7 µs per step → within ~5 % of the aggregate number. shared, stream, B=1: +26 .. 149 µs →
  ~8x .. ~4x.
- B=16 (vs H100 int8): stream collapses for both designs (≥ 2 ms per step; below the HBM design's 4.3x). near: both lose
  ~10-20 %; per-loop also pays 29 vs 10 chips of pipeline hops; per-chip ordering unchanged (shared ≫ HBM > per-loop).
- Expected headline: "stream" is never viable with sourced links; "near" costs per-loop at B=1 about 10-30 % and shared
  about 5 %, so at B=1 per-loop on-chip stays well above HBM but clearly below shared.

**Caveat I'll report with the table (layout, not a free parameter).** r assumes the weights packed on the fewest chips
with KV-only chips beside them (decision 3's separate KV unit). A different layout spreads the layers over all chips and
keeps each layer's KV beside its weights (r = 0; "aggregate" numbers become real), possible only where one layer's
weights + KV fit one chip: 1x2048 per-loop 51 + 34 MB (fits), 16x2048 per-loop 51 + 537 MB (does not), 16x2048 shared
51 + 134 MB (fits). The table will show this co-located row where it fits.

# P1 - results (`python kv_decode_only.py` → results/kv_placement_p1.txt; model only, proposed numbers)
Ranges over t_L 3.83-6.68 µs (install_int8, 20-40 cycles) x eff 1.0-1.3; (b) chip vs H100 + LoopSpec 3.74 ms, (c) both
LoopSpec, B=16 vs H100 int8 15.87 ms; 1x2048 / 16x2048. Remote fraction r: per-loop 0.79 / 0.99, shared 0.18 / 0.95.
Check: with 1 µs hops the co-located rows reproduce the old aggregate numbers (they are the same model); the per-loop
HBM rows are unchanged by link parameters except the per-chip hop term (≤ 0.1x).

Sourced links, optimistic (0.30 µs, 1.344 TB/s) .. pessimistic (0.722 µs, 235 GB/s):

| design | chips 1x / 16x | (b) B=1 | /chip | (c) B=1 | /chip | B=16 | /chip |
|---|---|---|---|---|---|---|---|
| per-loop, HBM (as trained) | 3 / 3 | 3.8-7.1x | 1.28-2.37 | 5.8-9.9x | 1.92-3.31 | 3.1-4.3x | 1.03-1.42 |
| per-loop, on-chip, near | 5 / 29 | 4.3-9.9x | 0.87-1.97 | 10.6-23.5x | 2.13-4.69 | 10.6-20.6x | 0.37-0.71 |
| per-loop, on-chip, stream | 5 / 29 | 1.0-5.2x | 0.21-1.04 | 0.7-5.3x | 0.14-1.06 | 0.3-1.9x | 0.01-0.07 |
| per-loop, on-chip, co-located | 5 / – | 5.1-11.0x | 1.02-2.20 | 12.6-26.2x | 2.52-5.23 | does not fit (588 MB/layer) | – |
| shared, on-chip, near | 4 / 10 | 4.9-10.7x | 1.22-2.68 | 10.9-24.0x | 2.72-5.99 | 11.3-21.3x | 1.13-2.13 |
| shared, on-chip, stream | 4 / 10 | 2.7-8.8x | 0.69-2.21 | 0.8-6.2x | 0.21-1.56 | 0.3-2.0x | 0.03-0.20 |
| shared, on-chip, co-located | 4 / 10 | 5.1-11.0x | 1.27-2.76 | 12.6-26.2x | 3.16-6.56 | 13.0-22.8x | 1.30-2.28 |

1 µs hops (sensitivity, 1.344 TB/s): per-loop near (b) 4.1-7.8x, (c) 10.2-18.7x, B=16 10.5-16.8x (0.36-0.58/chip);
shared near (b) 4.8-9.8x, (c) 10.5-19.7x, B=16 11.2-18.2x (1.12-1.82/chip); co-located rows as optimistic within 0.4x.

Against the predictions:
- per-loop stream B=1 below HBM at every sourced bandwidth: **matched** (5.2x at best vs HBM 7.1x; predicted ~4.5 .. 1.5x,
  got 5.2 .. 1.5x).
- per-loop near B=1, t_L 3.83, eff 1: predicted ~9-10 / ~8 / 7.5-8x (opt / pess / 1 µs); got 9.9 / 8.5 / 7.8x: **matched**.
- shared near B=1 within ~5 % of co-located: got −3 % (opt), −6 % (pess), −8 % (1 µs): **roughly**, slightly worse.
- shared stream B=1 ~8 .. ~4x: got 8.8 .. 4.5x: matched. B=16 stream collapses below HBM: matched (0.3-2.0x).
- B=16 near loses ~10-20 %: shared 22.8 → 21.3 (opt), 22.2 → 18.4 (pess): matched. Per-chip ordering at B=16
  shared (1.1-2.3) ≫ HBM (1.0-1.4) > per-loop (0.4-0.7): matched.
- **Not predicted:** with LoopSpec on the chip (row c) "stream" collapses even for shared KV (0.8-6.2x), because each
  lane re-reads its context and every lane's remote share crosses the link. Near/co-located are unaffected.

Reading:
1. Streaming KV over chip-to-chip links is never viable with sourced links (≤ 1.344 TB/s vs 150 TB/s SRAM). Remote KV
   must be read where it lives: attention on the KV chip, or KV beside its layer.
2. **Layout decides the B=1 per-loop cost, not sharing.** My earlier warning (b) assumed weights packed on 3 chips plus
   KV-only chips. At 1x2048 one layer's weights + per-loop KV (51 + 34 MB) fit on one chip, so spreading the 24 layers
   over 5 chips with each layer's KV beside it costs nothing extra: 5.1-11.0x (b), 12.6-26.2x (c), within 0.1x of shared
   co-located on 4 chips. With a separate KV unit ("near") per-loop loses 7-15 % at B=1 (row b) vs shared's 3-6 %.
3. At 16x2048 per-loop cannot be co-located (51 + 537 MB per layer > 500 MB), so it needs near-memory attention on 29
   chips: 10.6-20.6x but 0.37-0.71 per chip, the worst of all designs per chip. Shared co-located: 13.0-22.8x at
   1.30-2.28 per chip. Sharing (Week 6b, or a trained-in form) is what makes on-chip KV pay at B=16.
4. Old "per-loop on-chip" numbers (Step 1.4, sweep) stay withdrawn; the co-located row replaces them, with its layout
   condition stated.

# Correction to "P1 - results", reading 2 (2026-11-24)
Reading 2 says the separate-KV-unit ("near") layout costs per-loop "7-15 %" and shared "3-6 %" at B=1. I read those off a
few endpoint rows by eye. Computed over every row of results/kv_placement_p1.txt (cost = 1 − near / co-located, same
t_L, eff and links), the right ranges are:

| links | per-loop, row b | per-loop, row c | shared, row b | shared, row c |
|---|---|---|---|---|
| optimistic (0.30 µs, 1.344 TB/s) | 6-11 % | 6-10 % | 1-4 % | 6-9 % |
| pessimistic (0.722 µs, 235 GB/s) | 14-22 % | 16-22 % | 3-6 % | 13-20 % |
| 1 µs hops (sensitivity) | 18-27 % | 18-26 % | 5-8 % | 17-24 % |

Sourced range: per-loop 6-22 % (b and c); shared 1-6 % (b), 6-20 % (c). With LoopSpec on the chip the shared cost is
larger because each extra lane's pass pays the remote layers' round trips too. Decision 3's amendment (CALIBRATION.md,
2026-11-24) uses these corrected ranges, not the 7-15 % / 3-6 % Sid approved; flagged to him.

# R1 - fresh-eyes check of the kv_placement code (7c54f6b + kv_decode_only.py), 2026-11-24
A subagent that did not write the code got only loopcore_model.py, kv_decode_only.py, tests/ and LATENCY_SOURCES.md
(no notebook, handoff or CALIBRATION.md). It confirmed defaults unchanged (108 configs hashed old vs new: identical).
Findings, each re-checked by me with the model before acting:

Clear bugs (confirmed):
1. `chip_token_time_loopspec` passes LoopSpec's lanes (3.49-4.05) as B, so `kv_remote_fraction` sizes the cache as 3.5-4
   sequences: shared B=1 r = 0.764 instead of 0.175 (checked). Affects row (c) near/stream only.
2. Under "near"/"stream" the pipeline still charges `hops = n_chips`, including KV-only chips the token never visits
   (per-loop B=16: 29 hops instead of the 3 weight chips; checked). "near" already charges its own 2 hops per remote layer.
3. `colocated_fits` checks one layer + its KV against one chip, not that whole layers pack into the chip count: shared
   16x2048 is 186 MB per layer → 2 per chip → 12 chips, the table used 10 (checked; LM head fits on a 2-layer chip).
   Per-loop 1x2048 (5 chips) and shared 1x2048 (4 chips, the LM head needs the 4th) are right.

Modelling choices (recorded, not changed):
4. Link bandwidth is the chip's aggregate I/O (Groq 3: 96 x 112 Gbps; LPU v1: 235 GB/s over 11 ports), not one link
   (14 GB/s); the kv_placement_s docstring's "over one link" is wrong wording. This only makes "stream" look better; it
   already never pays. The optimistic set mixes LPU v1 latency (300 ns) with Groq 3 bandwidth; 722 ns is a per-hop figure
   from a pipelined all-reduce, not point-to-point.
5. Spare SRAM on the weight chips is pooled (166 MB over 3 chips); per chip ~55 MB, so per-loop B=1 has one more remote
   layer than counted (~0.7 µs per step, < 1 %).
6. Writing the new token's K/V to remote chips is not charged (≤ ~1 µs per step at B=16).
7. A per-loop B=16 layer's KV (537 MB) spans 2 KV chips. If split by heads (as I assume) no softmax combine is needed;
   split by sequence it would be. Whole-layer packing of shared B=16 "near" KV would need 11 chips instead of 10;
   splitting a layer's KV by heads across KV chips keeps 10.
Nits: an invalid kv_placement only raises when r > 0; no tests for kv_remote_fraction / kv_placement_s.

## Pre-registration: fixes F1-F3 (committed before the code change and before re-running the table)
- F1 (model): new optional `seqs` argument in step_time / step_terms (default = B, so defaults unchanged); the remote
  fraction uses `seqs`; `chip_token_time_loopspec` passes seqs = 1. Stream bytes stay lanes x C (each lane reads its
  context, as the existing KV term does).
- F2 (model): with kv_on_chip and kv_placement "near"/"stream", pipeline hops = weight chips only (the fewest chips
  holding weights + LM head).
- F3 (script): co-located chip count = chips needed to pack whole layers (each with its KV) plus the LM head.
- Tests: add checks for kv_remote_fraction, the seqs argument and the near/stream hop count to tests/test_loopcore_model.py.
- Predicted effects (from the check's estimates): F1 raises row (c) for shared stream ~2.4x (optimistic links), shared
  near ~+7 %, per-loop stream ~+13 %, per-loop near ~+1 %; rows (b) and B=16 unchanged. F2 raises near/stream: per-loop
  B=16 +4 % (optimistic) to +10 % (pessimistic, 1 µs), shared B=16 +1-3 %, B=1 +0.5-1.4 %. F3: shared co-located B=16
  chips 10 → 12, per chip / 1.2 (1.30-2.28 → ~1.07-1.89), speedup −< 1 %. Default reports must stay byte-identical.
- Consequences flagged for Sid, not edited: decision 3 (72eeeaf) says shared B=16 "10 chips" (becomes 12) and quotes the
  separate-KV-unit cost from near rows that F1/F2 change.

## R1 - fixes F1-F3 applied, results (table re-run: results/kv_placement_p1.txt replaced; model only)
Checks: `--loopspec-table`, `--kv-levers` and the default report byte-identical before/after; tests/test_loopcore_model.py
31 passed (+1 pre-existing xfail); the 3 new tests fail on the pre-fix model (checked) and pass after.
Against the predictions (optimistic links unless stated):
- F1 row (c): shared stream 3.7-6.2x → 8.3-15.1x (~2.4x, predicted ~2.4x); shared near 12.0-24.0 → 12.6-25.6x (+5-7 %,
  pred. ~+7 %); per-loop stream 3.2-5.3 → 3.7-6.1x (+13-15 %, pred. ~+13 %); per-loop near +1-2 % (pred. ~+1 %). Matched.
- F2: per-loop B=16 near 12.3-20.6 → 12.7-21.4x (+3-4 %), pessimistic 10.6-17.0 → 11.4-18.7x (+8-10 %); shared B=16 near
  +1 % / +3 %; B=1 ≤ 1 %. Matched.
- F3: shared co-located B=16 chips 10 → 12, per chip 1.30-2.28 → 1.08-1.90 (predicted ~1.07-1.89). Matched.

Corrected table, sourced links (optimistic .. pessimistic), t_L 3.83-6.68 µs x eff 1.0-1.3:

| design | chips 1x / 16x | (b) B=1 | /chip | (c) B=1 | /chip | B=16 | /chip |
|---|---|---|---|---|---|---|---|
| per-loop, HBM (as trained) | 3 / 3 | 3.8-7.1x | 1.28-2.37 | 5.8-9.9x | 1.92-3.31 | 3.1-4.3x | 1.03-1.42 |
| per-loop, on-chip, co-located | 5 / – | 5.1-11.0x | 1.02-2.20 | 12.6-26.2x | 2.52-5.23 | does not fit | – |
| per-loop, on-chip, near | 5 / 29 | 4.4-9.9x | 0.88-1.98 | 10.9-23.9x | 2.19-4.78 | 11.4-21.4x | 0.39-0.74 |
| per-loop, on-chip, stream | 5 / 29 | 1.0-5.2x | 0.21-1.04 | 0.9-6.1x | 0.17-1.21 | 0.3-1.9x | 0.01-0.07 |
| shared, on-chip, co-located | 4 / 12 | 5.1-11.0x | 1.27-2.76 | 12.6-26.2x | 3.16-6.56 | 13.0-22.8x | 1.08-1.90 |
| shared, on-chip, near | 4 / 10 | 4.9-10.8x | 1.23-2.69 | 12.2-25.6x | 3.05-6.41 | 11.5-21.5x | 1.15-2.15 |
| shared, on-chip, stream | 4 / 10 | 2.8-8.9x | 0.69-2.21 | 3.1-15.1x | 0.78-3.78 | 0.3-2.0x | 0.03-0.20 |

Separate-KV-unit ("near") cost vs co-located at B=1, sourced links: per-loop 6-20 % (b), 6-19 % (c); shared 1-6 % (b),
1-5 % (c). (1 µs hops: per-loop 18-26 %, shared 5-8 %.)

Reading changes:
- Not predicted: at 16x2048 shared "near" (10 chips) beats shared co-located (12 chips) per chip, 1.15-2.15 vs 1.08-1.90,
  at slightly lower speedup (11.5-21.5x vs 13.0-22.8x). Per-loop stays the worst per chip (0.39-0.74).
- Streaming still never pays: best case shared row (c) 15.1x vs co-located 26.2x; at B=16 ≤ 2.0x.
- **Decision 3 (72eeeaf) now has two wrong figures** (not edited; flagged to Sid): B=16 shared co-located is 12 chips,
  not 10; the separate-KV-unit cost is per-loop 6-20 % / shared 1-6 % (row b) and 6-19 % / 1-5 % (row c), not
  6-22 % / 1-6 % and 6-22 % / 6-20 %.
