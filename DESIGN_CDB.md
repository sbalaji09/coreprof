# Design: baseline batcher and Continuous Depth Batching (CDB) for Ouro-1.4B

Status: **approved 2026-10-23 and implemented** in `batching/` (see NOTES.md, "Week 2").

Line numbers refer to `modeling_ouro.py` at the pinned model revision
`574fa66` (see NOTES.md). Terms used throughout:
- **Loop** *u* ∈ {1..4}: one pass through the 24 shared layers, then the
  final norm, then the exit gate. Code indexes loops 0..3.
- **Block call**: one invocation of the looped block on a set of rows.
- **Row**: one token in flight at a given loop.
- **Slot**: one sequence's place in the batch and in the KV store.

The design decisions needing your approval are collected in §8.

---

## 1. Exit gate and threshold

**What the HF code does.** `OuroModel.forward` always runs all 4 loops
(lines 592–608). It records `gate_list[t] = early_exit_gate(norm(h_t))`
(a `Linear(2048, 1)`, line 513) after every loop. The exit decision is made
afterwards, in `OuroForCausalLM.forward` (lines 721–800):
λ_t = sigmoid(gate_t), p_t = λ_t·∏_{s<t}(1−λ_s), with the last loop taking
the remaining mass. Then:
- `exit_threshold` selects the first loop whose cumulative p reaches it, and
  `lm_head` is applied to that loop's hidden state.
- With the config default `early_exit_threshold = 1.0`, every token
  effectively uses loop 4.
- The HF code therefore *selects* an exit but never *skips* computation, and
  it does the CDF arithmetic in the model dtype (bf16).

**What the paper specifies** (Ouro paper, §3 "Inference with Early Exit" and
§5.4.1). It uses the Q-exit criterion, decided **per token**:

  CDF(n) = 1 − ∏_{j≤n}(1 − λ_j) for n < 4, CDF(4) = 1;
  t_exit = min{ m : CDF(m) ≥ q }.

The HF selection logic is this same rule. What we have to add ourselves is
**actually stopping**: we run loops one at a time and halt a row as soon as
CDF ≥ q.

**What we expose:**
- `exit_threshold: float = q` with default `1.0`. That matches the
  config/HF default and means "always run 4 loops". At `q ≥ 1` we skip the
  CDF entirely and force loop 4, so rounding can never cause an early exit.
- The CDF is computed in **fp32** from the gate logit (`gate.float()`) in
  every batcher and dtype, so the exit rule doesn't depend on bf16 rounding
  of probabilities.
- The output token for a row that exits at loop *k* is
  `argmax(lm_head(h_k))`, where `h_k` is the post-norm hidden state of loop
  *k* (the same tensor HF uses). Decoding is greedy, and ties go to the
  lowest index (`torch.argmax`).

**Threshold choice: measured, not guessed.** I ran a probe on 64 GSM8K
test questions plus their gold answers (7,927 answer tokens). It used the
HF full-compute forward in bf16 and measured exit loops across q.

| q | exit at loop 1 / 2 / 3 / 4 (%) | mean loops |
|---|---|---|
| 0.30 | 0.0 / 40.8 / 48.1 / 11.1 | 2.70 |
| 0.50 | 0.0 / 1.1 / 69.0 / 29.9 | 3.29 |
| 0.60 | 0.0 / 0.0 / 50.5 / 49.5 | 3.50 |
| 0.70 | 0.0 / 0.0 / 24.2 / 75.8 | 3.76 |
| ≥ 0.80 | 0 / 0 / 0 / 100 | 4.00 |

- Mean λ per loop is 0.07 / 0.19 / 0.43 / 0.48. The p90 of CDF at loop 3 is
  only 0.73, so **any q ≥ 0.8 gives no early exits at all**, and CDB would
  have nothing to gain.
- **Proposed operating points:** primary **q = 0.5** (exits mostly at loop
  3 vs 4, mean 3.29 loops) and secondary **q = 0.3** (spread across 2/3/4,
  mean 2.70). The equivalence tests run at both.
- Loop-1 exits never occur naturally. Unit tests force them with a tiny q
  such as 0.01.
- Caveats: this histogram is teacher-forced on gold answers, with real
  computed KV for every loop. The real runs log their own histogram
  (generated text, copy-forward KV), and it will differ somewhat.
- We have **not** checked answer quality at q = 0.5 or 0.3. See §9.

## 2. Slot semantics

A slot holds one sequence, and each sequence has **exactly one token in
flight** at any time, because decoding is autoregressive.

**CDB:**
- Every block call runs all occupied slots, each at its own loop.
- After the call, rows with CDF ≥ q (or at loop 4) exit. Each exiting row
  emits its token, copies its KV forward (§4), and that sequence's next token
  enters the **next** block call at loop 1.
- Non-exiting rows continue at loop u+1 in the next call.
- When a sequence emits EOS or reaches `max_new_tokens`, its slot is freed.
  The next queued request is prefilled into it (§3.4), and that request's
  first decode token joins the next block call at loop 1.

**Baseline:**
- A *decode step* processes one token for every occupied slot. It runs loop
  1 on all rows, then loop 2 on the rows that haven't exited, and so on until
  every row has exited. Exited rows wait (see decision D2).
- Only when the slowest row has exited does the next decode step start.
- Finished sequences are replaced from the queue at step boundaries
  (decision D3).

**Match with the paper.** RRT §2.4 describes the same thing: "once some
samples exit after certain looping iterations, queued requests can then be
immediately scheduled". Exits are per token ("some examples within each batch
can early-exit"). Two differences to note:
1. **RRT's numbers are simulated.** It reports "hypothetical throughput
   improvements based on an oracle-exiting approach" (§3.8) and leaves real
   deployment to future work (§5). We implement it for real, with a
   confidence-based exit (Q-exit), not an oracle.
2. **What gets scheduled into a freed position differs.** In RRT a "sample"
   frees its position after exiting. In autoregressive decode, the cheapest
   refill is **the same sequence's next token**, which is the rule you
   proposed. A *new request* only takes the slot when its sequence finishes.
   The effect is the same: no slot sits idle while work is available.

## 3. Mixed-depth forward

### 3.1 KV store (`batching/kv_store.py`)

There are two preallocated tensors, K and V, each shaped

  `[loops=4, layers=24, slots, kv_heads=16, max_len, head_dim=128]`

with `max_len = max(prompt_len) + max_new_tokens`, fixed per run.
- Memory: 0.79 MB per slot-token in bf16, 1.57 MB in fp32. For example,
  32 slots × 384 tokens is 9.7 GB in bf16.
- Per-slot host state: current length (the position of the token in
  flight), current loop, and its request.
- **Scatter:** row r with (slot s, loop u, position p) writes its new k and v
  into `K[u, l, s, :, p]` with one advanced-index assignment per layer.
- **Gather (for attention):** row r attends to `K[u, l, s, :, :p+1]`. We
  slice to `Lmax = max_r(p_r) + 1`, gather with `K[u_idx, l, s_idx, :, :Lmax]`
  into `[R, heads, Lmax, D]`, and pass a boolean mask that hides positions
  beyond each row's own length.
- This replaces HF's `UniversalTransformerCache` and its per-step `torch.cat`
  (NOTES.md §3).

### 3.2 Reusing Ouro's own layers

Ouro's layers are not reimplemented. The block call is the same loop body
the harness already times, and `selfcheck.py` verified it bit for bit:
`for layer in layers: h = layer(h, ...)`, then `norm`, then
`early_exit_gate`. Two adapters make mixed depths work:
- **Per-row positions:** `position_embeddings = rotary_emb(h,
  position_ids=[[p_r]])` with `position_ids` shaped `[R, 1]`. Ouro's rotary
  embedding already accepts a position per batch row (lines 467–489).
- **Per-row loop and slot:** a small `Cache` subclass, `MixedDepthCache`,
  is passed as `past_key_value`. Ouro calls
  `update(k, v, current_ut*24 + layer_idx)` (lines 314–322). Because loops
  are mixed within one call, we call the layers with `current_ut=0` and let
  the adapter look up each row's real (loop, slot, position) from arrays the
  scheduler sets before the call. `update` scatters the new k and v and
  returns the gathered padded K and V.
- **Masking:** the boolean mask `[R, 1, 1, Lmax]` goes in as
  `attention_mask`. SDPA applies it (`sdpa_attention_forward` slices it to
  the key length).

All three batchers (reference, baseline, CDB) use the same Ouro modules. The
only differences are which rows are in a call and how KV is stored.

### 3.3 Attention backend (decision D4)

Gather plus SDPA works in every dtype, which fp32 equivalence needs, and it
is simple. Its cost is a copy of `R × Lmax` KV per layer per call, the same
kind of overhead as HF's `torch.cat`. That cost is identical for baseline and
CDB, so the comparison stays fair, but it inflates absolute latency.

FlashAttention's `flash_attn_with_kvcache(cache_batch_idx=..., cache_seqlens=...)`
would read the store in place, with no gather and per-row lengths. It is
bf16/fp16 only and needs a wheel built for torch 2.5.1+cu124.

**Proposal:** Week 2 builds gather + SDPA behind a one-function attention
interface. Week 3 decides whether to add the flash path, validated against
gather + SDPA in bf16.

### 3.4 Prefill

Each new request is prefilled **alone (B=1)**, running all 4 loops over the
prompt with HF's own cache. Its 96 per-loop KV tensors are then copied into
the request's slot in the store.
- This follows the paper: "during the prefilling phase … all four recurrent
  steps require their own KV caches". It also makes every batcher's prefill
  bit-identical, which removes one source of divergence.
- The prompt's first output token is taken from the last prompt position
  using the same Q-exit rule on its 4 gates. All 4 loops were computed
  anyway, so this exit only selects which hidden state goes to `lm_head`,
  exactly as in HF. Its exit loop is logged separately from the decode
  histogram.
- Prefill is timed separately (§7). Week 3 can then report decode-only
  throughput without prefill distorting it.

## 4. KV for loops a token skipped (decision D1)

If token *t* exits at loop k < 4, the store has no KV at position *t* for
loops k+1..4. Later tokens running those loops will attend to that position.
Whatever we choose, **reference, baseline and CDB use the identical rule**:
it lives in one function, `kv_store.fill_skipped(slot, pos, exit_loop)`, or
its reference equivalent.

| Option | What it does | Compute saved? | At q = 1 equals real Ouro? | Paper support |
|---|---|---|---|---|
| **A. Copy forward** | `K[j, :, s, :, t] = K[k, :, s, :, t]` for j = k+1..4 | yes | **yes** (no exits, so no copies) | closest analogue: §5.4.2 decode-time KV reuse |
| B. Compute all loops for KV | keep running loops k+1..4 for that token, discarding outputs | **no**: same work as no exit, so CDB gains nothing | yes | what HF/eval effectively does; RRT §4 "parallel computation of missing KV caches" |
| C. Final-loop-only decode cache | decode positions store one KV per layer (from the exit loop), read by all loops; prompt keeps 4 caches | yes, plus ¾ less decode KV memory | **no**: differs from real Ouro even with no exits | §5.4.2 "last-step reuse": GSM8K 78.85 vs 78.92 full |

**What the papers say:**
- RRT names the missing-KV problem and leaves it open (§5: "key-value cache
  computation for exited tokens in remaining loops").
- Ouro doesn't discuss early-exit KV directly. Its §5.4.2 shows that during
  decode, deeper loops can read a single shared KV with almost no loss:
  last-step 78.85 and averaged 78.73 against 78.92 on GSM8K.
- First-step-only reuse collapses (18.73), so loop-1 KV is a poor substitute
  for deep-loop KV.

**Recommendation: A, copy forward.**
- It is the only option that both saves compute and reduces exactly to the
  real model when nothing exits. That gives a free end-to-end check: at
  q = 1, every batcher must match HF's own `generate()` token for token.
- It is in the spirit of the paper's finding that decode-time KV can be
  shared across loops.
- The copy is tiny: (4−k) × 24 layers × 2 × 16 × 128 values per exiting
  token.
- Its risk is the case the paper warns about: loop-1 exits would feed loop-1
  KV to loops 2–4 (the bad "first-step" pattern). At q ≥ 0.3 we measured
  0% loop-1 exits, and we log the histogram so this stays visible.

Option C is the natural next variant for memory savings, but it changes model
outputs even at q = 1, so I would not use it as the Week 2 baseline semantics.

## 5. Equivalence plan (`equivalence_test.py`)

**(a) Reference (`batching/reference.py`).** Each sequence is decoded alone
(B=1), one loop at a time, with Ouro's layers and HF's own
`UniversalTransformerCache` (no padding, no mask). It uses the same Q-exit
rule and copy-forward rule (A), with copies appended via `cache.update`. This
makes the reference independent of `kv_store.py`, so it catches store and
scheduling bugs instead of sharing them.
- **Check 0, reference vs HF:** at q = 1, fp32, the reference must match
  `model.generate()` token for token. This validates the reference itself.
  HF's own exit is forced to loop 4 with `early_exit_step=3`.

**(b) fp32, strict.** Setup: TF32 explicitly disabled
(`torch.backends.cuda.matmul.allow_tf32 = False`, same for cuDNN), 16
prompts, 32 new tokens, `max_slots = 4` so the queue forces refills.
- Checks 1–2: at q ∈ {0.3, 0.5}, reference, baseline and CDB must produce
  **identical tokens and identical per-token exit loops**.
- Check 3: at q = 1, all three must also equal HF `generate()`.

*Expected divergence:* in fp32 the results are not bit-identical. Padded,
masked, batched SDPA sums in a different order than unpadded B=1 (about
1e-6 relative error). A token or exit can therefore flip only when it sits
within about 1e-6 of a tie or of the threshold.
- On any mismatch the test prints the margin: the top-2 logit gap, or
  |CDF − q|.
- A mismatch with margin > 1e-4 means a real bug.
- A mismatch at margin ~1e-6 is a near-tie, which I would report to you
  rather than silently pass.

**(c) bf16, tolerance.** 128 prompts, 128 new tokens, `max_slots = 32`, at
q = 0.5 and 0.3. Reported for baseline vs reference and CDB vs reference:
- percentage of sequences that match exactly,
- mean length of the matching prefix, as a percentage of tokens,
- percentage of positions where both the token and the exit loop match,
- max |logit difference| over the *shared prefix only*, since logits after a
  divergence come from different inputs,
- exit-histogram difference.

*Expected divergence:* batch composition changes GEMM tiling and reduction
order, so bf16 rounding differs by row count (about 1e-2 on logits).
- Near-ties in argmax, or CDF values near q, will flip. With greedy decoding
  and copy-forward, a flip changes everything after it in that sequence, so
  exact-match percentages will be noticeably below 100%.
- Baseline vs CDB should look like either one vs the reference, because the
  math is the same and only the batch composition differs.
- This check reports numbers, with no pass threshold. Per-token logits are
  recorded only when the test asks for them (about 3 GB fp32 on GPU for this
  workload).

**Runtime:** the B=1 reference over 128 × 128 tokens takes about 15 minutes
in bf16. The fp32 checks take a few minutes.

## 6. Workload

- **Prompts:** GSM8K `main` test split, first 128 questions, formatted as
  `"Question: {q}\nAnswer:"` (zero-shot). The dataset is pinned to revision
  `740312a` and cached under `$HF_HOME` (/tmp/hf), so it re-downloads after a
  pod restart. The prompts are tokenized once and saved to a JSON file in the
  run's output directory.
- **Decoding:** greedy, `max_new_tokens = 128` (32 for the fp32 checks).
  Stop on EOS (id 0) or at the limit.
- **Threshold:** q = 0.5 primary, 0.3 secondary (§1). The run logs the
  exit-loop histogram and **warns loudly if more than 95% of tokens exit at
  loop 4**.
- **Seed:** fixed. Nothing is sampled, but the seed is set anyway.
- **Variation:** prompt lengths and per-token exit loops differ, so in CDB
  sequences finish at different times and refills happen naturally. A base
  model rarely emits EOS on this format, so most sequences end at
  `max_new_tokens`.

## 7. Measurement hooks for Week 3 (not the benchmark)

**Per block call** (`StepRecord`): `call_idx`; `decode_step` (baseline
only); `n_rows`; `max_slots`; rows per loop depth `[n_u1..n_u4]`; `n_exits`;
`n_refills`; `n_finished`.
- All of these come from data the host already has after the exit decision,
  so recording them adds no syncs.
- Optional per-call CUDA-event timing (`time_calls=True`) is off by default.

**Per run** (`RunStats`):
- tokens generated and sequences finished,
- total block calls and total row-loops, plus decode steps for the baseline,
- slot utilization = row-loops / (calls × max_slots),
- decode wall time (synchronized `perf_counter` at both ends) and prefill
  wall time, separately,
- exit-loop histogram (decode tokens; prefill tokens separately).

**Host-device syncs:**
- **Forced, once per block call:** the exit flags (R bools) must reach the
  host to decide which rows leave, which run `lm_head`, and which slots
  refill. Both batchers pay it equally. The baseline also needs it to know
  when the step ends.
- **A second small sync per call:** argmax token ids for exiting rows, needed
  for the EOS check and the next input.
- Week 3 can fuse this with the next call's flag transfer (a lagged EOS
  check). Week 2 keeps both for clarity, identically in both batchers.
- `lm_head` runs only on exiting rows. Running it on every row would add
  about 200 MB of weight reads to each call.

## 8. Decisions for you to approve

- **D1. Skipped-loop KV = copy forward (option A)**, in all batchers (§4).
- **D2. Baseline exited rows are dropped from later block calls in the same
  step** (compacted), not kept computing garbage. Baseline and CDB then do
  the same total row-loops, and CDB's gain comes purely from needing fewer
  block calls (better slot utilization).
  - The alternative, keeping exited rows in the call, is closer to a naive
    static batch. It adds wasted attention and KV reads that CDB doesn't
    have.
  - I can add it as a flag if you want both.
- **D3. Baseline refills finished slots at decode-step boundaries**
  (request-level continuous batching), not "wait for the whole batch, then
  load the next batch".
  - This isolates *depth* batching as the only difference from CDB.
  - Static batching would mix request-level and depth-level gains. It could
    be a `refill="batch"` option.
- **D4. Attention: gather + SDPA now**, flash-attn paged path deferred to
  Week 3 (§3.3).
- **D5. Thresholds:** q = 0.5 primary, 0.3 secondary; API default 1.0 (§1).
- **D6. Prefill alone (B=1) per request, outside block calls**, timed
  separately (§3.4).

## 9. Risks that could make Week 3 misleading (known now)

- **Answer quality at low q is unmeasured.** A threshold that produces
  spread but garbage text would make a hollow speedup. A cheap check is
  GSM8K accuracy on the 128 outputs at q = 0.3 / 0.5 / 1.0, scored from
  the outputs we already produce.
- **Launch-bound regime.** At small B the block call is CPU-launch-bound
  (README sanity check). CDB's per-call Python bookkeeping is slightly
  heavier than the baseline's, and that overhead counts against CDB.
- **The gather copy** (D4) adds per-call cost that grows with B × context and
  dilutes both batchers' differences at large B.
- **Prefill interference:** refills in CDB happen mid-stream. If prefill
  time isn't separated (it is, §7), CDB looks worse at high churn.
- **Hardware:** results are on an A6000, not the A10 target.
