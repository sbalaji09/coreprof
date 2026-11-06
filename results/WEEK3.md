# Week 3: Continuous Depth Batching (CDB) vs. baseline batcher

**Question.** Ouro-1.4B lets each token stop looping early (exit gate, threshold *q*). Does
Continuous Depth Batching, which refills a slot the moment its token exits instead of
waiting for the slowest token in the batch, turn those early exits into real throughput?
And how close does it get to the ideal?

**Answer in one line.** Yes. CDB is up to **1.47× faster** than the baseline batcher at q = 0.3
(L40S) and **1.27×** at q = 0.5 (A6000), with a **< 1 % difference at q = 1.0** (no early exit),
where it should show none. The speedup tracks the reduction in batch calls almost exactly
while the GPU is in its flat-latency regime, and falls behind it once the batch is large
enough that step time starts to grow with batch size.

## Setup

- Model: ByteDance/Ouro-1.4B (rev 574fa66), bf16, 4 recurrent steps, Q-exit rule
  (exit at the first loop whose cumulative exit probability ≥ q; see DESIGN_CDB.md).
- Workload: GSM8K test questions (fixed order), greedy decoding, 128 new tokens,
  `slots × 8` requests per run so each slot is refilled many times. Throughput is measured
  over a **steady-state window** that excludes the end-of-run tail; full-run numbers agree.
- Both batchers share the same engine, KV store and exit rule (`batching/`); they differ only
  in which rows go into each call. fp32 equivalence (Week 2) confirmed identical tokens and
  exit loops for baseline, CDB and a one-sequence-at-a-time reference.
- Metric: **useful row-loops per second** (one row × one loop iteration that the token actually
  needed). This normalizes for bf16 runs generating slightly different text. Each point is
  the median of 3 repeats (2 for the controls), with min–max in brackets.
- Hardware: two GPUs, never mixed within a comparison.
  - **RTX A6000** (NRP Nautilus, 4 CPU cores): q = 1.0 and q = 0.5, slots 1–64.
    `results/bench_cdb_a6000-c3_*`
  - **L40S** (Modal, 8 CPU cores; Modal does not report the CPU model): q = 0.3 for slots
    2–64, plus a q = 1.0 control at 4, 16 and 64 slots.
    `results/modal/bench_q03-l40s-20261105-083950/`
    The NRP JupyterHub stopped issuing GPU pods, so the last threshold moved to Modal
    (see "Caveats").

## Results

### A6000, q = 0.5 (mean exit depth ≈ 3.1 loops of 4)

| Slots | CDB speedup | Batch-call ratio (baseline / CDB) | % of ideal* |
|---:|---:|---:|---:|
| 1 | 1.00× [0.99–1.00] | 1.00 | 96 % |
| 2 | 1.06× [1.05–1.06] | 1.07 | 96 % |
| 4 | 1.12× [1.11–1.12] | 1.11 | 96 % |
| 8 | 1.18× [1.16–1.18] | 1.15 | 94 % |
| 16 | 1.23× [1.23–1.24] | 1.24 | 94 % |
| **32** | **1.27× [1.25–1.27]** | 1.26 | 95 % |
| 64 | 1.15× [1.15–1.15] | 1.28 | 95 % |

### L40S, q = 0.3 (mean exit depth ≈ 2.4 loops of 4)

| Slots | CDB speedup | Batch-call ratio | CDB / baseline tokens/s |
|---:|---:|---:|---:|
| 2 | 1.11× [1.08–1.16] | 1.11 | 60 / 54 |
| 4 | 1.24× [1.15–1.41] | 1.22 | 124 / 102 |
| 8 | 1.26× [1.18–1.55] | 1.33 | 245 / 171 |
| 16 | 1.38× [1.36–1.49] | 1.39 | 488 / 355 |
| **32** | **1.47× [1.42–1.54]** | 1.51 | 795 / 526 |
| 64 | 1.35× [1.34–1.35] | 1.58 | 1177 / 878 |

### Controls, q = 1.0 (no early exit; CDB should equal baseline)

- A6000, slots 1–64: 0.996× to 1.004×.
- L40S, slots 4 / 16 / 64: 1.006×, 1.010×, 0.998×.

The noise floor is about 1 %, so every q < 1 speedup above is a real effect.

\* *% of ideal* = measured speedup ÷ the speedup CDB would get if every call cost as much as
a full S-row call at q = 1.0 on the same GPU (`pct_ideal_full` in the CSVs). See the caveats
for why this column is reported only for the A6000.

### Answer quality (100 GSM8K questions, CDB, bf16, A6000)

| q | Accuracy | Mean loops used | Same answer as q = 1.0 |
|---:|---:|---:|---:|
| 1.0 | 78 % | 4.00 | — |
| 0.5 | 81 % | 3.22 | 86 / 100 |
| 0.3 | 75 % | 2.62 | 80 / 100 |

With 100 questions each accuracy has a standard error of about ±4 points, so the three
settings are statistically indistinguishable. Early exit saves about 35 % of loop work at
q = 0.3 without a detectable accuracy loss, although 14–20 % of individual answers change.
(q = 1.0 at 78 % is consistent with the 78.92 % reported for Ouro-1.4B on GSM8K; confirm
the figure against the Ouro paper before citing it.)

## What the numbers say

1. **More early exit → bigger CDB win.** Lowering q from 0.5 to 0.3 frees more slots for CDB
   to refill: the peak goes from 1.27× to 1.47×, and at 64 slots the baseline leaves 39 % of
   its slots idle (utilization 0.61) while CDB stays at 1.00.
2. **In the flat-latency regime, speedup ≈ call reduction.** Up to 32 slots the speedup stays
   within about 5 % of the batch-call ratio (L40S: 1.47× vs 1.51×; A6000: 1.27× vs 1.26×).
   That is what a memory/overhead-bound step predicts: a call costs about the same whether
   it carries 20 busy rows or 32, so fewer calls means proportionally less time.
3. **Past the flat region, CDB's advantage shrinks.** At 64 slots CDB still saves 1.58×
   (L40S) / 1.28× (A6000) in calls, but each call carries more busy rows and gets slower
   (L40S: 22.6 vs 18.5 ms per call), so the speedup drops to 1.35× / 1.15×. This matches
   the Week 1 latency curves: the A6000's step time starts rising between 16 and 32 rows,
   the L40S's between 32 and 64 (`results/WEEK1.md`, `results/modal/harness-l40s-*`).
   **CDB pays off most exactly where step time does not depend on batch size**, which is
   the regime LoopCore's on-chip-weight design targets.
4. **Against the theoretical limit.** If time scaled with loop work alone, early exit could
   give at most 4 / mean-exit-depth: ≈ 1.30× at q = 0.5 and ≈ 1.67× at q = 0.3. CDB reaches
   1.27 / 1.30 ≈ 97 % (A6000, 32 slots) and 1.47 / 1.67 ≈ 88 % (L40S, 32 slots), or 81 % at 64.

## Comparison with the published claims

| Claim | Source | Our result |
|---|---|---|
| Decode latency stays flat at small batch | Memory-bound decode; motivates CDB in Bae et al., *Relaxed Recursive Transformers* (2024, arXiv:2410.20672) | **Confirmed.** Per-loop step time is flat for 1–16 rows on the A6000 (16.9 → 18.2 ms) and 1–32 on the L40S (20.6 → 22.8 ms). |
| CDB turns early exits into throughput | Bae et al. 2024 (continuous depth-wise batching) | **Confirmed.** Up to 1.27× (q = 0.5) and 1.47× (q = 0.3) over the baseline, ≈ 1.00× at q = 1.0. |
| CDB reaches ≈ 99 % of the ideal speedup | Stated in our proposal's prior-findings summary; *source to be confirmed* | **Close but not reproduced.** ≈ 94–96 % of ideal on the A6000 at q = 0.5. The L40S's % of ideal is not reliable (see caveats). |

**Why we land below 99 %.**
- *Per-call host overhead.* Each batch call pays about two host–device syncs (exit flags,
  then exiting tokens) and Python scheduling. Both batchers pay this, but CDB's refill
  bookkeeping costs a little more per call. Week 1 suggests overhead dominates step time
  (≈ 17 ms measured vs ≈ 3–4 ms to stream the weights on an A6000), so small per-call
  differences show up directly in the ratio.
- *The gather copy.* CDB rows at different loop depths read their own per-loop KV through a
  gather; that cost grows with batch × context.
- *Leaving the flat regime* at large slot counts (point 3).
- *Different hardware and exit distribution* from the original measurements.

## Caveats

- **Two GPUs.** q = 0.5 / 1.0 come from an A6000 (NRP) and q = 0.3 / 1.0 from an L40S (Modal).
  Every speedup is measured within one GPU against its own baseline, so the ratios are valid.
  Absolute tokens/s should never be compared across the two tables.
- **Modal run-to-run noise is about 10–15 % per measurement** (e.g. L40S baseline at 8 slots:
  195 / 171 / 160 tok/s), versus about 1 % on the A6000. That is why the L40S rows have wide
  min–max ranges, and why the L40S "% of ideal" values come out as 114 % and 110 % at 4 and
  16 slots: those values depend on per-call times measured in separate runs, and the noise
  swamps them. They are not reported above.
- **`pct_ideal_own` is always 100 % by construction** (it equals CDB's slot utilization) and
  is not evidence of anything. Use `pct_ideal_full` (A6000) or the call-ratio comparison.
- **Do not compare tokens/s across thresholds.** Different q values generate different text
  (bf16 greedy), so `speedup_vs_noexit_tokens` can exceed the 4 / depth bound (e.g. 1.87× at
  4 slots on the L40S). Within-threshold baseline vs CDB is the valid comparison.
- **The 1-slot point was skipped on the L40S.** With one slot, CDB and baseline are identical by
  construction (A6000: 1.00×), and those runs were the slowest in the sweep.
- **The quality check is 100 questions,** not the full 1,319-question GSM8K test set.

## Files

- A6000: `results/bench_cdb_a6000-c3_{runs,points,speedup}.csv`, `.json`
- L40S: `results/modal/bench_q03-l40s-20261105-083950/` (CSVs, `log.txt`, `env.txt`, and the
  plots `*_throughput.png` and `*_pct_ideal.png`)
- Quality: `results/quality_20261030-170543.json`
- Code: `bench_cdb.py`, `batching/`, `plot_bench.py`; Modal runner: `modal_app.py`
