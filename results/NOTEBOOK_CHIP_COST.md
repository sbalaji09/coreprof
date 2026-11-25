# Lab notebook: LoopCore area, power and cost per chip (append-only)

Started 2026-11-24. Background: HANDOFF.md (top), results/NOTEBOOK_KV_SHARING.md (P1, R1: the per-chip table),
results/LATENCY_SOURCES.md, results/NOTEBOOK_W6B.md, loopcore_model.py (`sized`, `kv_placement`, chip constants).

# Frame
- **Question.** The write-up reports "speedup per chip" vs one H100, which assumes one LoopCore chip costs what one
  H100 costs. With silicon area, power and a cost proxy per chip taken from published sources, which design points
  in the per-chip table still beat the H100 (> 1.0) per mm², per watt and per unit cost?
- **The LoopCore chip as modelled** (loopcore_model.py, `SRAM_MB, SRAM_BW, COMPUTE_FP8, LINK_BW`): 500 MB SRAM,
  150 TB/s SRAM bandwidth, ~1.2 PFLOPS FP8, chip-to-chip links; these are the Groq 3 LPU's published figures. The
  "per-loop, HBM" design also assumes one HBM3 interface at 3.35 TB/s (`EXT_BW`, an assumption).
- **Metrics.** For each design point (the 7 rows of the R1 corrected table, at 1x2048 and 16x2048, rows b, c, B=16):
  speedup / (n_chips x A_chip / A_H100), speedup / (n_chips x P_chip / P_H100), speedup / (n_chips x cost_chip /
  cost_H100). Each as a range: the speedup range of the table times the range of the area/power/cost ratio.
- **H100 reference.** 814 mm² die, TSMC 4N, 700 W (SXM). Sourced in this notebook before use.
- **Guardrail.** Every constant has an opened source (URL + section) or is labelled "derived" or "from memory,
  unchecked". The estimate method must reproduce published chips (Groq TSP v1: 230 MB on ~725 mm² at 14 nm; Cerebras
  SRAM per area) within its own stated range before it is applied to LoopCore.
- **Success.** A verdict per design point: survives (> 1.0 over the whole range), fails (< 1.0 over the whole range),
  or undecided (the range straddles 1.0), with the assumption that decides it named.
- **Rules.** Proposed numbers only. Defaults, results and write-up numbers unchanged without Sid. No GPU spending.

# Plan (written before any source is opened)
1. Sources (web, parallel readers; each quote with URL and section):
   a. SRAM density at 7 / 5 / 4(3) nm: bitcell area and macro density (Mb/mm²) from TSMC/Samsung/Intel ISSCC or IEDM
      papers, plus chip-level MB/mm² from die analyses.
   b. Per chip: die area, SRAM, process, power for Groq TSP v1 / LPU v1 / Groq 3 LPU, Cerebras WSE-2 / WSE-3,
      Graphcore GC200 / Bow, Tenstorrent Wormhole / Blackhole, d-Matrix Corsair, SambaNova SN40L, and H100.
   c. Wafer prices by node and die/package cost estimates (H100 incl. HBM). If none is credible, say so and use area.
2. Calibration check: from (a), predict chip-level MB/mm² for the SRAM-heavy chips in (b) at their nodes; the method
   must bracket Groq TSP v1 and Cerebras.
3. LoopCore estimate, each a range with its assumptions: area = SRAM (500 MB / chip-level density) + compute
   (1.2 PFLOPS FP8, scaled from a published FP8 or int8 datapath area) + links/IO (SerDes area per lane) + HBM PHY
   for the HBM design only. Power = published TDPs of comparable chips, cross-checked against energy-per-bit/op
   arithmetic (SRAM reads at 150 TB/s, MACs, SerDes). Cost = area-based die cost (wafer price, yield model) + HBM
   stacks where used; H100 likewise incl. its 80 GB HBM.
4. Re-express the table (a script next to kv_decode_only.py, reading the same rows; no model defaults changed).
5. Verdict in plain language.

# Pre-registration: per-mm² and per-watt (committed before any source is opened or any number computed)
My priors (from memory, unchecked; the sources will replace them): an SRAM-dominated 500 MB chip at 4-5 nm is
~500-900 mm², i.e. 0.6-1.1x an H100 die; its TDP is ~250-500 W, i.e. 0.35-0.7x an H100. So per mm² ≈ per chip x
0.9-1.6 and per watt ≈ per chip x 1.4-2.8. The HBM design pays extra area (HBM PHY) and power (HBM stacks).

Per-chip ranges being normalised (R1 corrected table, sourced links, t_L 3.83-6.68 µs, eff 1.0-1.3):

| design | (b) B=1 /chip | (c) B=1 /chip | B=16 /chip |
|---|---|---|---|
| per-loop, HBM | 1.28-2.37 | 1.92-3.31 | 1.03-1.42 |
| per-loop, co-located | 1.02-2.20 | 2.52-5.23 | does not fit |
| per-loop, near | 0.88-1.98 | 2.19-4.78 | 0.39-0.74 |
| per-loop, stream | 0.21-1.04 | 0.17-1.21 | 0.01-0.07 |
| shared, co-located | 1.27-2.76 | 3.16-6.56 | 1.08-1.90 |
| shared, near | 1.23-2.69 | 3.05-6.41 | 1.15-2.15 |
| shared, stream | 0.69-2.21 | 0.78-3.78 | 0.03-0.20 |

Predictions:
- **Per mm², stays > 1.0 over the whole range:** row (c) for both co-located and both near designs and per-loop HBM;
  row (b) shared co-located and shared near. **Straddles 1.0:** row (b) per-loop HBM, per-loop co-located,
  per-loop near; B=16 shared co-located, shared near and per-loop HBM (the HBM PHY pushes its low end below 1.0).
  **Below 1.0 everywhere:** per-loop near at B=16 and every stream row at B=16.
- **Per watt, stays > 1.0:** everything that survives per mm², plus row (b) per-loop co-located / near / HBM and B=16
  shared co-located / near. Per-loop HBM at B=16 straddles (HBM stack power). Per-loop near B=16 straddles at best;
  stream rows at B=16 stay below 1.0.
- **Ordering unchanged** by any normalisation: shared ≥ per-loop co-located > per-loop near ≫ stream at every point,
  because all LoopCore designs use the same chip (only the HBM design differs).
- **Calibration:** the method brackets Groq TSP v1 (~0.32 MB/mm² chip-level at 14 nm) and Cerebras (~1 MB/mm² of
  wafer at 5-7 nm, from memory, unchecked) within a factor of 1.5.
- **What would change the verdict:** if the LoopCore die is > 1,000 mm² (e.g. SRAM macro density much lower than I
  assume, or the 96 links at 112 Gbps cost > 150 mm²), the B=1 row (b) designs fall to straddling or below.

# Pre-registration: shared-cache KV window (task 2, model option; committed before the code change)
**Change.** New option `Hardware.kv_window` (tokens, default 0 = unchanged). With one shared cache (kv_caches = 1),
each of the other (total_loops - 1) loops keeps its own last `kv_window` tokens per layer per sequence
(sharefirst_w64, NOTEBOOK_W6B.md E4). It adds B x min(C, window) x (loops - 1) x KV bytes per token per loop to
`kv_footprint`. Read/write time of the window (64 keys per later loop per layer) is not added: ≤ 64 / C of the KV
term, 3 % at 2048 (stated as a known omission). Defaults and every existing report stay byte-identical.
**Predicted effect (hand arithmetic, fp8 KV 98,304 B per token per cache, int8 weights + LM head 1,334 MB):**
- Extra KV = 3 x 64 x 98,304 B = 18.9 MB per sequence: +9.4 % at 2048 (201 → 220 MB), +2.3 % at 8192.
- 1x2048 shared: 1,535 → 1,554 MB, 4 chips (unchanged). 16x2048 shared (`sized`): 4,555 → 4,857 MB, 10 chips
  (unchanged, 97 % full). 16x2048 co-located: 185.6 → 198.2 MB per layer, still 2 layers per chip, LM head still fits
  (104 MB spare vs 100.7 MB), 12 chips (unchanged). 16x8192 shared: 14.22 → 14.52 GB, 29 → 30 chips (+1).
- Speedups: unchanged at the table's resolution where the chip count is unchanged; −< 1 % where "near" gains a remote
  layer.
