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

# W1 - results: shared-cache KV window (model option `Hardware.kv_window`; proposed numbers, defaults unchanged)
Code: `kv_footprint` adds B x min(C, kv_window) x (loops - 1) tokens per layer when kv_caches = 1; kv_decode_only.py
`--window 64` reruns the placement table with it (results/kv_placement_p1_w64.txt), `--window-chips` prints the counts.
Checks: default report, `--loopspec-table`, `--kv-levers` and `python kv_decode_only.py` byte-identical before/after;
tests/test_loopcore_model.py 32 passed (+1 pre-existing xfail); the new test fails on the pre-change model (checked).

| shared KV, fp8 | KV MB without -> with window | chips, sized (near/stream) | chips, co-located |
|---|---|---|---|
| 1x2048 | 201 -> 220 (+9.4 %) | 4 -> 4 | 4 -> 4 |
| 16x2048 | 3,221 -> 3,523 (+9.4 %) | 10 -> 10 | 12 -> 12 |
| 1x8192 | 805 -> 824 (+2.3 %) | 5 -> 5 | 5 -> 5 |
| 16x8192 | 12,885 -> 13,187 (+2.3 %) | 29 -> 30 | does not fit either way |

Against the predictions: KV +9.4 % / +2.3 %, chip counts unchanged at 2048, 16x8192 29 -> 30: **all matched**.
Speedups (diff of the two placement tables): co-located rows identical; B=16 rows identical; shared near B=1 −0-1 %
(remote fraction 0.18 -> 0.25 at 1x2048; predicted −< 1 %: matched). **Not predicted:** shared *stream* at B=1 loses
6-8 % (row b) and 14-22 % (row c), because the extra remote KV crosses the links every step. Stream was already
never the best design, so no conclusion changes. Not modelled: reading/writing the window (≤ 3 % of the KV term at 2048).

# C1 - sources (opened 2026-11-24 by three parallel readers; quotes checked by them in the opened page or PDF)
Labels: **Q** quoted from an opened source; **D** derived (my or the reader's arithmetic on Q numbers); **U** search
snippet or press relay whose page could not be opened (unverified); **M** from memory, unchecked.
Fetch failures: WikiChip (refused), web.archive.org via WebFetch (blocked; curl copies used), docs.graphcore.ai (TLS;
archive.org), AnandTech (down), Tom's Hardware bodies (navigation only), SemiAnalysis (paywall), IEEE Xplore (paywall).

## Chips
| chip | node | die mm² | SRAM | power | compute | label / source (URL, location) |
|---|---|---|---|---|---|---|
| NVIDIA H100 SXM | TSMC 4N | 814 | 50 MB L2 + 33 MB RF | 700 W | 1,979 TF FP8 dense | Q: developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth Table 3 ("814 mm2", "4N", "700 Watts"); nvidia.com/en-us/data-center/h100 ("3,958 teraFLOPS" with sparsity) |
| H100 memory | – | – | 80 GB HBM3, 5 stacks, 3.35 TB/s | – | – | Q: same blog ("5 HBM3 stacks, 10 512-bit memory controllers"); product page |
| NVIDIA Groq 3 LPU (LP30) | Samsung 4 nm (press only) | **not published** | 500 MB, 150 TB/s | **not published** | 1.2 PF FP8 | Q: developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-... ("500 MB of high-speed on-chip SRAM", "150 TB/s", "96 C2C links running at 112 Gbps each", Table 2: 19.2 PF/16 chips); Q: tomshardware.com .../nvidias-20-billion-groq-deal-produces-its-first-chip ("manufactured by Samsung on a 4nm process"). U: an analyst "600 W" per chip (not used) |
| Groq TSP v1 / GroqChip | 14 nm (GF, U) | 725 (25 x 29) | 220 MiB = 230 MB | TDP 215 W, max 300 W | 750 TOPS int8 | Q: Abts et al. ISCA 2020 abstract and §II (archive.org copy of groq.com PDF); GroqChip product brief v1.5 ("Max: 300W; TDP: 215W; Average: 185W") |
| Graphcore GC200 | TSMC N7 | 823 | 900 MB | 300 W | ~250 TF FP16 (D: 0.999 PF / 4) | Q: theregister.com/2020/07/16/graphcore_colossus_mk2 ("823mm2 die", "900MB"); eenewseurope.com graphcore-details-power-figures ("the chip consumes 300W"); IPU-M2000 datasheet Table 2.4 |
| Graphcore Bow | TSMC N7 + power wafer | n/f | 900 MB | ~310 W avg (D, system/4) | ~349 TF FP16 (D) | Q: spectrum.ieee.org/graphcore-ai-processor; Bow-2000 datasheet Tables 2.4/2.7 |
| Cerebras WSE-2 | TSMC 7 nm | 46,225 | 40 GB | CS-2 23 kW max | – | Q: cerebras press release "2.5 trillion transistor mark"; cerebras.ai blog; techinsights.com/ja/node/49220 |
| Cerebras WSE-3 | TSMC 5 nm | 46,225 | 44 GB | ~15 kW wafer, ~23 kW system | 125 PF (precision n/s) | Q: cerebras press release "third generation wafer scale engine"; cerebras.ai/chip; theregister.com/a/5289286 |
| Tenstorrent Wormhole | GF 12 nm | 670 (analyst) | 120 MB | 150 W chip; n150 160 W board | 292 TF FP8 | Q: SemiAnalysis "Tenstorrent Wormhole analysis"; github tenstorrent.github.io core/aibs/wormhole; HPC Asia 2025 slide 5 |
| Tenstorrent Blackhole | 6 nm | n/f | 180 MB (120 cores) | 300 W board | 774 TF FP8 | Q: tenstorrent.github.io core/aibs/blackhole; HPC Asia 2025 slide 5 |
| d-Matrix Corsair | TSMC 6 nm | n/f | 2 GB / 8 chiplets | 600 W card | 2,400 TF MXINT8 | Q: Bhoja, Hot Chips 2025, slide 6 |
| SambaNova SN40L | TSMC 5 nm | < 650 per die, 2 dies | 520 MB | n/f (rack 10 kW / 16, U) | 638 TF BF16 | Q: arXiv 2405.07518v2 Table II |

Chip-level SRAM per area (D): Groq TSP v1 0.32 MB/mm² (14 nm); GC200 1.09 (N7); WSE-2 0.87 (7 nm); WSE-3 0.95 (5 nm);
SN40L > 0.40 (5 nm, with HBM PHYs); Wormhole 0.18 (12 nm). n/f = not found.

## SRAM density
| node | value | level | label / source |
|---|---|---|---|
| TSMC N5 / N3E | 0.021 µm² HD 6T | bitcell | Q: tomshardware.com/news/no-sram-scaling-implies-on-more-expensive-cpus-and-gpus ¶2 |
| TSMC N3B | 0.0199 µm² | bitcell | Q: same |
| TSMC N2 | 38.1 Mb/mm² (4.76 MB/mm²) macro, 0.021 µm² cell | macro | Q: spectrum.ieee.org/sram-intel-tsmc; marklapedus.substack.com "Intel, TSMC tout SRAM breakthroughs" (ISSCC 2025) |
| TSMC N3 macro | ~4.3 MB/mm² | macro | D: 38.1 / 1.10-1.12 (the N2 paper's stated gain) |
| N7 AMD 3D V-Cache | 64 MB on 41 mm² (gen 1), 36 mm² (gen 2) → 1.56-1.78 MB/mm² | chip | Q: techpowerup.com/292256; tomshardware.com amd-shares-new-second-gen-3d-v-cache |
| N4P Zen 5 L3 | 32 MiB on ~16 mm² → 2.1 MB/mm² | chip (die-shot estimate) | Q: pcgameshardware.de Zen-5 die-shot analysis |
| TSMC N7 | 0.027 µm² | bitcell | U: IEDM 2016 title from search |
| Samsung 4 nm | not found | – | LP30's node; TSMC N4/N5 used as its proxy (M: similar HD cell) |

## Cost and energy
| quantity | value | label / source |
|---|---|---|
| wafer N5 / N4 / N3 | $16,988 (CSET 2020, model) ; $17.5k (P5 16k-P95 19k) / $18.5k (16-20k) / $20k (Epoch) ; $17k 4NP (Epoch B200) | Q: cset.georgetown.edu AI Chips report App. D Table 9; epoch.ai/data/ai-chip-components-documentation/methodology; epoch.ai/data-insights/b200-cost-breakdown |
| defect density N5 | 0.10-0.11 /cm² (2020) | U: AnandTech relay of a TSMC symposium slide (site down) |
| H100 total cost | $3,320 | U-grade: Raymond James via The Decoder relay, no method; not used as an input, only as a cross-check |
| HBM price | $14-17/GB HBM3E (Epoch); ~$12/GB HBM3 (U) | Q: Epoch B200 page |
| HBM + packaging share of H100 cost | "~50%+" | Q: SemiAnalysis "The Memory Wall" (2024) |
| CoWoS-S | $12k per interposer wafer | Q: Epoch methodology |
| B200 cost template | logic $900, HBM $2,900, CoWoS-L $1,100, packaging yield loss $1,000, aux $480 | Q: Epoch B200 page (its total does not equal its rows: $6,250 vs $6,380) |
| SRAM read energy, 7 nm | 7.5-14 pJ per 64 bit (8 KB-1 MB) → 0.12-0.22 pJ/bit | Q: Jouppi et al. ISCA 2021 (TPUv4i) Table 2 (gwern.net copy) |
| int8 mult / add, 7 nm | 0.07 / 0.007 pJ; no FP8 figure found | Q: same Table 2 |
| 112G SerDes energy, 5 nm | 4.5 pJ/b (Marvell ISSCC 2022), 4.63 pJ/b (ISSCC 2023) | U: abstract / title only |
| 112G SerDes area | 0.28-0.63 mm² per lane | U: conflicting snippets; weakest area input |
| HBM2 access energy | 3.97 pJ/bit | Q: O'Connor et al. MICRO-50 2017 (NVIDIA research PDF) |

Not found anywhere reachable: LP30 die area and power; any die-area breakdown (SRAM / compute / IO) for these chips
except Groq v1's "ICU < 3 %"; tensor-core share of the H100 die; HBM3 stack die area (so the "H100 incl. HBM silicon"
area sensitivity promised in the plan is not done: no sourced number).
**Wafer/die cost estimates are credible enough to use as a range** (independent wafer-price estimates agree within
±15 %); published H100 totals are not (no method), so H100 cost is rebuilt bottom-up with the same model as LoopCore.

# C2 - one LoopCore chip: area, power, cost proxy (`python3 chip_cost.py`; every input labelled at its top)
| quantity | range | built from |
|---|---|---|
| area | **310-753 mm²** (0.38-0.93x H100) | SRAM 500 MB at 3.8 .. 1.56 MB/mm² = 132-321 mm² (D, from C1 density rows) + FP8 datapath 123-247 mm² (D: H100 1,979 TF at 25-50 % of its die, the share is M) + 96 SerDes lanes 27-60 mm² (U) + 10-20 % other (M). HBM design + 15-30 mm² PHY (M). |
| power, full load | **310-576 W** (0.44-0.82x H100's 700 W) | SRAM reads 150 TB/s x 0.12-0.22 pJ/bit = 144-264 W (Q, 7 nm, so pessimistic at 4 nm) + MACs 0.6e15/s x 0.077 pJ = 46 W (Q/D, int8 as FP8 proxy) + SerDes 96 x 112 Gb/s x 4.5-4.63 pJ/b = 48-50 W (U) , all x 1.3-1.6 overhead (M). HBM design + 21-32 W per stack (D/U). |
| cost proxy (manufacturing, not price) | LoopCore **$225-1,049**; + 1 HBM stack and CoWoS **$993-2,189**; H100 rebuilt the same way **$2,193-3,603** | Murphy-yield die cost ($16-18.5k wafer, D0 0.07-0.10, Q/U) + package $100-300 (M) or 16 GB HBM at $12-17/GB + CoWoS $600-900 (Q/D); H100 = its die + 80 GB HBM + CoWoS; x 1.10-1.25 assembly loss (D, Epoch). Ratio over input corners with shared inputs equal for both chips: **0.08-0.38**, HBM design **0.39-0.67**. The H100 total brackets Raymond James's $3,320 (not an input). |

Checks:
- **Calibration (same area method on Graphcore GC200, N7, 900 MB, ~250 TF FP16): predicted 508-914 mm², actual 823.**
  Brackets it. Groq TSP v1 could not be checked: no 14 nm SRAM density from an opened source.
- Chip-level SRAM per area of the estimate: 0.66-1.61 MB/mm², vs WSE-3 0.95, WSE-2 0.87, GC200 1.09 (C1, D). Brackets
  Cerebras, but the high end (310 mm²) is 1.7x Cerebras and denser than any shipping SRAM-heavy chip: the low-area end
  is optimistic. This end only affects upper bounds; every "above 1" verdict below uses the 753 mm² end.
- Power: GroqChip v1 TDP 215 W (14 nm, 725 mm²), GC200 300 W, Blackhole board 300 W, d-Matrix card 600 W / 8 chiplets
  (C1, Q). 310-576 W for an LP30-class chip (150 TB/s SRAM, 1.2 PF FP8) sits at or above these. LP30's own power
  is not published.
- Referee (fresh subagent, given only chip_cost.py and the table): recomputed parsing, every formula, the cost corners
  and the GC200 check by hand: no errors; "above 1" verdicts not too generous. One misleading comment (SRAM range read
  as 2.0-3.8, code applies 1.56-3.8) fixed; numbers unchanged.

# C3 - results (results/chip_cost_table.md; proposed numbers, write-up unchanged)
Speedup per chip / per mm² / per watt / per cost vs one H100 (sourced links, t_L 3.83-6.68 µs, eff 1.0-1.3; low end
uses the highest ratio). Verdict: ✔ above 1 over the whole range, ~ straddles 1, ✘ below 1.

| design (chips 1x / 16x) | per chip (b / c / B=16) | per mm² (b / c / B=16) | per watt (b / c / B=16) | per cost (b / c / B=16) |
|---|---|---|---|---|
| per-loop, HBM (3 / 3) | 1.27-2.37 ✔ / 1.93-3.30 ✔ / 1.03-1.43 ✔ | 1.32-5.93 ✔ / 2.01-8.26 ✔ / 1.07-3.59 ✔ | 1.46-5.00 ✔ / 2.23-6.98 ✔ / 1.19-3.03 ✔ | 1.88-6.04 ✔ / 2.87-8.42 ✔ / 1.53-3.66 ✔ |
| per-loop, co-located (5 / –) | 1.02-2.20 ✔ / 2.52-5.24 ✔ / – | 1.10-5.78 ✔ / 2.72-13.76 ✔ / – | 1.24-4.97 ✔ / 3.06-11.83 ✔ / – | 2.65-29.0 ✔ / 6.56-69.2 ✔ / – |
| per-loop, near (5 / 29) | 0.88-1.98 ~ / 2.18-4.78 ✔ / 0.39-0.74 ✘ | 0.95-5.20 ~ / 2.36-12.6 ✔ / 0.42-1.94 ~ | 1.07-4.47 ✔ / 2.65-10.8 ✔ / 0.48-1.67 ~ | 2.29-26.1 ✔ / 5.67-63.1 ✔ / 1.02-9.74 ✔ |
| per-loop, stream (5 / 29) | 0.20-1.04 ~ / 0.18-1.22 ~ / 0.01-0.07 ✘ | 0.22-2.73 ~ / 0.19-3.20 ~ / 0.01-0.17 ✘ | 0.24-2.35 ~ / 0.22-2.75 ~ / 0.01-0.15 ✘ | 0.52-13.7 ~ / 0.47-16.1 ~ / 0.03-0.86 ✘ |
| shared, co-located (4 / 12) | 1.27-2.75 ✔ / 3.15-6.55 ✔ / 1.08-1.90 ✔ | 1.38-7.22 ✔ / 3.40-17.2 ✔ / 1.17-4.99 ✔ | 1.55-6.21 ✔ / 3.83-14.8 ✔ / 1.32-4.29 ✔ | 3.32-36.3 ✔ / 8.19-86.4 ✔ / 2.82-25.1 ✔ |
| shared, near (4 / 10) | 1.23-2.70 ✔ / 3.05-6.40 ✔ / 1.15-2.15 ✔ | 1.32-7.09 ✔ / 3.30-16.8 ✔ / 1.24-5.64 ✔ | 1.49-6.09 ✔ / 3.71-14.4 ✔ / 1.40-4.85 ✔ | 3.19-35.6 ✔ / 7.93-84.5 ✔ / 2.99-28.4 ✔ |
| shared, stream (4 / 10) | 0.70-2.23 ~ / 0.78-3.77 ~ / 0.03-0.20 ✘ | 0.76-5.84 ~ / 0.84-9.91 ~ / 0.03-0.53 ✘ | 0.85-5.02 ~ / 0.94-8.52 ~ / 0.04-0.45 ✘ | 1.82-29.4 ✔ / 2.02-49.8 ✔ / 0.08-2.64 ~ |

Sensitivity, if the LoopCore die were reticle-sized (858 mm², M; ratio 1.05): three per-mm² "above 1" verdicts become
"straddles": per-loop HBM at B=16 (low 0.95), per-loop co-located (b) (0.97); per-loop near (b) falls further (0.83).
Every other per-mm² verdict holds.

## Against the pre-registration (7a9b14e)
- Priors: area 500-900 mm² → estimate 310-753 (smaller); power 250-500 W → 310-576 W (higher). So per mm² came out more
  generous and per watt less generous than I predicted.
- Per mm², predicted ✔: (c) for co-located, near and per-loop HBM; (b) shared co-located and near: **all matched.**
  Predicted ~ for (b) per-loop HBM and per-loop co-located: **refuted, both ✔** (low ends 1.32, 1.10). Predicted ~ for
  B=16 shared co-located, shared near, per-loop HBM: **refuted, all ✔** (1.17, 1.24, 1.07). Predicted ✘ for per-loop
  near B=16: **refuted, ~** (0.42-1.94). Stream B=16 ✘: **matched.**
- Per watt, predicted ✔ for the per-mm² set plus (b) per-loop co-located / near / HBM and B=16 shared: **matched**
  (per-loop near (b) low end 1.07). Per-loop HBM B=16 predicted ~: **refuted, ✔** (1.19). Per-loop near B=16 ~ and
  stream B=16 ✘: **matched.**
- Ordering unchanged by any normalisation: **matched** (same chip in every design; the HBM design's extra area/power
  is 2-6 %).
- Calibration "within a factor 1.5 of Groq TSP v1 and Cerebras": **partly**: brackets GC200's area and Cerebras's
  MB/mm², but the dense end is 1.7x Cerebras; TSP v1 not checkable.

## Verdict (plain language)
1. **Survives everything (per chip, per mm², per watt, per cost, at both batch sizes):** one shared KV cache on-chip,
   co-located or near. The trained-in sharing from Week 6b remains the design that wins on every axis.
2. **Survives at batch 1, not at batch 16:** per-loop caches on-chip. Co-located (5 chips) beats the H100 per mm², watt
   and cost at batch 1. At batch 16 it needs 29 chips and only "near" works; it is below the H100 per chip, straddles
   per mm² and per watt, and is above 1 only per manufacturing cost (1.02 at the low end, before boards and links for
   29 chips, so not a claim to make).
3. **Survives but with the smallest margin:** per-loop KV in HBM. It is above 1 everywhere, but at batch 16 its low ends
   are 1.03-1.19 except per cost; a reticle-sized die would push it under 1 per mm².
4. **Fails:** streaming KV over chip links. It straddles 1 at batch 1 and is far below 1 at batch 16 on every axis
   except, for the shared cache, per cost.
5. **What normalising changed:** "per chip" understated LoopCore. The estimated chip is smaller (0.38-0.93x), lower
   power (0.44-0.82x) and much cheaper to make (0.08-0.38x, mostly because it has no HBM) than an H100, so every per-mm²,
   per-watt and per-cost number is at or above its per-chip number. The paper's per-chip claims are conservative, not
   generous; normalising strengthens them rather than weakening them.
6. **Weakest inputs** (what a reviewer can attack): LP30's real die size and power (not published); the tensor-core
   share of the H100 die (memory); SerDes area per lane (unverified). Per watt compares TDPs: neither chip's power at
   batch-1 decode was measured. Per cost is manufacturing cost of the chips only: no boards, links, hosts, NRE or
   margin, which hurts the 10-29 chip designs most.
