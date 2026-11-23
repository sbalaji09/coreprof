# Lab notebook: LoopCore per-layer latency

Append-only. Started 2026-11-22.

# Frame
- **Question.** What per-layer latency t_L (the fixed time per layer per loop step for everything except streaming
  weights and KV: norms, rotary, activations, softmax, cache write, stage hand-offs) should `loopcore_model.py` use for a
  fused dataflow chip at B=1 (and B=16), with what uncertainty? Today it is a swept assumption (`LAYER_LATENCY`).
- **Why it matters.** With the KV cache on-chip, t_L is 84–88 % of a B=1 token (CALIBRATION.md, Week 6 plan).
  ≥ 10× vs H100 int8 at B=1, C=2048 needs t_L < ~6 µs; ≥ 3× needs < ~21 µs (HANDOFF.md, `max_layer_latency`).
- **Metric.** An estimate of t_L in µs with a low/central/high range, each part traceable to a source or a count.
- **Guardrail.** The method must reproduce the GPU's measured per-layer non-weight time before it is applied to the chip.
- **Success.** A range narrow enough to say which side of 6 µs and 21 µs the chip lands on, or an honest statement that
  it cannot be decided without specific design details (and which).

# Prior work (timeboxed; sources opened 2026-11-22)
- Published end-to-end decode speed on SRAM-based systems, converted to µs per layer (time per token ÷ layers). These are
  per-user serving speeds; they include everything (KV attention, sampling, multi-chip hops, any serving batch), so they
  are realistic upper anchors, not minimum-latency limits:
  - Cerebras, Llama-3.1-8B, 1,800 tok/s (Aug 2024 press release; Artificial Analysis quoted), 32 layers → **17.4 µs/layer**.
    https://cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference
  - Cerebras, Llama-3.1-70B, 450 tok/s, 80 layers → **27.8 µs/layer** (same source).
  - Groq, Llama-2-70B, 241 tok/s (Artificial Analysis) to 300 tok/s (Groq claim), reportedly 576 chips, 80 layers →
    **41.7–51.9 µs/layer**. https://groq.com/groq-lpu-inference-engine-crushes-first-public-llm-benchmark/ ,
    https://news.ycombinator.com/item?id=38739199 (chip count uncertain).
  - Cerebras gpt-oss-120B, ~1,730 tok/s (Artificial Analysis via secondary source) to ~3,000 (vendor), 36 layers →
    **9.3–16 µs/layer** (MoE; secondary sources only: https://www.spheron.network/blog/groq-lpu-vs-cerebras-wse-cheapest-inference-2026/).
- Groq TSP (ISC 2020 ML HW workshop slides, https://mlhardware.github.io/2020/groq.pdf, text extracted locally):
  320-element vectors (20 superlanes × 16 lanes), VXM 5,120 32-bit ops/cycle, MXM 320×320 (~750 TOPS int8),
  220 MiB SRAM, worst-case system network latency 2.5 µs. Clock 900 MHz nominal (groq.com ISCA page, via search
  summary only). **Per-instruction / per-stage latencies (cycles) are not in any source I opened** (the ISCA paper PDF
  link returned 404). Not filled in from memory.

# Hypotheses for what sets t_L
| # | hypothesis | prediction for the chip |
|---|---|---|
| HA | throughput: t_L ≈ non-weight FLOPs / compute + bytes / bandwidth | < 0.5 µs (tiny work per layer at B=1) |
| HB | latency chain: t_L ≈ S dependent stages × per-stage latency (fill/drain, reductions, on-chip transport) | S ≈ 10–14 stages; depends on cycles per stage |
| HC | empirical: real SRAM dataflow systems sit at 9–50 µs/layer end to end | t_L ~ 10–20 µs unless the design is unusually latency-optimised |

# Experiments

## L1 - Does the throughput model (HA) explain the GPU? (existing data, free)
Prediction (written before computing): HA under-predicts the GPU by > 5×; if within 2×, HA is viable.
Method: Ouro layer at B=1, C=128, non-weight work: 1.13 MFLOP and ~1.13 MB (KV read bf16 + activations). Measured =
in-step busy time of the non-weight kernels (no gaps): L40S first-graph repeat run c3, H100 cd target run.
Result: throughput limit **1.52 µs (L40S) / 0.39 µs (H100)** vs measured **11.5 / 13.2 µs**: 8× and 34× too low.
H100 has 4× L40S's bandwidth yet is slower per layer. Measured: 5.2 non-weight kernels per layer, 2.2 / 2.6 µs per kernel;
isolated tiny kernels that do almost no work take 1.0–1.4 µs each.
Verdict: **HA refuted** as a predictor on GPUs. Per-layer time is a chain of dependent steps with a fixed cost each (HB).
Same signature in the published systems: e.g. weight reads on wafer-scale SRAM are far below Cerebras's 17 µs/layer (exact
WSE-3 bandwidth not checked in an opened source), so that time is latency too.
Chip relevance: the chip estimate must count dependent stages and their latency, not operations.

# Plan (pre-registered before the bottom-up count)
L2: GPU check of HB with per-stage latency measured in isolation (tiny-kernel busy times + bytes/BW) → should land within
±25 % of 11.5 / 13.2 µs. If not, HB in that form is refuted for GPUs too.
L3: chip count: list the dependent stages of one Ouro layer in a fused weight-stationary dataflow design (norm, QKV,
rotary + cache write, QKᵀ, softmax, AV, o_proj + residual, norm, gate/up, SiLU·mul, down + residual), and for each the
cycles for: vector fill/drain at the stated width, reduction depth (log2 over the reduced dimension and across tiles),
on-chip transport between units. Clock and widths from cited figures (Groq slides above; Groq-3 LPU figures already
cited in `loopcore_model.py`); per-stage cycle costs that no source gives are swept, not invented.
Expected (written now): S ≈ 11–13; central t_L ≈ 2–8 µs; the dominant unknown is the matmul reduction latency across
K = 2048 / 5632 spread over tiles. If the bottom-up count lands far below the 9–17 µs of the best published systems, the
gap must be explained (serving batch, sampling, host, multi-chip) before the low number is used.

## L2 - GPU check of the latency-chain model (HB) — pre-registered before computing
Formula (fixed now): per-layer non-weight busy time = attention call at 1x128 (split-KV + combine, isolated busy) +
1 elementwise kernel (rotary + cache write; proxy: isolated silu_mul busy at B=1) + 2 RMSNorm kernels (isolated
rmsnorm_2048 busy at B=1). Kernel list from the in-step cd profile: 3 attention-category + 2 norm-fused kernels per layer.
Inputs: v1.2 kernel microbenchmarks (`delta_kernel_bench-*-20261118-1234*`), not the target runs.
Compare with measured in-step busy: L40S 11.5 µs (first-graph repeat run c3), H100 13.2 µs (cd target run).
Pass: within ±25 % on both GPUs. Expected: ~10 µs L40S, ~11 µs H100 (−10…−20 %).
Result: L40S 6.23 + 1.01 + 2 × 1.32 = **9.89 µs vs 11.5 (−14.0 %)**; H100 7.13 + 1.04 + 2 × 1.36 = **10.88 µs vs 13.2 (−17.6 %)**.
Verdict: **HB supported** on both GPUs (inside ±25 %, as predicted). Systematic −14…−18 %: the in-step kernels are fused
with extra work (e.g. the norms carry casts/residual adds), so isolated stand-ins are a little cheaper.
Learned: per-layer time = dependent stages × per-stage latency is the right form; on GPUs each stage costs ~1–6 µs
(launch ramp, DRAM round trips), almost independent of the work done.

## New anchor (opened 2026-11-22)
Cerebras, Llama-3.1-405B, **969 output tok/s**, 16-bit, 128K context, Nov 18 2024; vendor-measured (the page cites
Artificial Analysis only for a separate comparison); speculative decoding not mentioned; number of systems and batch not
stated. https://www.cerebras.ai/press-release/cerebras-inference-llama-405b . With 126 layers (from memory, not checked)
→ **8.2 µs per layer**, including everything per token, for a layer 8× wider than Ouro's (d = 16384 vs 2048, from memory).
Same vendor's 8B figure (17.4 µs/layer, Aug 2024) is 2× worse per layer for a smaller layer, so serving configuration and
software, not hardware per-layer latency, dominate the published numbers. Best demonstrated upper anchor: ~8 µs/layer.

## L3 - bottom-up stage count for a fused dataflow chip — pre-registered before computing
Model (`loopcore_model.layer_latency_bottom_up`): 11 dependent stages per layer at B=1: RMSNorm, QKV matmul, rotary + KV
write, QKᵀ, softmax, AV, o_proj + residual, RMSNorm, gate/up matmul, SiLU·mul, down + residual. Cycles per stage:
- elementwise: move + vec + beats(n); norm: move + 2·vec + beats(n) + red·⌈log2 n⌉
- matmul (weight-stationary, K inputs → N outputs): move + mm + beats(K) + red·⌈log2⌈K/W⌉⌉ (summing K-tiles) + beats(N)
- QKᵀ: move + vec + red·⌈log2 hd⌉; softmax: move + 2·vec + 2·red·⌈log2 C⌉; AV: move + vec + red·⌈log2 C⌉
  (streaming the C cached keys/values is bandwidth, already in the model's KV term, so not counted here)
- beats(n) = ⌈n / W⌉, W = 320 lanes (Groq TSP slides).
Parameters (low / central / high): clock 1.5 / 1.0 / 0.9 GHz (0.9 = Groq TSP nominal, cited via search summary; others
assumed); move (hand-off between units across the chip) 10 / 50 / 200 cycles; vec (vector unit pipeline) 5 / 10 / 30;
mm (matrix unit, first output after input; broadcast-tree vs a 320-deep systolic fill) 20 / 100 / 400; red (cycles per
reduction-tree level) 1 / 2 / 4. **All per-stage cycle values are assumptions: no opened source gives them.**
Not included: chip-to-chip hops (separate model term), KV access latency if not prefetched, control/synchronisation.
Expected: I sketched this formula roughly while designing it, which suggested ~1–5 µs, so my expectation is
**central ~1–2 µs, range ~0.3–5 µs at C = 2048**. If so, the bottom-up count sits 4–8× below the best published system
(~8 µs), and the honest use is: bottom-up = an optimistic floor, published = a demonstrated ceiling.
Decision rule for the model input (proposed to Sid, not applied): sweep t_L between the bottom-up central value and the
published ~8 µs, and report the speedup at both ends rather than a single number.
Result (`layer_latency_bottom_up`, committed before computing): C = 2048: **low 0.31 µs, central 1.36 µs, high 5.06 µs**
per layer (C = 128 / 8192 within ±0.05 µs). Central breakdown: 1,356 cycles; matmul stages 54 %, unit-to-unit hand-offs
alone 41 % (11 × 50 cycles). As predicted (central ~1–2 µs, range ~0.3–5 µs).
Verdict: the count is dominated by the assumed per-stage costs (hand-off and matrix-unit fill), which no opened source
gives. It is an optimistic floor for a design that achieves those costs, not evidence that a real chip will.
Reconciliation with published systems: the bottom-up central is ~6× below the best published (~8.2 µs, 405B) and ~13×
below Cerebras 8B. Candidates for the gap (untested): per-token costs spread over layers (sampling, host, LM head),
multi-chip / multi-region hops, serving batch > 1, synchronisation/control not in the count, larger per-stage costs.
Implication (vs measured H100 int8, q = 0.5; GPU keeps bf16 KV, so the fp8-KV gain is not LoopCore-only):
| per-layer latency | Ouro as trained (int8, fp8 KV, 4 caches, HBM) 1x2048 / 16x2048 | with trained KV sharing (1 cache on-chip) 1x2048 / 16x2048 |
|---|---|---|
| 0.31 µs (bottom-up low) | 19.1× / 4.8× | 65.6× / 56.8× |
| 1.36 µs (bottom-up central) | 14.6× / 4.7× | 31.9× / 44.1× |
| 5.06 µs (bottom-up high) | 8.0× / 4.3× | 11.3× / 24.5× |
| 8.2 µs (best published, 405B) | 5.8× / 4.1× | 7.3× / 17.8× |
| 17.4 µs (Cerebras 8B) | 3.2× / 3.4× | 3.6× / 9.9× |
- At B=1 the answer swings 3–19× (Ouro as trained) on this one unknown; at B=16 it barely matters (3.4–4.8×) because KV
  reads dominate there.
- 10× at B=1 for Ouro as trained needs t_L ≲ 3–4 µs: inside the bottom-up range, below anything published.
Proposed (Sid to decide; LAYER_LATENCY default unchanged): report LoopCore at B=1 as a band, bottom-up central (1.4 µs)
to best published (8.2 µs), i.e. 5.8–14.6× for Ouro as trained at 1x2048, and say which design properties (hand-off and
matrix-fill cycles) put a chip at the optimistic end.
Next: (a) find per-stage cycle figures (Groq ISCA 2020 paper text, Groq 2022 ISCA multi-chip paper, Cerebras/SambaNova
papers) to replace the swept values; (b) test the gap candidates against published small- vs large-model speeds.

# 2026-11-23: chip-side LoopSpec (L4) and a weight-install term (L5) — pre-registered before computing
Sources: `results/LATENCY_SOURCES.md` (copied today from `~/Documents/coreprof/LATENCY_SOURCES.md`, where it had been
written; content unchanged). LoopSpec counters from the raw run files `loopspec-l40s-20261123-112959` and
`loopspec_notinygemm-h100-20261123-120532` (results/NOTEBOOK_LOOPSPEC.md).

## Measured LoopSpec statistics (raw files, both GPUs, 40 prompts, ~4,600 tokens each)
| | passes of the block per token | first-proposal acceptance | second proposals accepted / rejected / skipped per token |
|---|---|---|---|
| "1" (L40S / H100) | 1.305 / 1.310 | 0.907 / 0.905 | – |
| "1:2" (L40S / H100) | 1.191 / 1.189 | 0.903 / 0.901 | 0.064 / 0.011 / 0.023 (L40S); 0.067 / 0.010 / 0.022 (H100) |
One LoopSpec "pipeline cycle" = one pass of the shared block over every live lane (`loopspec/pipeline.py`: "One cycle
advances every live lane by K recurrent iterations", K = 1). Normal decoding = 4 passes per token, one lane each.

## L4 - chip + LoopSpec (pre-registered)
Model (laptop, current chip model `step_time`, config D sized, int8, LM head on chip):
- chip + LoopSpec token time = c x step_time(B_eff rows, C, R = 4 loops) + LM-head readouts per token x LM-head time,
  c = measured passes per token (1.19 for "1:2"). Per pass the per-layer latency is paid once (shared by all lanes);
  weights are read once; KV is read once per lane (each lane attends over its own context).
- B_eff (lanes per pass) is NOT logged. Bounds from the measured counters plus the pipeline structure: lower
  4 / c (each committed token needs 4 lane-passes, no waste); upper (4 + 6 x first-reject rate + 3 x second-proposal
  rate) / c (a rejected draft discards at most 3+2+1 lane-passes). "1:2": 3.36 .. 3.98 lanes.
- Readouts per token: 1 final + 1 first proposal + second-proposal rate.
- GPU + LoopSpec at 1x2048: NOT measured yet (task 1). Provisional = measured LoopSpec ms at C≈500 (H100, "1:2", 2.89 ms)
  x (our bf16 4-loop step at 1x2048 / at 1x512), labelled provisional; replaced by task 1.
- Rows at B=1, 1x2048: (a) chip without LoopSpec (early exit, depth 3.22) vs GPU int8 (old); (b) chip without
  LoopSpec vs GPU + LoopSpec; (c) chip + LoopSpec (depth 4) vs GPU + LoopSpec.
Hypotheses and predictions (written now):
- H1 (full gain): chip + LoopSpec / chip ≈ the GPU's gain (2.3-2.6x). Expected to hold only where the chip's step is
  latency-dominated and KV reads are cheap: **shared KV on-chip**.
- H2 (verification adds per-lane work): with KV in HBM (Ouro as trained), each extra lane re-reads its 2048-token
  context (~60 µs per pass at fp8, 3.35 TB/s) — comparable to the latency part — so the chip's gain is only
  **~1.3-1.6x**. Expected to hold for Ouro as trained.
- H3 (the draft costs the chip more than the GPU): a draft is an extra LM-head readout; on-chip LM head ≈ 1-2 µs vs a
  token time of hundreds of µs. Expected **refuted** (negligible).
- Predicted row (c), Ouro as trained, 1x2048, at the new install-based latency range (L5): **~6-10x**, i.e. between
  (b) and (a). With shared KV on-chip: (c) ≈ (a), since both sides gain ~2.5x.

## L5 - weight-install term (pre-registered)
New variant, existing levels and LAYER_LATENCY untouched: matmul stage = move + installs x c_install + installs x (B−1)
+ beats(N); installs = ceil(weights in the stage / capacity), capacity 409,600 (int8; source A) or 204,800 (fp16;
LATENCY_SOURCES assumption); c_install 20 (low) .. 40 (high) cycles (sources A, C: "<40 cycles", 20-cycle load);
B−1 extra cycles per install for B rows (C, R); clock 900 MHz (A). Non-matmul stages: the "central" parameters at 900 MHz.
Predicted per layer at B=1, C=2048: matmul installs int8 31+11+57+29 = 128 → 2.8-5.7 µs; fp16 253 → 5.6-11.2 µs; plus
~0.7 µs of vector/attention stages: **int8 ~3.5-6.5 µs, fp16 ~6.5-12 µs per layer** (B=16 adds 15 x 128 cycles ≈ 2.1 µs
int8). Against the published cross-checks (Groq BERT-base 8.9, WSE-2 Llama3-8B 11.6, CS-3 8B 17 µs/layer): the gap from
the old bottom-up central (1.36 µs; 6.5-12.5x below) narrows to ~1.4-5x below.
Then: speedup tables (as trained, shared KV on-chip; B=1, 16; C=2048) over {int8 install low/high} x {eff 1.0, 1.3} x
{rows a, b, c at B=1}. Proposed numbers only; no write-up changed.

## L5 - results (`layer_latency_bottom_up(m, 2048, "install_<fmt>_<low|high>", B)`)
| variant | per layer, B=1 | of which matmul | B=16 | pre-registered |
|---|---|---|---|---|
| install int8, 20 / 40 cycles | **3.83 / 6.68 µs** | 3.14 / 5.99 | 5.97 / 8.81 | 3.5-6.5 (top end 0.2 µs above) |
| install fp16, 20 / 40 cycles | 6.61 / 12.23 µs | 5.92 / 11.54 | 10.83 / 16.45 | 6.5-12 |
| old bottom-up central (assumed costs) | 1.36 µs | 0.74 | 1.36 | – |
Matmul installs per layer: int8 128, fp16 253 (as in LATENCY_SOURCES §2). Against the published per-layer cross-checks
(Groq BERT-base 8.9, WSE-2 Llama3-8B 11.6, CS-3 8B 17 µs; all Derived, different models/settings): the int8 range is now
1.3-4.4x below them (old central: 6.5-12.5x below) → **the gap narrows**, as predicted. The remaining gap is consistent with
larger layers in those models (Llama3-8B: ~4x Ouro's weights per layer) and per-token costs spread over layers.
Evidence level: Derived from published primitives (install latency, array size, clock); the install throughput is a
latency used as a throughput (LATENCY_SOURCES notes this), and vector-stage costs remain assumptions.

## L4 - results (`python3 loopcore_model.py --loopspec-table`; vs H100, 1x2048 unless noted)
Validity: (a) at central latency, eff 1.0 = 14.6x, reproducing L3.
LoopSpec "1:2" (H100 counters): 1.189 passes/token, lanes 3.37-4.06, readouts 2.08/token. GPU per token: int8 5.01 ms
(B=1), 15.87 ms (B=16); GPU + LoopSpec 3.54 ms **PROVISIONAL** (measured 2.89 ms at C≈500, scaled by our bf16 step growth).
| design | t_L level | eff | (a) chip vs GPU int8 | (b) chip vs GPU+LS | (c) chip+LS vs GPU+LS | chip's own LS gain | B=16 vs GPU int8 |
|---|---|---|---|---|---|---|---|
| as trained | central 1.36 µs | 1.0 / 1.3 | 14.6 / 11.2x | 10.3 / 7.9x | 10.0-11.7 / 7.7-9.0x | 0.97-1.13x | 4.7 / 3.6x |
| as trained | install int8 low 3.83 µs | 1.0 / 1.3 | 9.4 / 7.2x | 6.6 / 5.1x | 8.1-9.2 / 6.2-7.1x | 1.22-1.39x | 4.3 / 3.3x |
| as trained | install int8 high 6.68 µs | 1.0 / 1.3 | 6.6 / 5.1x | 4.7 / 3.6x | 6.8-7.6 / 5.2-5.9x | 1.45-1.62x | 4.0 / 3.1x |
| shared KV on-chip | central | 1.0 / 1.3 | 31.9 / 24.6x | 22.6 / 17.4x | 48.5-50.7 / 37.3-39.0x | 2.15-2.25x | 44.1 / 33.9x |
| shared KV on-chip | install int8 low | 1.0 / 1.3 | 14.4 / 11.1x | 10.2 / 7.8x | 22.7-23.6 / 17.5-18.2x | 2.23-2.32x | 22.2 / 17.0x |
| shared KV on-chip | install int8 high | 1.0 / 1.3 | 8.8 / 6.8x | 6.2 / 4.8x | 14.9-15.3 / 11.5-11.8x | 2.40-2.46x | 16.9 / 13.0x |
Hypotheses:
- **H2 supported (Ouro as trained, KV in HBM):** the chip's own LoopSpec gain is 1.2-1.6x at the sourced latencies (and
  ~1.0x at 1.36 µs), vs ~2.5x on the GPU: each extra lane re-reads its 2048-token context from HBM (~60 µs per pass), so
  sharing the per-pass latency buys little. Predicted 1.3-1.6x: matched.
- **H1 supported (shared KV on-chip):** 2.2-2.5x, close to the GPU's gain, because KV reads come from SRAM and the
  per-pass latency dominates.
- **H3 refuted:** a draft readout costs the chip 0.84 µs (LM head on chip) per extra readout vs ~400-700 µs per token.
- Predicted row (c) as trained at the install range: ~6-10x; got 5.2-9.2x (eff 1.0-1.3): low end slightly below.
Proposed numbers only (GPU + LoopSpec at 2048 provisional until task 1 of the 2026-11-23 brief). No write-up changed.

## Robustness: does each conclusion survive? (vs H100; B=1 rows at 1x2048)
Columns: formula error ±30 % (chip time x 0.7-1.3; only the 1.3 side can break a "≥" claim); full per-layer latency range
0.31-17.4 µs (bottom-up low .. Cerebras 8B); LoopSpec (GPU + LS, chip with and without LS; provisional at 2048);
software-efficiency band 1.0-1.3 (S1). "Yes" = the stated direction/threshold holds at every point of that column.
| conclusion | ±30 % formula | full t_L range | LoopSpec | efficiency band |
|---|---|---|---|---|
| At B=1 the per-layer latency is the main chip-design lever | yes | yes (3.2-19x as trained across it) | yes (rows a-c all move with t_L) | yes |
| At B=16 with KV in HBM, speedup is ~3-5x whatever the latency | yes (3.1-4.7x) | yes (3.4-4.8x) | n/a (LoopSpec is B=1 only) | yes |
| Keeping (shared) KV on-chip is the largest lever | yes | yes (B=16: 9.9-57x vs 3.4-4.8x) | yes (chip gains more from LS with on-chip KV) | yes |
| Ouro as trained beats H100 by ≥ 3x at B=1 | yes | **no** at ≥ ~17 µs (3.2x at 17.4 µs, ~2.5x with 1.3) | yes at sourced latencies (≥ 3.6x) | yes at sourced latencies |
| Ouro as trained reaches 10x at B=1 | needs t_L < ~1.9 µs (S1) | only at t_L ≲ 3.4 µs | **no** at sourced latencies (≤ 9.2x in row c, ≤ 6.6x in row b) | no at 1.3 |
| Shared on-chip KV reaches 10x at B=1 | yes in row (c) | no at ≥ ~8 µs | yes if the chip also runs LS (11.5-23.6x); **no** if only the GPU does (row b 4.8-10.2x) | yes in row (c) |
| Ouro as trained ≥ 3x at 16x8192 | **no** (2.3x at 1.3, S1) | no at 17.4 µs (2.8x) | n/a | no |

## L4 update - GPU + LoopSpec MEASURED at 1x2048 (replaces the provisional 3.54 ms; counters from the 2048 run)
H100 + LoopSpec "1:2" = 3.74 ms/token (measured; provisional was 3.54). Counters at 2048: 1.147 passes/token, lanes
3.49-4.05, readouts 2.06. Same table, recomputed (`python3 loopcore_model.py --loopspec-table`):
| design | t_L level | eff | (a) chip vs GPU int8 | (b) chip vs GPU+LS | (c) chip+LS vs GPU+LS | chip's LS gain | B=16 |
|---|---|---|---|---|---|---|---|
| as trained | install int8 low 3.83 µs | 1.0 / 1.3 | 9.4 / 7.2x | 7.0 / 5.4x | 8.9-9.9 / 6.8-7.6x | 1.26-1.41x | 4.3 / 3.3x |
| as trained | install int8 high 6.68 µs | 1.0 / 1.3 | 6.6 / 5.1x | 5.0 / 3.8x | 7.5-8.2 / 5.7-6.3x | 1.51-1.65x | 4.0 / 3.1x |
| shared KV on-chip | install int8 low | 1.0 / 1.3 | 14.4 / 11.1x | 10.7 / 8.3x | 24.9-25.7 / 19.1-19.7x | 2.31-2.39x | 22.2 / 17.0x |
| shared KV on-chip | install int8 high | 1.0 / 1.3 | 8.8 / 6.8x | 6.6 / 5.1x | 16.4-16.7 / 12.6-12.8x | 2.48-2.53x | 16.9 / 13.0x |
(central 1.36 µs rows: as trained (b) 8.4-10.9x, (c) 8.4-12.4x; shared (c) 40.9-55.0x.)
Robustness table changes with the measured value: none of the verdicts flip. "Ouro as trained reaches 10x at B=1" stays
**no** at sourced latencies (max 9.9x in row c at 3.83 µs, eff 1.0). "Shared on-chip KV ≥ 10x at B=1": row (c) yes
(12.6-25.7x), row (b) 5.1-10.7x (no at the high-latency end).
