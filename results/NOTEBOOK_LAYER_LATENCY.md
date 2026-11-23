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
