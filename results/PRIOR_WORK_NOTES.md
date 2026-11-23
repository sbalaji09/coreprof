# Prior-work check: hardware and serving for looped / weight-shared models (2026-11-22, web search only)

Search summaries only unless marked "opened". Verify each before citing.

**Hardware that keeps depth-shared weights on-chip (closest precedents):**
- EdgeBERT (12 nm accelerator, ALBERT-style shared layers kept on-chip, plus entropy-based early exit; MICRO 2021):
  https://arxiv.org/pdf/2011.14203.pdf . Encoder model, small scale; no decode KV cache.
- Analog 14 nm AI chip running ALBERT with one shared weight set reused over 12 layers (Nature Communications 2025):
  https://www.nature.com/articles/s41467-025-63794-4 .
- Heterogeneous on-chip memory for adapter-ALBERT: https://web3.arxiv.org/pdf/2303.16100v1 .
- No accelerator found for looped/recurrent-depth DECODER LLMs (Ouro, Huginn). LoopCore's novelty claim should be:
  LLM-scale looped decoders, per-loop KV caches, decode latency, GPU-calibrated requirements; not "first weight-shared chip".

**Looped-LM serving on GPUs (overlaps our GPU characterisation; must cite):**
- "What Sets the Serving Speed of Looped Language Models? Measurements and Optimizations Graded by Exactness"
  https://arxiv.org/pdf/2611.05150 (says Ouro launches many small kernels; Huginn wastes KV on dead depth layers).
- "Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching" https://arxiv.org/html/2608.09444v2
  (same name as this repo's DESIGN_CDB.md: ask Sid whether it is his).

**GPU-baseline fairness risk:** self-speculative decoding for looped LMs (LoopAhead https://arxiv.org/pdf/2610.21594 ,
WaveFront https://arxiv.org/pdf/2609.23033 , LoopSpec) may speed up the GPU baseline. Decision 4 requires the fastest
GPU implementation; check whether these apply at B=1 and how much they gain.
**KV:** ResidualQuant (KV quantization for looped transformers, arXiv 2610.10381).
