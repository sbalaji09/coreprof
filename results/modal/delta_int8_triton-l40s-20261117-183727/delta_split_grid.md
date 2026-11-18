# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 1.43 ms (spec), 1.87 ms (measured copy peak 659 GB/s), 1.80 ms (measured read peak 685 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.2 %, exit gate 0.188, top-1 next token agrees 88 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.20 | 2.71 | 1.50 | 1.55× |
| 16 | 2048 | 13.91 | 11.51 | 2.40 | 1.21× |
| 16 | 8192 | 39.54 | 37.13 | 2.42 | 1.07× |

Weight time at B=1, C=128: bf16 3.772 ms (654 GB/s) → int8 ≈ 2.276 ms (542 GB/s), i.e. 0.60× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 2.62 ms (compile 18.7 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 11.51 ms (compile 20.7 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 659 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 685 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 2.276 ms |
| effective weight bandwidth | 542 GB/s (79% of read peak) |
| δ (M2) | 2.650 ± 0.023 ms |
| γ (M2) | 0.0101 ± 0.0010 ms per sequence |
| α′ (M2) | 0.2624 ± 0.0006 µs per cached token |
| α′-implied KV bandwidth | 749 GB/s (109% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.446, 16x512: 0.277, 16x2048: 0.268, 16x8192: 0.263 |
| small kernels (copy + other), graph, B=1, C=128 | 2.320 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.194 ms |
| X1 crossover (weight time ÷ α′) | 8,677 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.295 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.331 | 0.295 | 682 |
| 16 | 0.304 | 0.305 | 659 |
| 64 | 0.342 | 0.344 | 585 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 340 | 0.01 | 0.20 | 0.13 | 2.19 | 2.53 |
| int8_graph 1x128 | 342 | 0.01 | 0.19 | 0.12 | 2.19 | 2.51 |
| int8_compiled 16x512 | 315 | 0.00 | 2.28 | 0.14 | 2.58 | 5.00 |
| int8_graph 16x512 | 317 | 0.00 | 2.27 | 0.13 | 2.55 | 4.96 |
| int8_compiled 16x2048 | 315 | 0.00 | 8.79 | 0.14 | 2.51 | 11.45 |
| int8_graph 16x2048 | 317 | 0.00 | 8.79 | 0.13 | 2.48 | 11.41 |
| int8_compiled 16x8192 | 315 | 0.00 | 34.42 | 0.14 | 2.51 | 37.08 |
| int8_graph 16x8192 | 317 | 0.00 | 34.44 | 0.13 | 2.49 | 37.06 |
| compiled_graph 1x128 | 342 | 3.77 | 0.17 | 0.12 | 0.02 | 4.09 |

Top kernels (first row): `_int8_wo_kernel` 2.17 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3` 0.05 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6` 0.05 ms; `triton_poi_fused__scaled_dot_product_flash_attention_add_cat` 0.03 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.197 | 1.5384 | 128 | 19% |
| int8_graph 1x128 | 0.185 | 1.4463 | 136 | 20% |
| int8_compiled 16x512 | 2.276 | 0.2778 | 708 | 103% |
| int8_graph 16x512 | 2.273 | 0.2774 | 709 | 103% |
| int8_compiled 16x2048 | 8.792 | 0.2683 | 733 | 107% |
| int8_graph 16x2048 | 8.788 | 0.2682 | 733 | 107% |
| int8_compiled 16x8192 | 34.424 | 0.2626 | 749 | 109% |
| int8_graph 16x8192 | 34.440 | 0.2628 | 748 | 109% |
| compiled_graph 1x128 | 0.171 | 1.3348 | 147 | 22% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.058 ± 1.545 | 0.2746 ± 0.0496 | 0.002068 ± 0.000388 |  | 0.5323 | 173.9 | 19.82 |
| M2 | 2.650 ± 0.023 | 0.0101 ± 0.0010 |  | 0.2624 ± 0.0006 | 0.9998 | -192.5 | 0.26 |
| M3 | 2.645 ± 0.030 | 0.0103 ± 0.0012 | 0.000002 ± 0.000009 | 0.2622 ± 0.0008 | 0.9998 | -190.5 | 0.26 |

M2 KV slope → effective KV bandwidth 749 GB/s (109% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.71 | 2.28 | 0.43 | 6.29× |
| 2 | 128 | 2.69 | 2.28 | 0.42 | 6.45× |
| 4 | 128 | 2.75 | 2.28 | 0.48 | 5.76× |
| 8 | 128 | 3.14 | 2.28 | 0.86 | 3.65× |
| 16 | 128 | 3.41 | 2.28 | 1.14 | 3.00× |
| 32 | 128 | 3.79 | 2.28 | 1.52 | 2.50× |
| 64 | 128 | 5.39 | 2.28 | 3.11 | 1.73× |
| 1 | 256 | 2.65 | 2.28 | 0.37 | 7.15× |
| 2 | 256 | 2.77 | 2.28 | 0.49 | 5.62× |
| 4 | 256 | 2.91 | 2.28 | 0.64 | 4.57× |
| 8 | 256 | 3.42 | 2.28 | 1.15 | 2.98× |
| 16 | 256 | 3.95 | 2.28 | 1.68 | 2.36× |
| 32 | 256 | 4.91 | 2.28 | 2.63 | 1.87× |
| 64 | 256 | 7.62 | 2.28 | 5.34 | 1.43× |
| 1 | 512 | 2.72 | 2.28 | 0.44 | 6.15× |
| 2 | 512 | 2.91 | 2.28 | 0.64 | 4.58× |
| 4 | 512 | 3.17 | 2.28 | 0.90 | 3.54× |
| 8 | 512 | 3.96 | 2.28 | 1.68 | 2.35× |
| 16 | 512 | 5.07 | 2.28 | 2.79 | 1.82× |
| 32 | 512 | 7.10 | 2.28 | 4.83 | 1.47× |
| 64 | 512 | 11.96 | 2.28 | 9.69 | 1.23× |
| 1 | 1024 | 2.87 | 2.28 | 0.59 | 4.84× |
| 2 | 1024 | 3.19 | 2.28 | 0.91 | 3.50× |
| 4 | 1024 | 3.71 | 2.28 | 1.43 | 2.59× |
| 8 | 1024 | 5.04 | 2.28 | 2.77 | 1.82× |
| 16 | 1024 | 7.24 | 2.28 | 4.96 | 1.46× |
| 32 | 1024 | 11.45 | 2.28 | 9.17 | 1.25× |
| 64 | 1024 | 20.72 | 2.28 | 18.44 | 1.12× |
| 1 | 2048 | 3.15 | 2.28 | 0.88 | 3.59× |
| 2 | 2048 | 3.75 | 2.28 | 1.47 | 2.55× |
| 4 | 2048 | 4.78 | 2.28 | 2.51 | 1.91× |
| 8 | 2048 | 7.22 | 2.28 | 4.94 | 1.46× |
| 16 | 2048 | 11.51 | 2.28 | 9.24 | 1.25× |
| 32 | 2048 | 19.97 | 2.28 | 17.69 | 1.13× |
| 64 | 2048 | 37.80 | 2.28 | 35.53 | 1.06× |
| 1 | 4096 | 3.70 | 2.28 | 1.42 | 2.60× |
| 2 | 4096 | 4.84 | 2.28 | 2.56 | 1.89× |
| 4 | 4096 | 6.94 | 2.28 | 4.67 | 1.49× |
| 8 | 4096 | 11.52 | 2.28 | 9.24 | 1.25× |
| 16 | 4096 | 20.09 | 2.28 | 17.81 | 1.13× |
| 32 | 4096 | 37.13 | 2.28 | 34.85 | 1.07× |
| 1 | 8192 | 4.78 | 2.28 | 2.50 | 1.91× |
| 2 | 8192 | 6.96 | 2.28 | 4.68 | 1.49× |
| 4 | 8192 | 11.26 | 2.28 | 8.98 | 1.25× |
| 8 | 8192 | 20.08 | 2.28 | 17.80 | 1.13× |
| 16 | 8192 | 37.13 | 2.28 | 34.85 | 1.07× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

## Errors

- **static_vs_hf_check**: `AssertionError: decode kernel: one BLOCK_M covers M <= 64`
