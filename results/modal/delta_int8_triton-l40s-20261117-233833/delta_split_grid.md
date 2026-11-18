# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 1.43 ms (spec), 1.94 ms (measured copy peak 636 GB/s), 1.79 ms (measured read peak 689 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.2 %, exit gate 0.188, top-1 next token agrees 88 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.18 | 2.60 | 1.58 | 1.61× |
| 16 | 2048 | 13.99 | 11.30 | 2.68 | 1.24× |
| 16 | 8192 | 39.58 | 36.92 | 2.66 | 1.07× |

Weight time at B=1, C=128: bf16 3.770 ms (654 GB/s) → int8 ≈ 2.188 ms (564 GB/s), i.e. 0.58× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 2.60 ms (compile 25.1 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 11.32 ms (compile 27.6 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 636 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 689 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 2.188 ms |
| effective weight bandwidth | 564 GB/s (82% of read peak) |
| δ (M2) | 2.544 ± 0.021 ms |
| γ (M2) | 0.0112 ± 0.0009 ms per sequence |
| α′ (M2) | 0.2618 ± 0.0005 µs per cached token |
| α′-implied KV bandwidth | 751 GB/s (109% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.456, 16x512: 0.277, 16x2048: 0.268, 16x8192: 0.263 |
| small kernels (copy + other), graph, B=1, C=128 | 2.323 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.082 ms |
| X1 crossover (weight time ÷ α′) | 8,356 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.302 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.351 | 0.302 | 667 |
| 16 | 0.324 | 0.311 | 647 |
| 64 | 0.363 | 0.350 | 576 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 340 | 0.01 | 0.20 | 0.13 | 2.20 | 2.54 |
| int8_graph 1x128 | 342 | 0.01 | 0.19 | 0.13 | 2.20 | 2.52 |
| int8_compiled 16x512 | 315 | 0.00 | 2.27 | 0.14 | 2.43 | 4.85 |
| int8_graph 16x512 | 317 | 0.00 | 2.27 | 0.14 | 2.39 | 4.79 |
| int8_compiled 16x2048 | 315 | 0.00 | 8.80 | 0.15 | 2.37 | 11.31 |
| int8_graph 16x2048 | 317 | 0.00 | 8.79 | 0.14 | 2.32 | 11.25 |
| int8_compiled 16x8192 | 315 | 0.00 | 34.43 | 0.14 | 2.35 | 36.93 |
| int8_graph 16x8192 | 317 | 0.00 | 34.41 | 0.13 | 2.31 | 36.86 |
| compiled_graph 1x128 | 342 | 3.77 | 0.17 | 0.13 | 0.03 | 4.10 |

Top kernels (first row): `_int8_wo_kernel` 2.17 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3` 0.05 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6` 0.05 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms; `triton_poi_fused__scaled_dot_product_flash_attention_add_cat` 0.03 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.199 | 1.5556 | 126 | 18% |
| int8_graph 1x128 | 0.186 | 1.4556 | 135 | 20% |
| int8_compiled 16x512 | 2.274 | 0.2776 | 708 | 103% |
| int8_graph 16x512 | 2.270 | 0.2771 | 709 | 103% |
| int8_compiled 16x2048 | 8.795 | 0.2684 | 732 | 106% |
| int8_graph 16x2048 | 8.789 | 0.2682 | 733 | 106% |
| int8_compiled 16x8192 | 34.429 | 0.2627 | 748 | 109% |
| int8_graph 16x8192 | 34.414 | 0.2626 | 749 | 109% |
| compiled_graph 1x128 | 0.172 | 1.3440 | 146 | 21% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | -0.053 ± 1.540 | 0.2753 ± 0.0495 | 0.002067 ± 0.000386 |  | 0.5345 | 173.6 | 19.84 |
| M2 | 2.544 ± 0.021 | 0.0112 ± 0.0009 |  | 0.2618 ± 0.0005 | 0.9999 | -202.3 | 0.32 |
| M3 | 2.526 ± 0.027 | 0.0118 ± 0.0011 | 0.000009 ± 0.000008 | 0.2614 ± 0.0007 | 0.9999 | -201.6 | 0.31 |

M2 KV slope → effective KV bandwidth 751 GB/s (109% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.60 | 2.19 | 0.41 | 6.31× |
| 2 | 128 | 2.61 | 2.19 | 0.42 | 6.20× |
| 4 | 128 | 2.72 | 2.19 | 0.54 | 5.08× |
| 8 | 128 | 2.85 | 2.19 | 0.66 | 4.30× |
| 16 | 128 | 3.22 | 2.19 | 1.03 | 3.12× |
| 32 | 128 | 3.71 | 2.19 | 1.52 | 2.44× |
| 64 | 128 | 5.41 | 2.19 | 3.22 | 1.68× |
| 1 | 256 | 2.63 | 2.19 | 0.44 | 5.92× |
| 2 | 256 | 2.68 | 2.19 | 0.49 | 5.44× |
| 4 | 256 | 2.88 | 2.19 | 0.69 | 4.15× |
| 8 | 256 | 3.13 | 2.19 | 0.94 | 3.32× |
| 16 | 256 | 3.78 | 2.19 | 1.59 | 2.38× |
| 32 | 256 | 4.83 | 2.19 | 2.64 | 1.83× |
| 64 | 256 | 7.59 | 2.19 | 5.41 | 1.40× |
| 1 | 512 | 2.70 | 2.19 | 0.51 | 5.25× |
| 2 | 512 | 2.82 | 2.19 | 0.63 | 4.45× |
| 4 | 512 | 3.14 | 2.19 | 0.95 | 3.31× |
| 8 | 512 | 3.67 | 2.19 | 1.48 | 2.48× |
| 16 | 512 | 4.87 | 2.19 | 2.68 | 1.82× |
| 32 | 512 | 7.01 | 2.19 | 4.83 | 1.45× |
| 64 | 512 | 11.92 | 2.19 | 9.74 | 1.22× |
| 1 | 1024 | 2.85 | 2.19 | 0.67 | 4.29× |
| 2 | 1024 | 3.09 | 2.19 | 0.90 | 3.42× |
| 4 | 1024 | 3.67 | 2.19 | 1.48 | 2.48× |
| 8 | 1024 | 4.76 | 2.19 | 2.57 | 1.85× |
| 16 | 1024 | 7.26 | 2.19 | 5.07 | 1.43× |
| 32 | 1024 | 11.36 | 2.19 | 9.18 | 1.24× |
| 64 | 1024 | 20.63 | 2.19 | 18.44 | 1.12× |
| 1 | 2048 | 3.15 | 2.19 | 0.96 | 3.28× |
| 2 | 2048 | 3.66 | 2.19 | 1.47 | 2.49× |
| 4 | 2048 | 4.74 | 2.19 | 2.56 | 1.86× |
| 8 | 2048 | 6.93 | 2.19 | 4.74 | 1.46× |
| 16 | 2048 | 11.30 | 2.19 | 9.12 | 1.24× |
| 32 | 2048 | 19.90 | 2.19 | 17.71 | 1.12× |
| 64 | 2048 | 37.75 | 2.19 | 35.56 | 1.06× |
| 1 | 4096 | 3.69 | 2.19 | 1.50 | 2.46× |
| 2 | 4096 | 4.74 | 2.19 | 2.55 | 1.86× |
| 4 | 4096 | 6.91 | 2.19 | 4.72 | 1.46× |
| 8 | 4096 | 11.53 | 2.19 | 9.34 | 1.23× |
| 16 | 4096 | 19.89 | 2.19 | 17.70 | 1.12× |
| 32 | 4096 | 37.07 | 2.19 | 34.88 | 1.06× |
| 1 | 8192 | 4.77 | 2.19 | 2.58 | 1.85× |
| 2 | 8192 | 6.85 | 2.19 | 4.67 | 1.47× |
| 4 | 8192 | 11.21 | 2.19 | 9.03 | 1.24× |
| 8 | 8192 | 19.79 | 2.19 | 17.61 | 1.12× |
| 16 | 8192 | 36.92 | 2.19 | 34.73 | 1.06× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

## Errors

- **static_vs_hf_check**: `AssertionError: decode kernel: one BLOCK_M covers M <= 64`
