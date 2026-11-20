# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.37 ms (spec), 0.41 ms (measured copy peak 3014 GB/s), 0.43 ms (measured read peak 2871 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.2 %, exit gate 0.312, top-1 next token agrees 88 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.07 | 2.42 | -0.35 | 0.86× |
| 16 | 2048 | 5.12 | 4.89 | 0.22 | 1.05× |
| 16 | 8192 | 12.40 | 12.18 | 0.23 | 1.02× |

Weight time at B=1, C=128: bf16 1.562 ms (1579 GB/s) → int8 ≈ 1.913 ms (645 GB/s), i.e. 1.22× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 2.43 ms (compile 17.9 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 4.92 ms (compile 19.3 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,014 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,871 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 1.913 ms |
| effective weight bandwidth | 645 GB/s (22% of read peak) |
| δ (M2) | 2.208 ± 0.048 ms |
| γ (M2) | 0.0255 ± 0.0021 ms per sequence |
| α′ (M2) | 0.0744 ± 0.0007 µs per cached token |
| α′-implied KV bandwidth | 2,643 GB/s (92% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.667, 16x512: 0.102, 16x2048: 0.081, 16x8192: 0.074 |
| small kernels (copy + other), graph, B=1, C=128 | 2.043 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.164 ms |
| X1 crossover (weight time ÷ α′) | 25,713 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.096 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.118 | 0.096 | 2089 |
| 16 | 0.098 | 0.092 | 2187 |
| 64 | 0.111 | 0.100 | 2016 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 340 | 0.00 | 0.22 | 0.15 | 1.92 | 2.30 |
| int8_graph 1x128 | 342 | 0.00 | 0.21 | 0.14 | 1.90 | 2.26 |
| int8_compiled 16x512 | 315 | 0.00 | 0.85 | 0.16 | 1.95 | 2.96 |
| int8_graph 16x512 | 317 | 0.00 | 0.84 | 0.15 | 1.91 | 2.90 |
| int8_compiled 16x2048 | 315 | 0.00 | 2.74 | 0.16 | 1.95 | 4.85 |
| int8_graph 16x2048 | 317 | 0.00 | 2.66 | 0.15 | 1.90 | 4.72 |
| int8_compiled 16x8192 | 315 | 0.00 | 9.95 | 0.16 | 1.92 | 12.03 |
| int8_graph 16x8192 | 317 | 0.00 | 9.66 | 0.15 | 1.87 | 11.68 |
| compiled_graph 1x128 | 342 | 1.56 | 0.22 | 0.16 | 0.03 | 1.97 |

Top kernels (first row): `_int8_wo_kernel` 1.89 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3` 0.06 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6` 0.05 ms; `triton_poi_fused__scaled_dot_product_flash_attention_add_cat` 0.03 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.221 | 1.7285 | 114 | 4% |
| int8_graph 1x128 | 0.213 | 1.6667 | 118 | 4% |
| int8_compiled 16x512 | 0.849 | 0.1036 | 1898 | 66% |
| int8_graph 16x512 | 0.837 | 0.1022 | 1924 | 67% |
| int8_compiled 16x2048 | 2.742 | 0.0837 | 2350 | 82% |
| int8_graph 16x2048 | 2.663 | 0.0813 | 2419 | 84% |
| int8_compiled 16x8192 | 9.949 | 0.0759 | 2590 | 90% |
| int8_graph 16x8192 | 9.655 | 0.0737 | 2669 | 93% |
| compiled_graph 1x128 | 0.223 | 1.7403 | 113 | 4% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.492 ± 0.697 | 0.1454 ± 0.0214 | 0.000928 ± 0.000167 |  | 0.6085 | 108.3 | 10.33 |
| M2 | 2.208 ± 0.048 | 0.0255 ± 0.0021 |  | 0.0744 ± 0.0007 | 0.9972 | -129.2 | 0.66 |
| M3 | 2.248 ± 0.063 | 0.0243 ± 0.0024 | -0.000018 ± 0.000018 | 0.0750 ± 0.0009 | 0.9973 | -128.3 | 0.59 |

M2 KV slope → effective KV bandwidth 2643 GB/s (92% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.42 | 1.91 | 0.51 | 4.74× |
| 2 | 128 | 2.41 | 1.91 | 0.49 | 4.87× |
| 4 | 128 | 2.44 | 1.91 | 0.53 | 4.63× |
| 8 | 128 | 2.48 | 1.91 | 0.57 | 4.35× |
| 16 | 128 | 2.58 | 1.91 | 0.67 | 3.85× |
| 32 | 128 | 3.00 | 1.91 | 1.08 | 2.77× |
| 64 | 128 | 4.65 | 1.91 | 2.74 | 1.70× |
| 1 | 256 | 2.43 | 1.91 | 0.52 | 4.69× |
| 2 | 256 | 2.41 | 1.91 | 0.50 | 4.85× |
| 4 | 256 | 2.49 | 1.91 | 0.58 | 4.30× |
| 8 | 256 | 2.58 | 1.91 | 0.66 | 3.88× |
| 16 | 256 | 2.72 | 1.91 | 0.81 | 3.37× |
| 32 | 256 | 3.30 | 1.91 | 1.39 | 2.38× |
| 64 | 256 | 5.23 | 1.91 | 3.32 | 1.58× |
| 1 | 512 | 2.47 | 1.91 | 0.56 | 4.41× |
| 2 | 512 | 2.48 | 1.91 | 0.57 | 4.35× |
| 4 | 512 | 2.59 | 1.91 | 0.67 | 3.84× |
| 8 | 512 | 2.69 | 1.91 | 0.78 | 3.45× |
| 16 | 512 | 3.06 | 1.91 | 1.15 | 2.67× |
| 32 | 512 | 3.88 | 1.91 | 1.97 | 1.97× |
| 64 | 512 | 6.50 | 1.91 | 4.58 | 1.42× |
| 1 | 1024 | 2.54 | 1.91 | 0.63 | 4.05× |
| 2 | 1024 | 2.55 | 1.91 | 0.64 | 4.00× |
| 4 | 1024 | 2.70 | 1.91 | 0.79 | 3.42× |
| 8 | 1024 | 2.95 | 1.91 | 1.04 | 2.84× |
| 16 | 1024 | 3.60 | 1.91 | 1.69 | 2.13× |
| 32 | 1024 | 5.19 | 1.91 | 3.27 | 1.58× |
| 64 | 1024 | 8.87 | 1.91 | 6.96 | 1.27× |
| 1 | 2048 | 2.60 | 1.91 | 0.68 | 3.80× |
| 2 | 2048 | 2.67 | 1.91 | 0.76 | 3.51× |
| 4 | 2048 | 2.99 | 1.91 | 1.08 | 2.78× |
| 8 | 2048 | 3.52 | 1.91 | 1.60 | 2.19× |
| 16 | 2048 | 4.89 | 1.91 | 2.98 | 1.64× |
| 32 | 2048 | 7.51 | 1.91 | 5.59 | 1.34× |
| 64 | 2048 | 13.66 | 1.91 | 11.75 | 1.16× |
| 1 | 4096 | 2.90 | 1.91 | 0.98 | 2.94× |
| 2 | 4096 | 3.31 | 1.91 | 1.39 | 2.37× |
| 4 | 4096 | 3.53 | 1.91 | 1.61 | 2.18× |
| 8 | 4096 | 4.56 | 1.91 | 2.65 | 1.72× |
| 16 | 4096 | 7.31 | 1.91 | 5.40 | 1.35× |
| 32 | 4096 | 12.32 | 1.91 | 10.40 | 1.18× |
| 64 | 4096 | 23.93 | 1.91 | 22.02 | 1.09× |
| 1 | 8192 | 3.17 | 1.91 | 1.25 | 2.53× |
| 2 | 8192 | 3.50 | 1.91 | 1.59 | 2.20× |
| 4 | 8192 | 4.58 | 1.91 | 2.67 | 1.72× |
| 8 | 8192 | 6.62 | 1.91 | 4.71 | 1.41× |
| 16 | 8192 | 12.18 | 1.91 | 10.26 | 1.19× |
| 32 | 8192 | 22.58 | 1.91 | 20.67 | 1.09× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

