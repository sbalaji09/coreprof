# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.37 ms (spec), 0.41 ms (measured copy peak 3021 GB/s), 0.43 ms (measured read peak 2876 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.4 %, exit gate 0.438, top-1 next token agrees 94 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.71 | 1.33 | 0.38 | 1.29× |
| 16 | 2048 | 5.07 | n/a | n/a | n/a× |
| 16 | 8192 | 12.48 | n/a | n/a | n/a× |

Weight time at B=1, C=128: bf16 0.004 ms (560960 GB/s) → int8 ≈ -0.378 ms (n/a GB/s), i.e. -86.05× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 1.22 ms (compile 34.0 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 6.33 ms (compile 67.6 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,021 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,876 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | -0.378 ms |
| effective weight bandwidth | -3,263 GB/s (-113% of read peak) |
| δ (M2) | -0.682 ± 0.114 ms |
| γ (M2) | 2.0446 ± 0.0746 ms per sequence |
| α′ (M2) | 0.0787 ± 0.0084 µs per cached token |
| α′-implied KV bandwidth | 2,500 GB/s (87% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.689, 16x512: 0.119, 16x2048: 0.085, 16x8192: 0.075 |
| small kernels (copy + other), graph, B=1, C=128 | 0.916 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.190 ms |
| X1 crossover (weight time ÷ α′) | -4,811 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.094 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.114 | 0.094 | 2139 |
| 16 | 0.098 | 0.091 | 2211 |
| 64 | 0.106 | 0.099 | 2034 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 220 | 0.00 | 0.22 | 0.28 | 0.64 | 1.15 |
| int8_graph 1x128 | 222 | 0.00 | 0.22 | 0.28 | 0.64 | 1.14 |
| int8_compiled 16x512 | 555 | 1.26 | 0.95 | 1.21 | 0.50 | 3.92 |
| int8_graph 16x512 | 557 | 1.31 | 0.97 | 1.26 | 0.52 | 4.07 |
| int8_compiled 16x2048 | 555 | 1.25 | 2.82 | 1.21 | 0.52 | 5.80 |
| int8_graph 16x2048 | 557 | 1.28 | 2.79 | 1.22 | 0.52 | 5.80 |
| int8_compiled 16x8192 | 555 | 1.25 | 9.94 | 1.21 | 0.50 | 12.90 |
| int8_graph 16x8192 | 557 | 1.27 | 9.89 | 1.21 | 0.50 | 12.88 |
| compiled_graph 1x128 | 222 | 0.00 | 0.21 | 0.41 | 0.97 | 1.59 |

Top kernels (first row): `triton_red_fused_mm_5` 0.29 ms; `triton_red_fused_mm_6` 0.24 ms; `triton_red_fused_copy_mm_1` 0.17 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused_bmm_3` 0.11 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7` 0.10 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.222 | 1.7374 | 113 | 4% |
| int8_graph 1x128 | 0.216 | 1.6891 | 116 | 4% |
| int8_compiled 16x512 | 0.946 | 0.1155 | 1702 | 59% |
| int8_graph 16x512 | 0.974 | 0.1189 | 1653 | 57% |
| int8_compiled 16x2048 | 2.823 | 0.0862 | 2282 | 79% |
| int8_graph 16x2048 | 2.792 | 0.0852 | 2308 | 80% |
| int8_compiled 16x8192 | 9.943 | 0.0759 | 2592 | 90% |
| int8_graph 16x8192 | 9.895 | 0.0755 | 2604 | 91% |
| compiled_graph 1x128 | 0.207 | 1.6204 | 121 | 4% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | -0.967 ± 0.129 | 2.2272 ± 0.0789 | 0.000123 ± 0.000015 |  | 0.9875 | -51.0 | 0.37 |
| M2 | -0.682 ± 0.114 | 2.0446 ± 0.0746 |  | 0.0787 ± 0.0084 | 0.9896 | -53.5 | 0.31 |
| M3 | -0.793 ± 0.147 | 2.1116 ± 0.0931 | 0.000048 ± 0.000041 | 0.0498 ± 0.0261 | 0.9908 | -53.3 | 0.33 |

M2 KV slope → effective KV bandwidth 2500 GB/s (87% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.33 | -0.38 | 1.70 | 0.78× |
| 2 | 128 | 3.26 | -0.38 | 3.64 | 0.90× |
| 1 | 256 | 1.30 | -0.38 | 1.67 | 0.77× |
| 2 | 256 | 3.45 | -0.38 | 3.83 | 0.90× |
| 1 | 512 | 1.38 | -0.38 | 1.76 | 0.78× |
| 2 | 512 | 3.50 | -0.38 | 3.88 | 0.90× |
| 1 | 1024 | 1.45 | -0.38 | 1.82 | 0.79× |
| 2 | 1024 | 3.59 | -0.38 | 3.97 | 0.90× |
| 1 | 2048 | 1.53 | -0.38 | 1.90 | 0.80× |
| 2 | 2048 | 3.77 | -0.38 | 4.15 | 0.91× |
| 1 | 4096 | 1.74 | -0.38 | 2.12 | 0.82× |
| 2 | 4096 | 4.36 | -0.38 | 4.74 | 0.92× |
| 1 | 8192 | 2.11 | -0.38 | 2.48 | 0.85× |
| 2 | 8192 | 4.47 | -0.38 | 4.85 | 0.92× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

