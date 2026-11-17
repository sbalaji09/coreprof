# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 2.85 ms (spec), 3.88 ms (measured copy peak 635 GB/s), 3.31 ms (measured read peak 744 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 2.97e-02, gate error 3.12e-02 → **MISMATCH: compiled numbers below are not trustworthy**; compile 26.5 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 3.77 | 0.16 | 1.61 | 5.54 |
| compiled graph | 342 | 3.77 | 0.15 | 0.17 | 4.09 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.80 | 4.18 | 1.63 | 1.39× |
| 16 | 2048 | 15.05 | 14.06 | 0.99 | 1.07× |
| 16 | 8192 | 40.67 | 39.58 | 1.09 | 1.03× |

- max-autotune-no-cudagraphs at B=1, C=128: 4.23 ms (compile 84.9 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 13.08 ms (compile 74.2 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 635 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 744 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 3.772 ms |
| effective weight bandwidth | 654 GB/s (88% of read peak) |
| δ (M2) | 4.206 ± 0.024 ms |
| γ (M2) | 0.0704 ± 0.0010 ms per sequence |
| α′ (M2) | 0.2629 ± 0.0006 µs per cached token |
| α′-implied KV bandwidth | 748 GB/s (100% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.164, 16x512: 0.276, 16x2048: 0.268, 16x8192: 0.262 |
| small kernels (copy + other), graph, B=1, C=128 | 0.173 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.084 ms |
| X1 crossover (weight time ÷ α′) | 14,347 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.301 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.349 | 0.301 | 668 |
| 16 | 0.324 | 0.312 | 645 |
| 64 | 0.361 | 0.350 | 575 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 340 | 3.79 | 0.16 | 0.00 | 0.18 | 4.13 |
| compiled_graph 1x128 | 342 | 3.77 | 0.15 | 0.00 | 0.17 | 4.09 |
| compiled 16x512 | 315 | 5.02 | 2.27 | 0.00 | 0.20 | 7.48 |
| compiled_graph 16x512 | 317 | 4.96 | 2.26 | 0.00 | 0.19 | 7.42 |
| compiled 16x2048 | 315 | 4.94 | 8.77 | 0.00 | 0.21 | 13.92 |
| compiled_graph 16x2048 | 317 | 4.88 | 8.77 | 0.00 | 0.20 | 13.85 |
| compiled 16x8192 | 315 | 4.94 | 34.40 | 0.00 | 0.20 | 39.55 |
| compiled_graph 16x8192 | 317 | 4.88 | 34.41 | 0.00 | 0.20 | 39.48 |
| graph_static 1x128 | 1381 | 3.77 | 0.16 | 0.56 | 1.05 | 5.54 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 3.79 ms; `triton_` 0.18 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.11 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_` 0.00 ms; `std::enable_if<!(false), void>::type internal::gemvx::kernel` 0.00 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.159 | 1.2438 | 158 | 21% |
| compiled_graph 1x128 | 0.149 | 1.1637 | 169 | 23% |
| compiled 16x512 | 2.266 | 0.2766 | 711 | 95% |
| compiled_graph 16x512 | 2.263 | 0.2762 | 712 | 96% |
| compiled 16x2048 | 8.774 | 0.2678 | 734 | 99% |
| compiled_graph 16x2048 | 8.774 | 0.2678 | 734 | 99% |
| compiled 16x8192 | 34.405 | 0.2625 | 749 | 101% |
| compiled_graph 16x8192 | 34.405 | 0.2625 | 749 | 101% |
| graph_static 1x128 | 0.156 | 1.2162 | 162 | 22% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 1.624 ± 1.551 | 0.3353 ± 0.0498 | 0.002066 ± 0.000389 |  | 0.5877 | 174.3 | 20.28 |
| M2 | 4.206 ± 0.024 | 0.0704 ± 0.0010 |  | 0.2629 ± 0.0006 | 0.9999 | -190.6 | 0.27 |
| M3 | 4.221 ± 0.030 | 0.0699 ± 0.0012 | -0.000007 ± 0.000010 | 0.2632 ± 0.0008 | 0.9999 | -189.2 | 0.26 |

M2 KV slope → effective KV bandwidth 748 GB/s (100% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.18 | 3.79 | 0.38 | 10.87× |
| 2 | 128 | 4.44 | 3.79 | 0.65 | 6.88× |
| 4 | 128 | 4.54 | 3.79 | 0.75 | 6.08× |
| 8 | 128 | 5.00 | 3.79 | 1.21 | 4.13× |
| 16 | 128 | 5.92 | 3.79 | 2.13 | 2.78× |
| 32 | 128 | 7.76 | 3.79 | 3.97 | 1.96× |
| 64 | 128 | 10.66 | 3.79 | 6.86 | 1.55× |
| 1 | 256 | 4.19 | 3.79 | 0.39 | 10.65× |
| 2 | 256 | 4.51 | 3.79 | 0.71 | 6.31× |
| 4 | 256 | 4.70 | 3.79 | 0.90 | 5.20× |
| 8 | 256 | 5.28 | 3.79 | 1.48 | 3.56× |
| 16 | 256 | 6.46 | 3.79 | 2.66 | 2.42× |
| 32 | 256 | 8.81 | 3.79 | 5.02 | 1.76× |
| 64 | 256 | 12.85 | 3.79 | 9.06 | 1.42× |
| 1 | 512 | 4.27 | 3.79 | 0.48 | 8.92× |
| 2 | 512 | 4.65 | 3.79 | 0.86 | 5.41× |
| 4 | 512 | 4.96 | 3.79 | 1.17 | 4.25× |
| 8 | 512 | 5.82 | 3.79 | 2.03 | 2.87× |
| 16 | 512 | 7.54 | 3.79 | 3.75 | 2.01× |
| 32 | 512 | 10.97 | 3.79 | 7.17 | 1.53× |
| 64 | 512 | 17.22 | 3.79 | 13.42 | 1.28× |
| 1 | 1024 | 4.46 | 3.79 | 0.66 | 6.72× |
| 2 | 1024 | 4.96 | 3.79 | 1.16 | 4.26× |
| 4 | 1024 | 5.50 | 3.79 | 1.71 | 3.22× |
| 8 | 1024 | 6.90 | 3.79 | 3.11 | 2.22× |
| 16 | 1024 | 9.72 | 3.79 | 5.93 | 1.64× |
| 32 | 1024 | 15.34 | 3.79 | 11.55 | 1.33× |
| 64 | 1024 | 25.89 | 3.79 | 22.10 | 1.17× |
| 1 | 2048 | 4.74 | 3.79 | 0.95 | 5.01× |
| 2 | 2048 | 5.51 | 3.79 | 1.71 | 3.21× |
| 4 | 2048 | 6.59 | 3.79 | 2.80 | 2.36× |
| 8 | 2048 | 9.17 | 3.79 | 5.38 | 1.71× |
| 16 | 2048 | 14.06 | 3.79 | 10.27 | 1.37× |
| 32 | 2048 | 23.93 | 3.79 | 20.14 | 1.19× |
| 64 | 2048 | 43.04 | 3.79 | 39.24 | 1.10× |
| 1 | 4096 | 5.29 | 3.79 | 1.49 | 3.54× |
| 2 | 4096 | 6.63 | 3.79 | 2.84 | 2.34× |
| 4 | 4096 | 8.74 | 3.79 | 4.95 | 1.77× |
| 8 | 4096 | 13.41 | 3.79 | 9.62 | 1.39× |
| 16 | 4096 | 22.57 | 3.79 | 18.78 | 1.20× |
| 32 | 4096 | 41.09 | 3.79 | 37.30 | 1.10× |
| 1 | 8192 | 6.35 | 3.79 | 2.56 | 2.48× |
| 2 | 8192 | 8.73 | 3.79 | 4.94 | 1.77× |
| 4 | 8192 | 13.05 | 3.79 | 9.25 | 1.41× |
| 8 | 8192 | 21.92 | 3.79 | 18.13 | 1.21× |
| 16 | 8192 | 39.58 | 3.79 | 35.79 | 1.11× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

