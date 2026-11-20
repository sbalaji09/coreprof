# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.74 ms (spec), 0.81 ms (measured copy peak 3040 GB/s), 0.83 ms (measured read peak 2981 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 3.09e-02, gate error 3.12e-02 → **MISMATCH: compiled numbers below are not trustworthy**; compile 35.0 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 1.49 | 0.18 | 1.90 | 3.57 |
| compiled graph | 222 | 1.12 | 0.21 | 0.25 | 1.58 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.90 | 1.67 | 3.22 | 2.93× |
| 16 | 2048 | 8.41 | 5.17 | 3.24 | 1.63× |
| 16 | 8192 | 15.40 | 12.58 | 2.82 | 1.22× |

- max-autotune-no-cudagraphs at B=1, C=128: 2.19 ms (compile 43.3 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 4.89 ms (compile 73.7 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,040 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,981 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 1.123 ms |
| effective weight bandwidth | 2,197 GB/s (74% of read peak) |
| δ (M2) | 2.045 ± 0.051 ms |
| γ (M2) | 0.0347 ± 0.0022 ms per sequence |
| α′ (M2) | 0.0785 ± 0.0008 µs per cached token |
| α′-implied KV bandwidth | 2,505 GB/s (84% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.609, 16x512: 0.102, 16x2048: 0.081, 16x8192: 0.074 |
| small kernels (copy + other), graph, B=1, C=128 | 0.254 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.089 ms |
| X1 crossover (weight time ÷ α′) | 14,303 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.090 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.107 | 0.090 | 2247 |
| 16 | 0.097 | 0.088 | 2276 |
| 64 | 0.105 | 0.097 | 2086 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 220 | 1.13 | 0.21 | 0.11 | 0.14 | 1.60 |
| compiled_graph 1x128 | 222 | 1.12 | 0.21 | 0.11 | 0.14 | 1.58 |
| compiled 16x512 | 411 | 1.52 | 0.84 | 0.16 | 0.47 | 2.99 |
| compiled_graph 16x512 | 413 | 1.53 | 0.83 | 0.16 | 0.47 | 2.99 |
| compiled 16x2048 | 411 | 1.52 | 2.69 | 0.16 | 0.48 | 4.85 |
| compiled_graph 16x2048 | 413 | 1.54 | 2.64 | 0.15 | 0.47 | 4.80 |
| compiled 16x8192 | 411 | 1.53 | 9.76 | 0.16 | 0.47 | 11.93 |
| compiled_graph 16x8192 | 413 | 1.53 | 9.67 | 0.16 | 0.47 | 11.82 |
| graph_static 1x128 | 1381 | 1.49 | 0.18 | 0.66 | 1.24 | 3.57 |

Top kernels (first row): `triton_red_fused_mm_5` 0.47 ms; `triton_red_fused_mm_6` 0.36 ms; `triton_red_fused_copy_mm_1` 0.30 ms; `triton_red_fused_bmm_3` 0.14 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_bmm_mean_mul_pow_rsqrt_4` 0.06 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.213 | 1.6607 | 118 | 4% |
| compiled_graph 1x128 | 0.206 | 1.6088 | 122 | 4% |
| compiled 16x512 | 0.840 | 0.1025 | 1918 | 64% |
| compiled_graph 16x512 | 0.834 | 0.1018 | 1932 | 65% |
| compiled 16x2048 | 2.692 | 0.0822 | 2393 | 80% |
| compiled_graph 16x2048 | 2.641 | 0.0806 | 2439 | 82% |
| compiled 16x8192 | 9.763 | 0.0745 | 2640 | 89% |
| compiled_graph 16x8192 | 9.668 | 0.0738 | 2666 | 89% |
| graph_static 1x128 | 0.185 | 1.4427 | 136 | 5% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.276 ± 0.744 | 0.1611 ± 0.0228 | 0.000962 ± 0.000178 |  | 0.6141 | 114.7 | 10.77 |
| M2 | 2.045 ± 0.051 | 0.0347 ± 0.0022 |  | 0.0785 ± 0.0008 | 0.9973 | -123.5 | 0.85 |
| M3 | 2.152 ± 0.062 | 0.0317 ± 0.0023 | -0.000049 ± 0.000018 | 0.0801 ± 0.0009 | 0.9977 | -128.7 | 0.68 |

M2 KV slope → effective KV bandwidth 2505 GB/s (84% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.67 | 1.13 | 0.54 | 3.10× |
| 2 | 128 | 2.46 | 1.13 | 1.33 | 1.85× |
| 4 | 128 | 2.51 | 1.13 | 1.38 | 1.82× |
| 8 | 128 | 2.60 | 1.13 | 1.47 | 1.77× |
| 16 | 128 | 2.80 | 1.13 | 1.67 | 1.68× |
| 32 | 128 | 3.47 | 1.13 | 2.33 | 1.48× |
| 64 | 128 | 4.90 | 1.13 | 3.76 | 1.30× |
| 1 | 256 | 1.72 | 1.13 | 0.59 | 2.92× |
| 2 | 256 | 2.36 | 1.13 | 1.23 | 1.92× |
| 4 | 256 | 2.57 | 1.13 | 1.44 | 1.79× |
| 8 | 256 | 2.68 | 1.13 | 1.55 | 1.73× |
| 16 | 256 | 3.03 | 1.13 | 1.90 | 1.59× |
| 32 | 256 | 3.90 | 1.13 | 2.77 | 1.41× |
| 64 | 256 | 5.61 | 1.13 | 4.48 | 1.25× |
| 1 | 512 | 1.75 | 1.13 | 0.62 | 2.83× |
| 2 | 512 | 2.42 | 1.13 | 1.29 | 1.88× |
| 4 | 512 | 2.55 | 1.13 | 1.41 | 1.80× |
| 8 | 512 | 2.68 | 1.13 | 1.55 | 1.73× |
| 16 | 512 | 3.34 | 1.13 | 2.21 | 1.51× |
| 32 | 512 | 4.51 | 1.13 | 3.38 | 1.34× |
| 64 | 512 | 6.86 | 1.13 | 5.73 | 1.20× |
| 1 | 1024 | 1.79 | 1.13 | 0.65 | 2.73× |
| 2 | 1024 | 2.61 | 1.13 | 1.48 | 1.76× |
| 4 | 1024 | 2.76 | 1.13 | 1.63 | 1.69× |
| 8 | 1024 | 2.96 | 1.13 | 1.83 | 1.62× |
| 16 | 1024 | 3.81 | 1.13 | 2.67 | 1.42× |
| 32 | 1024 | 5.64 | 1.13 | 4.51 | 1.25× |
| 64 | 1024 | 9.15 | 1.13 | 8.02 | 1.14× |
| 1 | 2048 | 2.15 | 1.13 | 1.02 | 2.11× |
| 2 | 2048 | 2.73 | 1.13 | 1.59 | 1.71× |
| 4 | 2048 | 2.91 | 1.13 | 1.78 | 1.64× |
| 8 | 2048 | 3.58 | 1.13 | 2.45 | 1.46× |
| 16 | 2048 | 5.17 | 1.13 | 4.03 | 1.28× |
| 32 | 2048 | 8.18 | 1.13 | 7.05 | 1.16× |
| 64 | 2048 | 14.40 | 1.13 | 13.27 | 1.09× |
| 1 | 4096 | 2.18 | 1.13 | 1.05 | 2.08× |
| 2 | 4096 | 3.35 | 1.13 | 2.22 | 1.51× |
| 4 | 4096 | 3.54 | 1.13 | 2.41 | 1.47× |
| 8 | 4096 | 4.49 | 1.13 | 3.36 | 1.34× |
| 16 | 4096 | 7.54 | 1.13 | 6.41 | 1.18× |
| 32 | 4096 | 13.46 | 1.13 | 12.33 | 1.09× |
| 64 | 4096 | 25.13 | 1.13 | 23.99 | 1.05× |
| 1 | 8192 | 2.52 | 1.13 | 1.39 | 1.81× |
| 2 | 8192 | 3.41 | 1.13 | 2.27 | 1.50× |
| 4 | 8192 | 4.57 | 1.13 | 3.44 | 1.33× |
| 8 | 8192 | 6.62 | 1.13 | 5.49 | 1.21× |
| 16 | 8192 | 12.58 | 1.13 | 11.45 | 1.10× |
| 32 | 8192 | 24.07 | 1.13 | 22.94 | 1.05× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

