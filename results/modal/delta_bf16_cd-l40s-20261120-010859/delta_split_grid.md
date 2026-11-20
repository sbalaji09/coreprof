# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 2.85 ms (spec), 3.88 ms (measured copy peak 635 GB/s), 3.31 ms (measured read peak 745 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 2.98e-02, gate error 4.69e-02 → **MISMATCH: compiled numbers below are not trustworthy**; compile 37.4 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 3.77 | 0.16 | 1.64 | 5.57 |
| compiled graph | 222 | 3.12 | 0.18 | 0.39 | 3.69 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.80 | 3.75 | 2.06 | 1.55× |
| 16 | 2048 | 15.07 | 13.40 | 1.68 | 1.13× |
| 16 | 8192 | 40.73 | 39.01 | 1.72 | 1.04× |

- max-autotune-no-cudagraphs at B=1, C=128: 3.75 ms (compile 42.2 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 13.13 ms (compile 81.4 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 635 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 745 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 3.120 ms |
| effective weight bandwidth | 791 GB/s (106% of read peak) |
| δ (M2) | 4.066 ± 0.041 ms |
| γ (M2) | 0.0451 ± 0.0018 ms per sequence |
| α′ (M2) | 0.2621 ± 0.0010 µs per cached token |
| α′-implied KV bandwidth | 750 GB/s (101% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.408, 16x512: 0.281, 16x2048: 0.269, 16x8192: 0.263 |
| small kernels (copy + other), graph, B=1, C=128 | 0.394 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.052 ms |
| X1 crossover (weight time ÷ α′) | 11,902 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.302 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.349 | 0.302 | 668 |
| 16 | 0.324 | 0.310 | 648 |
| 64 | 0.362 | 0.349 | 576 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 220 | 3.13 | 0.19 | 0.10 | 0.29 | 3.71 |
| compiled_graph 1x128 | 222 | 3.12 | 0.18 | 0.10 | 0.30 | 3.69 |
| compiled 16x512 | 315 | 3.75 | 2.30 | 0.14 | 0.75 | 6.94 |
| compiled_graph 16x512 | 317 | 3.71 | 2.30 | 0.14 | 0.75 | 6.90 |
| compiled 16x2048 | 315 | 3.67 | 8.80 | 0.15 | 0.76 | 13.38 |
| compiled_graph 16x2048 | 317 | 3.62 | 8.81 | 0.14 | 0.75 | 13.32 |
| compiled 16x8192 | 315 | 3.67 | 34.42 | 0.15 | 0.76 | 38.99 |
| compiled_graph 16x8192 | 317 | 3.62 | 34.42 | 0.14 | 0.75 | 38.92 |
| graph_static 1x128 | 1381 | 3.77 | 0.16 | 0.57 | 1.07 | 5.57 |

Top kernels (first row): `triton_red_fused_mm_5` 1.48 ms; `triton_red_fused_copy_mm_1` 0.87 ms; `triton_red_fused_mm_6` 0.77 ms; `triton_red_fused_bmm_3` 0.29 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.12 ms; `triton_red_fused__to_copy_add_bmm_mean_mul_pow_rsqrt_4` 0.05 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.190 | 1.4857 | 132 | 18% |
| compiled_graph 1x128 | 0.180 | 1.4081 | 140 | 19% |
| compiled 16x512 | 2.304 | 0.2813 | 699 | 94% |
| compiled_graph 16x512 | 2.299 | 0.2807 | 701 | 94% |
| compiled 16x2048 | 8.803 | 0.2686 | 732 | 98% |
| compiled_graph 16x2048 | 8.807 | 0.2688 | 731 | 98% |
| compiled 16x8192 | 34.424 | 0.2626 | 749 | 100% |
| compiled_graph 16x8192 | 34.416 | 0.2626 | 749 | 100% |
| graph_static 1x128 | 0.156 | 1.2198 | 161 | 22% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 1.483 ± 1.546 | 0.3093 ± 0.0496 | 0.002063 ± 0.000388 |  | 0.5647 | 173.9 | 20.00 |
| M2 | 4.066 ± 0.041 | 0.0451 ± 0.0018 |  | 0.2621 ± 0.0010 | 0.9995 | -141.5 | 0.47 |
| M3 | 4.070 ± 0.052 | 0.0450 ± 0.0021 | -0.000002 ± 0.000016 | 0.2622 ± 0.0013 | 0.9995 | -139.5 | 0.48 |

M2 KV slope → effective KV bandwidth 750 GB/s (101% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.75 | 3.13 | 0.62 | 6.05× |
| 2 | 128 | 4.43 | 3.13 | 1.31 | 3.39× |
| 4 | 128 | 4.49 | 3.13 | 1.36 | 3.30× |
| 8 | 128 | 4.70 | 3.13 | 1.58 | 2.99× |
| 16 | 128 | 5.37 | 3.13 | 2.24 | 2.39× |
| 32 | 128 | 6.54 | 3.13 | 3.41 | 1.92× |
| 64 | 128 | 8.98 | 3.13 | 5.85 | 1.53× |
| 1 | 256 | 3.72 | 3.13 | 0.59 | 6.26× |
| 2 | 256 | 4.50 | 3.13 | 1.38 | 3.27× |
| 4 | 256 | 4.65 | 3.13 | 1.52 | 3.06× |
| 8 | 256 | 4.98 | 3.13 | 1.85 | 2.69× |
| 16 | 256 | 5.91 | 3.13 | 2.79 | 2.12× |
| 32 | 256 | 7.63 | 3.13 | 4.51 | 1.69× |
| 64 | 256 | 11.18 | 3.13 | 8.05 | 1.39× |
| 1 | 512 | 3.77 | 3.13 | 0.64 | 5.86× |
| 2 | 512 | 4.65 | 3.13 | 1.52 | 3.06× |
| 4 | 512 | 4.91 | 3.13 | 1.78 | 2.75× |
| 8 | 512 | 5.52 | 3.13 | 2.40 | 2.30× |
| 16 | 512 | 6.99 | 3.13 | 3.86 | 1.81× |
| 32 | 512 | 9.80 | 3.13 | 6.68 | 1.47× |
| 64 | 512 | 15.58 | 3.13 | 12.45 | 1.25× |
| 1 | 1024 | 3.98 | 3.13 | 0.86 | 4.66× |
| 2 | 1024 | 4.94 | 3.13 | 1.82 | 2.72× |
| 4 | 1024 | 5.52 | 3.13 | 2.40 | 2.30× |
| 8 | 1024 | 6.60 | 3.13 | 3.47 | 1.90× |
| 16 | 1024 | 9.25 | 3.13 | 6.12 | 1.51× |
| 32 | 1024 | 14.13 | 3.13 | 11.00 | 1.28× |
| 64 | 1024 | 24.23 | 3.13 | 21.10 | 1.15× |
| 1 | 2048 | 4.26 | 3.13 | 1.14 | 3.75× |
| 2 | 2048 | 5.50 | 3.13 | 2.37 | 2.32× |
| 4 | 2048 | 6.52 | 3.13 | 3.40 | 1.92× |
| 8 | 2048 | 8.77 | 3.13 | 5.64 | 1.55× |
| 16 | 2048 | 13.40 | 3.13 | 10.27 | 1.30× |
| 32 | 2048 | 22.64 | 3.13 | 19.51 | 1.16× |
| 64 | 2048 | 41.26 | 3.13 | 38.14 | 1.08× |
| 1 | 4096 | 4.79 | 3.13 | 1.66 | 2.88× |
| 2 | 4096 | 6.58 | 3.13 | 3.45 | 1.91× |
| 4 | 4096 | 8.67 | 3.13 | 5.54 | 1.56× |
| 8 | 4096 | 13.06 | 3.13 | 9.93 | 1.31× |
| 16 | 4096 | 22.06 | 3.13 | 18.93 | 1.17× |
| 32 | 4096 | 39.84 | 3.13 | 36.71 | 1.09× |
| 1 | 8192 | 5.86 | 3.13 | 2.73 | 2.14× |
| 2 | 8192 | 8.69 | 3.13 | 5.56 | 1.56× |
| 4 | 8192 | 12.98 | 3.13 | 9.85 | 1.32× |
| 8 | 8192 | 21.62 | 3.13 | 18.49 | 1.17× |
| 16 | 8192 | 39.01 | 3.13 | 35.88 | 1.09× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

