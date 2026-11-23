# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA A100-SXM4-80GB

Predicted weight fetch 1.21 ms (spec), 1.42 ms (measured copy peak 1737 GB/s), 1.61 ms (measured read peak 1527 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 2.96e-02, gate error 9.38e-02 → **MISMATCH: compiled numbers below are not trustworthy**; compile 28.5 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 2.72 | 0.30 | 3.55 | 6.57 |
| compiled graph | 222 | 2.01 | 0.27 | 0.16 | 2.45 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 6.67 | 2.86 | 3.81 | 2.33× |
| 16 | 2048 | 11.95 | 8.81 | 3.14 | 1.36× |
| 16 | 8192 | 26.47 | 23.42 | 3.05 | 1.13× |

- max-autotune-no-cudagraphs at B=1, C=128: 2.73 ms (compile 36.5 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 8.43 ms (compile 71.6 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| spec bandwidth | 2,039 GB/s |
| copy bandwidth (read + write) | 1,737 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 1,527 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 2.009 ms |
| effective weight bandwidth | 1,228 GB/s (80% of read peak) |
| δ (M2) | 3.182 ± 0.124 ms |
| γ (M2) | 0.0487 ± 0.0053 ms per sequence |
| α′ (M2) | 0.1244 ± 0.0019 µs per cached token |
| α′-implied KV bandwidth | 1,581 GB/s (104% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 2.144, 16x512: 0.203, 16x2048: 0.162, 16x8192: 0.152 |
| small kernels (copy + other), graph, B=1, C=128 | 0.163 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.413 ms |
| X1 crossover (weight time ÷ α′) | 16,159 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.137 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.148 | 0.137 | 1468 |
| 16 | 0.149 | 0.149 | 1348 |
| 64 | 0.157 | 0.159 | 1265 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 220 | 2.02 | 0.32 | 0.16 | 0.00 | 2.51 |
| compiled_graph 1x128 | 222 | 2.01 | 0.27 | 0.16 | 0.00 | 2.45 |
| compiled 16x512 | 412 | 3.17 | 1.70 | 0.27 | 0.06 | 5.20 |
| compiled_graph 16x512 | 414 | 3.05 | 1.67 | 0.24 | 0.06 | 5.01 |
| compiled 16x2048 | 412 | 3.13 | 5.33 | 0.27 | 0.06 | 8.80 |
| compiled_graph 16x2048 | 414 | 3.02 | 5.31 | 0.24 | 0.06 | 8.63 |
| compiled 16x8192 | 412 | 3.13 | 19.92 | 0.27 | 0.06 | 23.38 |
| compiled_graph 16x8192 | 414 | 3.02 | 19.90 | 0.24 | 0.06 | 23.22 |
| graph_static 1x128 | 1381 | 2.72 | 0.30 | 1.35 | 2.20 | 6.57 |

Top kernels (first row): `triton_red_fused_mm_5` 0.79 ms; `triton_red_fused_mm_6` 0.51 ms; `triton_red_fused_copy_mm_1` 0.48 ms; `triton_red_fused_bmm_3` 0.24 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.18 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.09 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.323 | 2.5263 | 78 | 5% |
| compiled_graph 1x128 | 0.274 | 2.1442 | 92 | 6% |
| compiled 16x512 | 1.697 | 0.2071 | 949 | 62% |
| compiled_graph 16x512 | 1.667 | 0.2035 | 966 | 63% |
| compiled 16x2048 | 5.334 | 0.1628 | 1208 | 79% |
| compiled_graph 16x2048 | 5.311 | 0.1621 | 1213 | 79% |
| compiled 16x8192 | 19.918 | 0.1520 | 1294 | 85% |
| compiled_graph 16x8192 | 19.901 | 0.1518 | 1295 | 85% |
| graph_static 1x128 | 0.297 | 2.3183 | 85 | 6% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.259 ± 1.159 | 0.2496 ± 0.0355 | 0.001573 ± 0.000278 |  | 0.6211 | 157.2 | 15.65 |
| M2 | 3.182 ± 0.124 | 0.0487 ± 0.0053 |  | 0.1244 ± 0.0019 | 0.9935 | -38.2 | 3.15 |
| M3 | 3.164 ± 0.164 | 0.0492 ± 0.0062 | 0.000008 ± 0.000048 | 0.1241 ± 0.0025 | 0.9935 | -36.3 | 3.13 |

M2 KV slope → effective KV bandwidth 1581 GB/s (104% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.86 | 2.02 | 0.84 | 3.41× |
| 2 | 128 | 2.99 | 2.02 | 0.97 | 3.08× |
| 4 | 128 | 3.60 | 2.02 | 1.58 | 2.28× |
| 8 | 128 | 3.96 | 2.02 | 1.94 | 2.04× |
| 16 | 128 | 4.16 | 2.02 | 2.14 | 1.94× |
| 32 | 128 | 5.15 | 2.02 | 3.13 | 1.65× |
| 64 | 128 | 7.51 | 2.02 | 5.49 | 1.37× |
| 1 | 256 | 2.79 | 2.02 | 0.77 | 3.61× |
| 2 | 256 | 3.38 | 2.02 | 1.36 | 2.48× |
| 4 | 256 | 3.66 | 2.02 | 1.64 | 2.24× |
| 8 | 256 | 3.88 | 2.02 | 1.86 | 2.09× |
| 16 | 256 | 4.45 | 2.02 | 2.43 | 1.83× |
| 32 | 256 | 5.79 | 2.02 | 3.77 | 1.54× |
| 64 | 256 | 8.40 | 2.02 | 6.37 | 1.32× |
| 1 | 512 | 2.90 | 2.02 | 0.88 | 3.29× |
| 2 | 512 | 3.15 | 2.02 | 1.12 | 2.80× |
| 4 | 512 | 3.73 | 2.02 | 1.71 | 2.18× |
| 8 | 512 | 4.41 | 2.02 | 2.39 | 1.85× |
| 16 | 512 | 5.20 | 2.02 | 3.18 | 1.64× |
| 32 | 512 | 6.77 | 2.02 | 4.75 | 1.43× |
| 64 | 512 | 10.46 | 2.02 | 8.44 | 1.24× |
| 1 | 1024 | 3.04 | 2.02 | 1.01 | 2.99× |
| 2 | 1024 | 3.64 | 2.02 | 1.61 | 2.25× |
| 4 | 1024 | 3.83 | 2.02 | 1.80 | 2.12× |
| 8 | 1024 | 4.27 | 2.02 | 2.25 | 1.90× |
| 16 | 1024 | 7.27 | 2.02 | 5.25 | 1.39× |
| 32 | 1024 | 8.75 | 2.02 | 6.73 | 1.30× |
| 64 | 1024 | 14.35 | 2.02 | 12.32 | 1.16× |
| 1 | 2048 | 3.18 | 2.02 | 1.16 | 2.74× |
| 2 | 2048 | 3.84 | 2.02 | 1.82 | 2.11× |
| 4 | 2048 | 4.59 | 2.02 | 2.57 | 1.79× |
| 8 | 2048 | 5.20 | 2.02 | 3.18 | 1.64× |
| 16 | 2048 | 8.81 | 2.02 | 6.79 | 1.30× |
| 32 | 2048 | 12.67 | 2.02 | 10.65 | 1.19× |
| 64 | 2048 | 22.29 | 2.02 | 20.27 | 1.10× |
| 1 | 4096 | 3.39 | 2.02 | 1.37 | 2.47× |
| 2 | 4096 | 4.49 | 2.02 | 2.47 | 1.82× |
| 4 | 4096 | 5.52 | 2.02 | 3.50 | 1.58× |
| 8 | 4096 | 7.05 | 2.02 | 5.03 | 1.40× |
| 16 | 4096 | 13.59 | 2.02 | 11.57 | 1.17× |
| 32 | 4096 | 20.61 | 2.02 | 18.58 | 1.11× |
| 64 | 4096 | 38.21 | 2.02 | 36.19 | 1.06× |
| 1 | 8192 | 3.84 | 2.02 | 1.82 | 2.11× |
| 2 | 8192 | 4.97 | 2.02 | 2.95 | 1.69× |
| 4 | 8192 | 6.88 | 2.02 | 4.86 | 1.42× |
| 8 | 8192 | 10.76 | 2.02 | 8.74 | 1.23× |
| 16 | 8192 | 23.42 | 2.02 | 21.39 | 1.09× |
| 32 | 8192 | 36.79 | 2.02 | 34.76 | 1.06× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

