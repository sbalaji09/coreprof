# Week 5 v1: CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 2.85 ms (spec), 3.75 ms (measured copy peak 659 GB/s), 3.57 ms (measured read peak 691 GB/s).

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 659 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 691 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 3.774 ms |
| effective weight bandwidth | 654 GB/s (95% of read peak) |
| δ (M2) | 6.380 ± 0.042 ms |
| γ (M2) | 0.0155 ± 0.0018 ms per sequence |
| α′ (M2) | 0.2657 ± 0.0010 µs per cached token |
| α′-implied KV bandwidth | 740 GB/s (107% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.216, 16x512: 0.275, 16x2048: 0.268, 16x8192: 0.262 |
| small kernels (copy + other), graph, B=1, C=128 | 1.614 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.328 ms |
| X1 crossover (weight time ÷ α′) | 14,201 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.330 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.330 | 0.330 | 610 |
| 16 | 0.327 | 0.302 | 667 |
| 64 | 0.341 | 0.341 | 590 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| eager_static 1x128 | 1379 | 3.79 | 0.17 | 0.61 | 1.18 | 5.76 |
| graph_static 1x128 | 1381 | 3.77 | 0.16 | 0.56 | 1.05 | 5.54 |
| eager_static 16x512 | 1354 | 4.11 | 2.26 | 0.67 | 1.57 | 8.61 |
| graph_static 16x512 | 1356 | 4.07 | 2.26 | 0.60 | 1.43 | 8.36 |
| eager_static 16x2048 | 1354 | 4.03 | 8.78 | 0.67 | 1.58 | 15.06 |
| graph_static 16x2048 | 1356 | 3.98 | 8.78 | 0.60 | 1.44 | 14.79 |
| eager_static 16x8192 | 1354 | 4.03 | 34.42 | 0.67 | 1.58 | 40.70 |
| graph_static 16x8192 | 1356 | 3.98 | 34.40 | 0.60 | 1.43 | 40.41 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 3.79 ms; `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<` 0.21 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.21 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.20 ms; `void at::native::elementwise_kernel<128, 4, at::native::gpu_` 0.18 ms; `void at::native::elementwise_kernel<128, 2, at::native::gpu_` 0.14 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| eager_static 1x128 | 0.173 | 1.3538 | 145 | 21% |
| graph_static 1x128 | 0.156 | 1.2158 | 162 | 23% |
| eager_static 16x512 | 2.260 | 0.2758 | 713 | 103% |
| graph_static 16x512 | 2.257 | 0.2755 | 714 | 103% |
| eager_static 16x2048 | 8.782 | 0.2680 | 734 | 106% |
| graph_static 16x2048 | 8.780 | 0.2679 | 734 | 106% |
| eager_static 16x8192 | 34.425 | 0.2626 | 749 | 108% |
| graph_static 16x8192 | 34.400 | 0.2625 | 749 | 108% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 3.735 ± 1.562 | 0.2836 ± 0.0502 | 0.002102 ± 0.000392 |  | 0.5393 | 174.9 | 20.29 |
| M2 | 6.380 ± 0.042 | 0.0155 ± 0.0018 |  | 0.2657 ± 0.0010 | 0.9995 | -138.2 | 0.61 |
| M3 | 6.349 ± 0.054 | 0.0165 ± 0.0022 | 0.000016 ± 0.000017 | 0.2649 ± 0.0013 | 0.9995 | -137.2 | 0.58 |

M2 KV slope → effective KV bandwidth 740 GB/s (107% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.87 | 3.79 | 2.08 | 2.83× |
| 2 | 128 | 6.43 | 3.79 | 2.63 | 2.44× |
| 4 | 128 | 6.49 | 3.79 | 2.70 | 2.40× |
| 8 | 128 | 6.71 | 3.79 | 2.92 | 2.30× |
| 16 | 128 | 7.15 | 3.79 | 3.36 | 2.13× |
| 32 | 128 | 7.91 | 3.79 | 4.12 | 1.92× |
| 64 | 128 | 9.26 | 3.79 | 5.47 | 1.69× |
| 1 | 256 | 5.86 | 3.79 | 2.06 | 2.84× |
| 2 | 256 | 6.70 | 3.79 | 2.91 | 2.30× |
| 4 | 256 | 6.64 | 3.79 | 2.84 | 2.33× |
| 8 | 256 | 6.99 | 3.79 | 3.19 | 2.19× |
| 16 | 256 | 7.67 | 3.79 | 3.88 | 1.98× |
| 32 | 256 | 8.96 | 3.79 | 5.16 | 1.73× |
| 64 | 256 | 11.59 | 3.79 | 7.80 | 1.49× |
| 1 | 512 | 6.28 | 3.79 | 2.49 | 2.53× |
| 2 | 512 | 6.68 | 3.79 | 2.88 | 2.32× |
| 4 | 512 | 6.99 | 3.79 | 3.20 | 2.19× |
| 8 | 512 | 7.56 | 3.79 | 3.76 | 2.01× |
| 16 | 512 | 8.81 | 3.79 | 5.02 | 1.76× |
| 32 | 512 | 11.66 | 3.79 | 7.87 | 1.48× |
| 64 | 512 | 16.07 | 3.79 | 12.28 | 1.31× |
| 1 | 1024 | 6.44 | 3.79 | 2.65 | 2.43× |
| 2 | 1024 | 7.09 | 3.79 | 3.30 | 2.15× |
| 4 | 1024 | 7.73 | 3.79 | 3.94 | 1.96× |
| 8 | 1024 | 8.88 | 3.79 | 5.09 | 1.75× |
| 16 | 1024 | 11.25 | 3.79 | 7.46 | 1.51× |
| 32 | 1024 | 15.95 | 3.79 | 12.15 | 1.31× |
| 64 | 1024 | 24.89 | 3.79 | 21.09 | 1.18× |
| 1 | 2048 | 6.79 | 3.79 | 2.99 | 2.27× |
| 2 | 2048 | 7.73 | 3.79 | 3.94 | 1.96× |
| 4 | 2048 | 8.80 | 3.79 | 5.01 | 1.76× |
| 8 | 2048 | 11.00 | 3.79 | 7.21 | 1.53× |
| 16 | 2048 | 15.55 | 3.79 | 11.76 | 1.32× |
| 32 | 2048 | 24.49 | 3.79 | 20.70 | 1.18× |
| 64 | 2048 | 41.99 | 3.79 | 38.19 | 1.10× |
| 1 | 4096 | 7.27 | 3.79 | 3.48 | 2.09× |
| 2 | 4096 | 8.74 | 3.79 | 4.95 | 1.77× |
| 4 | 4096 | 10.90 | 3.79 | 7.11 | 1.53× |
| 8 | 4096 | 15.32 | 3.79 | 11.52 | 1.33× |
| 16 | 4096 | 24.06 | 3.79 | 20.27 | 1.19× |
| 32 | 4096 | 41.71 | 3.79 | 37.91 | 1.10× |
| 1 | 8192 | 8.47 | 3.79 | 4.67 | 1.81× |
| 2 | 8192 | 10.89 | 3.79 | 7.10 | 1.53× |
| 4 | 8192 | 15.35 | 3.79 | 11.56 | 1.33× |
| 8 | 8192 | 24.00 | 3.79 | 20.20 | 1.19× |
| 16 | 8192 | 41.12 | 3.79 | 37.33 | 1.10× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

