# Week 5 v1: CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.74 ms (spec), 0.82 ms (measured copy peak 3018 GB/s), 0.86 ms (measured read peak 2874 GB/s).

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,018 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,874 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 1.491 ms |
| effective weight bandwidth | 1,655 GB/s (58% of read peak) |
| δ (M2) | 4.620 ± 0.045 ms |
| γ (M2) | 0.0207 ± 0.0019 ms per sequence |
| α′ (M2) | 0.0758 ± 0.0007 µs per cached token |
| α′-implied KV bandwidth | 2,594 GB/s (90% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.424, 16x512: 0.099, 16x2048: 0.081, 16x8192: 0.075 |
| small kernels (copy + other), graph, B=1, C=128 | 1.887 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.669 ms |
| X1 crossover (weight time ÷ α′) | 19,666 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.094 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.113 | 0.094 | 2137 |
| 16 | 0.100 | 0.094 | 2133 |
| 64 | 0.108 | 0.100 | 2007 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| eager_static 1x128 | 1379 | 1.51 | 0.19 | 0.70 | 1.32 | 3.72 |
| graph_static 1x128 | 1381 | 1.49 | 0.18 | 0.65 | 1.24 | 3.56 |
| eager_static 16x512 | 1474 | 1.75 | 0.81 | 0.82 | 1.92 | 5.30 |
| graph_static 16x512 | 1476 | 1.76 | 0.81 | 0.77 | 1.82 | 5.15 |
| eager_static 16x2048 | 1474 | 1.75 | 2.67 | 0.81 | 1.90 | 7.12 |
| graph_static 16x2048 | 1476 | 1.77 | 2.65 | 0.77 | 1.80 | 6.99 |
| eager_static 16x8192 | 1474 | 1.75 | 9.78 | 0.82 | 1.91 | 14.26 |
| graph_static 16x8192 | 1476 | 1.76 | 9.77 | 0.77 | 1.81 | 14.12 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 0.91 ms; `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` 0.60 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.26 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.25 ms; `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<` 0.23 ms; `void at::native::elementwise_kernel<128, 4, at::native::gpu_` 0.21 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| eager_static 1x128 | 0.188 | 1.4679 | 134 | 5% |
| graph_static 1x128 | 0.182 | 1.4243 | 138 | 5% |
| eager_static 16x512 | 0.813 | 0.0992 | 1982 | 69% |
| graph_static 16x512 | 0.811 | 0.0990 | 1986 | 69% |
| eager_static 16x2048 | 2.666 | 0.0814 | 2416 | 84% |
| graph_static 16x2048 | 2.655 | 0.0810 | 2427 | 84% |
| eager_static 16x8192 | 9.779 | 0.0746 | 2635 | 92% |
| graph_static 16x8192 | 9.765 | 0.0745 | 2639 | 92% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 2.892 ± 0.714 | 0.1428 ± 0.0219 | 0.000937 ± 0.000171 |  | 0.5934 | 110.7 | 10.33 |
| M2 | 4.620 ± 0.045 | 0.0207 ± 0.0019 |  | 0.0758 ± 0.0007 | 0.9976 | -135.4 | 0.55 |
| M3 | 4.693 ± 0.057 | 0.0186 ± 0.0021 | -0.000033 ± 0.000017 | 0.0769 ± 0.0009 | 0.9978 | -137.5 | 0.49 |

M2 KV slope → effective KV bandwidth 2594 GB/s (90% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.23 | 1.51 | 2.72 | 1.55× |
| 2 | 128 | 4.70 | 1.51 | 3.19 | 1.47× |
| 4 | 128 | 4.81 | 1.51 | 3.30 | 1.46× |
| 8 | 128 | 5.06 | 1.51 | 3.55 | 1.42× |
| 16 | 128 | 5.36 | 1.51 | 3.85 | 1.39× |
| 32 | 128 | 5.75 | 1.51 | 4.24 | 1.36× |
| 64 | 128 | 6.41 | 1.51 | 4.90 | 1.31× |
| 1 | 256 | 4.24 | 1.51 | 2.73 | 1.55× |
| 2 | 256 | 4.72 | 1.51 | 3.21 | 1.47× |
| 4 | 256 | 4.89 | 1.51 | 3.38 | 1.45× |
| 8 | 256 | 5.14 | 1.51 | 3.63 | 1.42× |
| 16 | 256 | 5.51 | 1.51 | 4.01 | 1.38× |
| 32 | 256 | 6.08 | 1.51 | 4.57 | 1.33× |
| 64 | 256 | 7.05 | 1.51 | 5.54 | 1.27× |
| 1 | 512 | 4.27 | 1.51 | 2.76 | 1.55× |
| 2 | 512 | 4.80 | 1.51 | 3.29 | 1.46× |
| 4 | 512 | 4.95 | 1.51 | 3.44 | 1.44× |
| 8 | 512 | 5.23 | 1.51 | 3.72 | 1.41× |
| 16 | 512 | 5.81 | 1.51 | 4.30 | 1.35× |
| 32 | 512 | 6.71 | 1.51 | 5.20 | 1.29× |
| 64 | 512 | 8.31 | 1.51 | 6.81 | 1.22× |
| 1 | 1024 | 4.33 | 1.51 | 2.83 | 1.53× |
| 2 | 1024 | 4.88 | 1.51 | 3.37 | 1.45× |
| 4 | 1024 | 5.12 | 1.51 | 3.61 | 1.42× |
| 8 | 1024 | 5.60 | 1.51 | 4.09 | 1.37× |
| 16 | 1024 | 6.49 | 1.51 | 4.99 | 1.30× |
| 32 | 1024 | 8.00 | 1.51 | 6.49 | 1.23× |
| 64 | 1024 | 10.80 | 1.51 | 9.29 | 1.16× |
| 1 | 2048 | 4.46 | 1.51 | 2.95 | 1.51× |
| 2 | 2048 | 5.03 | 1.51 | 3.52 | 1.43× |
| 4 | 2048 | 5.39 | 1.51 | 3.88 | 1.39× |
| 8 | 2048 | 6.11 | 1.51 | 4.60 | 1.33× |
| 16 | 2048 | 7.75 | 1.51 | 6.24 | 1.24× |
| 32 | 2048 | 10.39 | 1.51 | 8.89 | 1.17× |
| 64 | 2048 | 15.47 | 1.51 | 13.96 | 1.11× |
| 1 | 4096 | 4.71 | 1.51 | 3.20 | 1.47× |
| 2 | 4096 | 5.63 | 1.51 | 4.12 | 1.37× |
| 4 | 4096 | 5.93 | 1.51 | 4.42 | 1.34× |
| 8 | 4096 | 7.13 | 1.51 | 5.63 | 1.27× |
| 16 | 4096 | 10.12 | 1.51 | 8.61 | 1.18× |
| 32 | 4096 | 15.09 | 1.51 | 13.59 | 1.11× |
| 64 | 4096 | 25.81 | 1.51 | 24.30 | 1.06× |
| 1 | 8192 | 5.02 | 1.51 | 3.51 | 1.43× |
| 2 | 8192 | 5.83 | 1.51 | 4.32 | 1.35× |
| 4 | 8192 | 6.96 | 1.51 | 5.46 | 1.28× |
| 8 | 8192 | 9.20 | 1.51 | 7.69 | 1.20× |
| 16 | 8192 | 14.85 | 1.51 | 13.35 | 1.11× |
| 32 | 8192 | 25.46 | 1.51 | 23.96 | 1.06× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

