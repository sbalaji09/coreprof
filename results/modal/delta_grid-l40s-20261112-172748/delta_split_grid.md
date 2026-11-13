# Week 5 v1: CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 2.85 ms (spec), 3.89 ms (measured copy peak 635 GB/s).

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| eager_static 1x128 | 1379 | 3.97 | 0.00 | 0.62 | 1.19 | 5.77 |
| graph_static 1x128 | 1381 | 3.93 | 0.00 | 0.56 | 1.07 | 5.57 |
| eager_static 16x2048 | 1354 | 12.81 | 0.00 | 0.67 | 1.61 | 15.09 |
| graph_static 16x2048 | 1356 | 12.77 | 0.00 | 0.60 | 1.45 | 14.82 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 3.79 ms; `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<` 0.22 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.21 ms; `void at::native::unrolled_elementwise_kernel<at::native::dir` 0.20 ms; `void at::native::elementwise_kernel<128, 4, at::native::gpu_` 0.18 ms; `void at::native::elementwise_kernel<128, 2, at::native::gpu_` 0.14 ms

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 3.520 ± 1.561 | 0.2821 ± 0.0501 | 0.002086 ± 0.000392 |  | 0.5366 | 174.8 | 20.29 |
| M2 | 6.134 ± 0.032 | 0.0152 ± 0.0014 |  | 0.2648 ± 0.0008 | 0.9997 | -164.0 | 0.37 |
| M3 | 6.132 ± 0.041 | 0.0152 ± 0.0016 | 0.000001 ± 0.000013 | 0.2648 ± 0.0010 | 0.9997 | -162.0 | 0.37 |

M2 KV slope → effective KV bandwidth 742 GB/s (117% of measured copy peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.82 | 3.97 | 1.85 | 3.14× |
| 2 | 128 | 6.28 | 3.97 | 2.31 | 2.72× |
| 4 | 128 | 6.35 | 3.97 | 2.39 | 2.66× |
| 8 | 128 | 6.56 | 3.97 | 2.59 | 2.53× |
| 16 | 128 | 7.06 | 3.97 | 3.09 | 2.28× |
| 32 | 128 | 7.77 | 3.97 | 3.80 | 2.04× |
| 64 | 128 | 9.11 | 3.97 | 5.14 | 1.77× |
| 1 | 256 | 5.89 | 3.97 | 1.92 | 3.06× |
| 2 | 256 | 6.32 | 3.97 | 2.35 | 2.69× |
| 4 | 256 | 6.52 | 3.97 | 2.55 | 2.55× |
| 8 | 256 | 6.93 | 3.97 | 2.97 | 2.34× |
| 16 | 256 | 7.58 | 3.97 | 3.61 | 2.10× |
| 32 | 256 | 8.87 | 3.97 | 4.91 | 1.81× |
| 64 | 256 | 11.37 | 3.97 | 7.40 | 1.54× |
| 1 | 512 | 5.93 | 3.97 | 1.97 | 3.02× |
| 2 | 512 | 6.46 | 3.97 | 2.50 | 2.59× |
| 4 | 512 | 6.84 | 3.97 | 2.87 | 2.38× |
| 8 | 512 | 7.38 | 3.97 | 3.42 | 2.16× |
| 16 | 512 | 8.65 | 3.97 | 4.69 | 1.85× |
| 32 | 512 | 11.09 | 3.97 | 7.12 | 1.56× |
| 64 | 512 | 15.67 | 3.97 | 11.71 | 1.34× |
| 1 | 1024 | 6.10 | 3.97 | 2.13 | 2.86× |
| 2 | 1024 | 6.75 | 3.97 | 2.78 | 2.43× |
| 4 | 1024 | 7.39 | 3.97 | 3.42 | 2.16× |
| 8 | 1024 | 8.67 | 3.97 | 4.70 | 1.84× |
| 16 | 1024 | 10.80 | 3.97 | 6.83 | 1.58× |
| 32 | 1024 | 15.67 | 3.97 | 11.70 | 1.34× |
| 64 | 1024 | 24.39 | 3.97 | 20.42 | 1.19× |
| 1 | 2048 | 6.42 | 3.97 | 2.46 | 2.61× |
| 2 | 2048 | 7.31 | 3.97 | 3.35 | 2.19× |
| 4 | 2048 | 8.37 | 3.97 | 4.41 | 1.90× |
| 8 | 2048 | 10.63 | 3.97 | 6.66 | 1.60× |
| 16 | 2048 | 15.08 | 3.97 | 11.12 | 1.36× |
| 32 | 2048 | 24.00 | 3.97 | 20.03 | 1.20× |
| 64 | 2048 | 41.70 | 3.97 | 37.73 | 1.11× |
| 1 | 4096 | 6.92 | 3.97 | 2.96 | 2.34× |
| 2 | 4096 | 8.39 | 3.97 | 4.43 | 1.90× |
| 4 | 4096 | 10.53 | 3.97 | 6.57 | 1.60× |
| 8 | 4096 | 15.18 | 3.97 | 11.21 | 1.35× |
| 16 | 4096 | 23.73 | 3.97 | 19.76 | 1.20× |
| 32 | 4096 | 41.38 | 3.97 | 37.42 | 1.11× |
| 1 | 8192 | 8.27 | 3.97 | 4.30 | 1.92× |
| 2 | 8192 | 10.68 | 3.97 | 6.71 | 1.59× |
| 4 | 8192 | 14.86 | 3.97 | 10.89 | 1.36× |
| 8 | 8192 | 23.63 | 3.97 | 19.66 | 1.20× |
| 16 | 8192 | 40.99 | 3.97 | 37.02 | 1.11× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

