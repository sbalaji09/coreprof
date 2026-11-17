# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.74 ms (spec), 0.82 ms (measured copy peak 3005 GB/s), 0.87 ms (measured read peak 2849 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 2.91e-02, gate error 7.03e-02 → **MISMATCH: compiled numbers below are not trustworthy**; compile 28.8 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 1.51 | 0.19 | 1.89 | 3.58 |
| compiled graph | 342 | 1.52 | 0.18 | 0.21 | 1.91 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.51 | 2.14 | 2.36 | 2.10× |
| 16 | 2048 | 7.98 | 5.27 | 2.71 | 1.51× |
| 16 | 8192 | 15.12 | 12.61 | 2.51 | 1.20× |

- max-autotune-no-cudagraphs at B=1, C=128: 2.56 ms (compile 96.2 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 6.80 ms (compile 85.2 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,005 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,849 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 1.515 ms |
| effective weight bandwidth | 1,628 GB/s (57% of read peak) |
| δ (M2) | 2.125 ± 0.039 ms |
| γ (M2) | 0.0509 ± 0.0017 ms per sequence |
| α′ (M2) | 0.0767 ± 0.0006 µs per cached token |
| α′-implied KV bandwidth | 2,563 GB/s (90% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.445, 16x512: 0.098, 16x2048: 0.080, 16x8192: 0.074 |
| small kernels (copy + other), graph, B=1, C=128 | 0.211 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.232 ms |
| X1 crossover (weight time ÷ α′) | 19,753 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.101 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.142 | 0.101 | 1994 |
| 16 | 0.131 | 0.100 | 2009 |
| 64 | 0.137 | 0.106 | 1898 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 340 | 1.51 | 0.19 | 0.00 | 0.21 | 1.91 |
| compiled_graph 1x128 | 342 | 1.52 | 0.18 | 0.00 | 0.21 | 1.91 |
| compiled 16x512 | 411 | 2.14 | 0.81 | 0.00 | 0.23 | 3.18 |
| compiled_graph 16x512 | 413 | 2.11 | 0.80 | 0.00 | 0.22 | 3.14 |
| compiled 16x2048 | 411 | 2.16 | 2.70 | 0.00 | 0.23 | 5.08 |
| compiled_graph 16x2048 | 413 | 2.13 | 2.62 | 0.00 | 0.22 | 4.97 |
| compiled 16x8192 | 411 | 2.13 | 9.71 | 0.00 | 0.23 | 12.07 |
| compiled_graph 16x8192 | 413 | 2.11 | 9.66 | 0.00 | 0.22 | 11.99 |
| graph_static 1x128 | 1381 | 1.51 | 0.19 | 0.66 | 1.23 | 3.58 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 0.91 ms; `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` 0.60 ms; `triton_` 0.21 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.14 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms; `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStr` 0.00 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.192 | 1.5039 | 131 | 5% |
| compiled_graph 1x128 | 0.185 | 1.4448 | 136 | 5% |
| compiled 16x512 | 0.812 | 0.0991 | 1984 | 70% |
| compiled_graph 16x512 | 0.804 | 0.0981 | 2004 | 70% |
| compiled 16x2048 | 2.696 | 0.0823 | 2389 | 84% |
| compiled_graph 16x2048 | 2.618 | 0.0799 | 2461 | 86% |
| compiled 16x8192 | 9.706 | 0.0741 | 2655 | 93% |
| compiled_graph 16x8192 | 9.655 | 0.0737 | 2669 | 94% |
| graph_static 1x128 | 0.185 | 1.4459 | 136 | 5% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.377 ± 0.722 | 0.1745 ± 0.0221 | 0.000948 ± 0.000173 |  | 0.6507 | 111.8 | 10.56 |
| M2 | 2.125 ± 0.039 | 0.0509 ± 0.0017 |  | 0.0767 ± 0.0006 | 0.9984 | -148.0 | 0.75 |
| M3 | 2.200 ± 0.049 | 0.0488 ± 0.0018 | -0.000034 ± 0.000014 | 0.0778 ± 0.0007 | 0.9986 | -151.8 | 0.60 |

M2 KV slope → effective KV bandwidth 2563 GB/s (90% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.14 | 1.51 | 0.64 | 3.36× |
| 2 | 128 | 2.41 | 1.51 | 0.91 | 2.66× |
| 4 | 128 | 2.50 | 1.51 | 0.99 | 2.52× |
| 8 | 128 | 2.71 | 1.51 | 1.21 | 2.25× |
| 16 | 128 | 2.92 | 1.51 | 1.41 | 2.07× |
| 32 | 128 | 4.09 | 1.51 | 2.58 | 1.58× |
| 64 | 128 | 6.09 | 1.51 | 4.58 | 1.33× |
| 1 | 256 | 2.15 | 1.51 | 0.65 | 3.33× |
| 2 | 256 | 2.43 | 1.51 | 0.93 | 2.62× |
| 4 | 256 | 2.55 | 1.51 | 1.05 | 2.44× |
| 8 | 256 | 2.78 | 1.51 | 1.28 | 2.18× |
| 16 | 256 | 3.08 | 1.51 | 1.58 | 1.96× |
| 32 | 256 | 4.43 | 1.51 | 2.93 | 1.51× |
| 64 | 256 | 6.70 | 1.51 | 5.19 | 1.29× |
| 1 | 512 | 2.18 | 1.51 | 0.67 | 3.24× |
| 2 | 512 | 2.49 | 1.51 | 0.99 | 2.52× |
| 4 | 512 | 2.63 | 1.51 | 1.13 | 2.34× |
| 8 | 512 | 2.90 | 1.51 | 1.40 | 2.08× |
| 16 | 512 | 3.40 | 1.51 | 1.89 | 1.79× |
| 32 | 512 | 5.04 | 1.51 | 3.54 | 1.43× |
| 64 | 512 | 7.98 | 1.51 | 6.47 | 1.23× |
| 1 | 1024 | 2.24 | 1.51 | 0.73 | 3.05× |
| 2 | 1024 | 2.58 | 1.51 | 1.07 | 2.41× |
| 4 | 1024 | 2.75 | 1.51 | 1.24 | 2.21× |
| 8 | 1024 | 3.18 | 1.51 | 1.67 | 1.90× |
| 16 | 1024 | 4.01 | 1.51 | 2.50 | 1.60× |
| 32 | 1024 | 6.29 | 1.51 | 4.78 | 1.31× |
| 64 | 1024 | 10.37 | 1.51 | 8.86 | 1.17× |
| 1 | 2048 | 2.33 | 1.51 | 0.83 | 2.82× |
| 2 | 2048 | 2.71 | 1.51 | 1.20 | 2.25× |
| 4 | 2048 | 3.02 | 1.51 | 1.52 | 1.99× |
| 8 | 2048 | 3.69 | 1.51 | 2.19 | 1.69× |
| 16 | 2048 | 5.27 | 1.51 | 3.76 | 1.40× |
| 32 | 2048 | 8.74 | 1.51 | 7.23 | 1.21× |
| 64 | 2048 | 15.21 | 1.51 | 13.71 | 1.11× |
| 1 | 4096 | 2.61 | 1.51 | 1.10 | 2.37× |
| 2 | 4096 | 3.33 | 1.51 | 1.82 | 1.83× |
| 4 | 4096 | 3.56 | 1.51 | 2.05 | 1.73× |
| 8 | 4096 | 4.77 | 1.51 | 3.26 | 1.46× |
| 16 | 4096 | 7.66 | 1.51 | 6.15 | 1.24× |
| 32 | 4096 | 13.71 | 1.51 | 12.20 | 1.12× |
| 64 | 4096 | 25.76 | 1.51 | 24.25 | 1.06× |
| 1 | 8192 | 2.90 | 1.51 | 1.40 | 2.08× |
| 2 | 8192 | 3.50 | 1.51 | 1.99 | 1.76× |
| 4 | 8192 | 4.60 | 1.51 | 3.10 | 1.49× |
| 8 | 8192 | 6.81 | 1.51 | 5.31 | 1.28× |
| 16 | 8192 | 12.61 | 1.51 | 11.11 | 1.14× |
| 32 | 8192 | 24.29 | 1.51 | 22.78 | 1.07× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

