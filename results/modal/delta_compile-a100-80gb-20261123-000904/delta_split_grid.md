# Week 5 v1: torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA A100-SXM4-80GB

Predicted weight fetch 1.21 ms (spec), 1.42 ms (measured copy peak 1738 GB/s), 1.60 ms (measured read peak 1545 GB/s).

## torch.compile vs uncompiled (same container)

Compiled vs eager static step (same cache, B=2, C=64): relative error 3.00e-02, gate error 1.56e-01 → **MISMATCH: compiled numbers below are not trustworthy**; compile 26.5 s, graph breaks 0.

| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |
|---|---:|---:|---:|---:|---:|
| uncompiled graph | 1381 | 2.70 | 0.29 | 3.51 | 6.50 |
| compiled graph | 342 | 2.37 | 0.27 | 0.25 | 2.90 |

| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.62 | 3.36 | 2.26 | 1.67× |
| 16 | 2048 | 11.70 | 8.91 | 2.80 | 1.31× |
| 16 | 8192 | 26.53 | 24.36 | 2.16 | 1.09× |

- max-autotune-no-cudagraphs at B=1, C=128: 3.83 ms (compile 78.0 s)
- max-autotune-no-cudagraphs at B=16, C=2048: 8.58 ms (compile 73.1 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| spec bandwidth | 2,039 GB/s |
| copy bandwidth (read + write) | 1,738 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 1,545 GB/s |
| weight time (GEMV/GEMM kernels, graph, B=1, C=128) | 2.371 ms |
| effective weight bandwidth | 1,040 GB/s (67% of read peak) |
| δ (M2) | 3.050 ± 0.104 ms |
| γ (M2) | 0.0591 ± 0.0045 ms per sequence |
| α′ (M2) | 0.1380 ± 0.0016 µs per cached token |
| α′-implied KV bandwidth | 1,425 GB/s (92% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 2.148, 16x512: 0.203, 16x2048: 0.161, 16x8192: 0.157 |
| small kernels (copy + other), graph, B=1, C=128 | 0.255 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.464 ms |
| X1 crossover (weight time ÷ α′) | 17,182 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.144 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.157 | 0.144 | 1402 |
| 16 | 0.161 | 0.154 | 1308 |
| 64 | 0.167 | 0.162 | 1246 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| compiled 1x128 | 340 | 2.47 | 0.32 | 0.24 | 0.06 | 3.09 |
| compiled_graph 1x128 | 342 | 2.37 | 0.27 | 0.21 | 0.05 | 2.90 |
| compiled 16x512 | 412 | 3.35 | 1.68 | 0.25 | 0.06 | 5.34 |
| compiled_graph 16x512 | 414 | 3.24 | 1.66 | 0.22 | 0.05 | 5.17 |
| compiled 16x2048 | 412 | 3.32 | 5.33 | 0.25 | 0.06 | 8.96 |
| compiled_graph 16x2048 | 414 | 3.20 | 5.29 | 0.22 | 0.05 | 8.76 |
| compiled 16x8192 | 412 | 3.41 | 20.59 | 0.26 | 0.06 | 24.32 |
| compiled_graph 16x8192 | 414 | 3.29 | 20.55 | 0.22 | 0.06 | 24.12 |
| graph_static 1x128 | 1381 | 2.70 | 0.29 | 1.34 | 2.16 | 6.50 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 1.62 ms; `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn` 0.84 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.18 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3` 0.09 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.09 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6` 0.09 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| compiled 1x128 | 0.325 | 2.5355 | 78 | 5% |
| compiled_graph 1x128 | 0.275 | 2.1482 | 92 | 6% |
| compiled 16x512 | 1.682 | 0.2053 | 958 | 62% |
| compiled_graph 16x512 | 1.659 | 0.2025 | 971 | 63% |
| compiled 16x2048 | 5.328 | 0.1626 | 1209 | 78% |
| compiled_graph 16x2048 | 5.287 | 0.1613 | 1219 | 79% |
| compiled 16x8192 | 20.587 | 0.1571 | 1252 | 81% |
| compiled_graph 16x8192 | 20.545 | 0.1567 | 1254 | 81% |
| graph_static 1x128 | 0.291 | 2.2727 | 87 | 6% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | -0.099 ± 1.302 | 0.2814 ± 0.0399 | 0.001707 ± 0.000312 |  | 0.6163 | 168.3 | 18.12 |
| M2 | 3.050 ± 0.104 | 0.0591 ± 0.0045 |  | 0.1380 ± 0.0016 | 0.9963 | -54.9 | 2.28 |
| M3 | 3.177 ± 0.134 | 0.0555 ± 0.0050 | -0.000058 ± 0.000039 | 0.1399 ± 0.0020 | 0.9965 | -55.2 | 2.43 |

M2 KV slope → effective KV bandwidth 1425 GB/s (92% of measured read peak).

## Upper bound if weights were free (weight term = GEMM kernel time (profiled))

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.36 | 2.47 | 0.90 | 3.75× |
| 2 | 128 | 2.98 | 2.47 | 0.52 | 5.76× |
| 4 | 128 | 3.25 | 2.47 | 0.78 | 4.15× |
| 8 | 128 | 3.60 | 2.47 | 1.14 | 3.17× |
| 16 | 128 | 4.27 | 2.47 | 1.81 | 2.37× |
| 32 | 128 | 5.57 | 2.47 | 3.10 | 1.80× |
| 64 | 128 | 8.19 | 2.47 | 5.73 | 1.43× |
| 1 | 256 | 2.99 | 2.47 | 0.53 | 5.70× |
| 2 | 256 | 3.37 | 2.47 | 0.91 | 3.72× |
| 4 | 256 | 3.71 | 2.47 | 1.24 | 2.99× |
| 8 | 256 | 4.20 | 2.47 | 1.73 | 2.42× |
| 16 | 256 | 4.57 | 2.47 | 2.10 | 2.17× |
| 32 | 256 | 6.08 | 2.47 | 3.62 | 1.68× |
| 64 | 256 | 9.16 | 2.47 | 6.70 | 1.37× |
| 1 | 512 | 3.43 | 2.47 | 0.96 | 3.56× |
| 2 | 512 | 3.50 | 2.47 | 1.03 | 3.39× |
| 4 | 512 | 3.43 | 2.47 | 0.97 | 3.55× |
| 8 | 512 | 3.91 | 2.47 | 1.44 | 2.71× |
| 16 | 512 | 5.99 | 2.47 | 3.52 | 1.70× |
| 32 | 512 | 7.11 | 2.47 | 4.65 | 1.53× |
| 64 | 512 | 11.36 | 2.47 | 8.89 | 1.28× |
| 1 | 1024 | 3.55 | 2.47 | 1.08 | 3.28× |
| 2 | 1024 | 3.55 | 2.47 | 1.08 | 3.28× |
| 4 | 1024 | 4.02 | 2.47 | 1.55 | 2.59× |
| 8 | 1024 | 4.82 | 2.47 | 2.35 | 2.05× |
| 16 | 1024 | 6.40 | 2.47 | 3.93 | 1.63× |
| 32 | 1024 | 9.24 | 2.47 | 6.78 | 1.36× |
| 64 | 1024 | 15.78 | 2.47 | 13.31 | 1.19× |
| 1 | 2048 | 3.70 | 2.47 | 1.24 | 3.00× |
| 2 | 2048 | 3.81 | 2.47 | 1.34 | 2.84× |
| 4 | 2048 | 4.16 | 2.47 | 1.70 | 2.45× |
| 8 | 2048 | 5.31 | 2.47 | 2.85 | 1.87× |
| 16 | 2048 | 8.91 | 2.47 | 6.44 | 1.38× |
| 32 | 2048 | 13.72 | 2.47 | 11.25 | 1.22× |
| 64 | 2048 | 24.65 | 2.47 | 22.19 | 1.11× |
| 1 | 4096 | 3.89 | 2.47 | 1.42 | 2.74× |
| 2 | 4096 | 4.01 | 2.47 | 1.54 | 2.60× |
| 4 | 4096 | 5.09 | 2.47 | 2.62 | 1.94× |
| 8 | 4096 | 7.17 | 2.47 | 4.70 | 1.52× |
| 16 | 4096 | 13.92 | 2.47 | 11.45 | 1.22× |
| 32 | 4096 | 22.71 | 2.47 | 20.25 | 1.12× |
| 64 | 4096 | 42.79 | 2.47 | 40.32 | 1.06× |
| 1 | 8192 | 3.99 | 2.47 | 1.52 | 2.62× |
| 2 | 8192 | 5.27 | 2.47 | 2.80 | 1.88× |
| 4 | 8192 | 7.01 | 2.47 | 4.54 | 1.54× |
| 8 | 8192 | 10.76 | 2.47 | 8.30 | 1.30× |
| 16 | 8192 | 24.36 | 2.47 | 21.89 | 1.11× |
| 32 | 8192 | 41.01 | 2.47 | 38.54 | 1.06× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

