# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.37 ms (spec), 0.41 ms (measured copy peak 3017 GB/s), 0.43 ms (measured read peak 2864 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.4 %, exit gate 0.438, top-1 next token agrees 94 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.07 | 3.05 | -0.98 | 0.68× |
| 16 | 2048 | 5.15 | 6.34 | -1.19 | 0.81× |
| 16 | 8192 | 12.40 | 13.58 | -1.18 | 0.91× |

Weight time at B=1, C=128: bf16 1.502 ms (1642 GB/s) → int8 ≈ 2.483 ms (497 GB/s), i.e. 1.65× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 3.30 ms (compile 77.6 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 6.06 ms (compile 58.2 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,017 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,864 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 2.483 ms |
| effective weight bandwidth | 497 GB/s (17% of read peak) |
| δ (M2) | 3.147 ± 0.034 ms |
| γ (M2) | 0.0501 ± 0.0015 ms per sequence |
| α′ (M2) | 0.0757 ± 0.0005 µs per cached token |
| α′-implied KV bandwidth | 2,598 GB/s (91% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.724, 16x512: 0.114, 16x2048: 0.084, 16x8192: 0.075 |
| small kernels (copy + other), graph, B=1, C=128 | 1.407 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.237 ms |
| X1 crossover (weight time ÷ α′) | 32,803 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.096 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.116 | 0.096 | 2106 |
| 16 | 0.102 | 0.091 | 2222 |
| 64 | 0.108 | 0.100 | 2012 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 508 | 1.19 | 0.23 | 1.37 | 0.04 | 2.83 |
| int8_graph 1x128 | 510 | 1.19 | 0.22 | 1.37 | 0.04 | 2.82 |
| int8_compiled 16x512 | 579 | 1.87 | 0.94 | 1.38 | 0.04 | 4.23 |
| int8_graph 16x512 | 581 | 1.87 | 0.93 | 1.37 | 0.04 | 4.21 |
| int8_compiled 16x2048 | 579 | 1.87 | 2.81 | 1.38 | 0.04 | 6.10 |
| int8_graph 16x2048 | 581 | 1.86 | 2.77 | 1.37 | 0.04 | 6.05 |
| int8_compiled 16x8192 | 579 | 1.86 | 9.90 | 1.38 | 0.05 | 13.19 |
| int8_graph 16x8192 | 581 | 1.86 | 9.84 | 1.37 | 0.04 | 13.11 |
| compiled_graph 1x128 | 342 | 1.50 | 0.22 | 0.15 | 0.03 | 1.90 |

Top kernels (first row): `triton_poi_fused__to_copy_5` 0.76 ms; `std::enable_if<!(false), void>::type internal::gemvx::kernel` 0.67 ms; `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_re` 0.52 ms; `triton_poi_fused__to_copy_1` 0.45 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.14 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7` 0.12 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.228 | 1.7787 | 111 | 4% |
| int8_graph 1x128 | 0.221 | 1.7238 | 114 | 4% |
| int8_compiled 16x512 | 0.939 | 0.1146 | 1716 | 60% |
| int8_graph 16x512 | 0.930 | 0.1135 | 1732 | 60% |
| int8_compiled 16x2048 | 2.807 | 0.0856 | 2296 | 80% |
| int8_graph 16x2048 | 2.769 | 0.0845 | 2327 | 81% |
| int8_compiled 16x8192 | 9.899 | 0.0755 | 2603 | 91% |
| int8_graph 16x8192 | 9.836 | 0.0750 | 2620 | 91% |
| compiled_graph 1x128 | 0.216 | 1.6843 | 117 | 4% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 1.391 ± 0.706 | 0.1722 ± 0.0216 | 0.000948 ± 0.000169 |  | 0.6574 | 109.5 | 10.37 |
| M2 | 3.147 ± 0.034 | 0.0501 ± 0.0015 |  | 0.0757 ± 0.0005 | 0.9988 | -163.1 | 0.62 |
| M3 | 3.171 ± 0.044 | 0.0494 ± 0.0017 | -0.000011 ± 0.000013 | 0.0760 ± 0.0007 | 0.9988 | -161.9 | 0.57 |

M2 KV slope → effective KV bandwidth 2598 GB/s (91% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.05 | 2.48 | 0.57 | 5.36× |
| 2 | 128 | 3.35 | 2.48 | 0.86 | 3.87× |
| 4 | 128 | 3.42 | 2.48 | 0.93 | 3.66× |
| 8 | 128 | 3.63 | 2.48 | 1.14 | 3.17× |
| 16 | 128 | 3.93 | 2.48 | 1.45 | 2.71× |
| 32 | 128 | 5.10 | 2.48 | 2.62 | 1.95× |
| 64 | 128 | 7.01 | 2.48 | 4.53 | 1.55× |
| 1 | 256 | 3.07 | 2.48 | 0.58 | 5.25× |
| 2 | 256 | 3.37 | 2.48 | 0.89 | 3.79× |
| 4 | 256 | 3.49 | 2.48 | 1.01 | 3.47× |
| 8 | 256 | 3.73 | 2.48 | 1.25 | 2.99× |
| 16 | 256 | 4.11 | 2.48 | 1.63 | 2.53× |
| 32 | 256 | 5.48 | 2.48 | 3.00 | 1.83× |
| 64 | 256 | 7.62 | 2.48 | 5.13 | 1.48× |
| 1 | 512 | 3.09 | 2.48 | 0.61 | 5.07× |
| 2 | 512 | 3.44 | 2.48 | 0.96 | 3.59× |
| 4 | 512 | 3.59 | 2.48 | 1.11 | 3.24× |
| 8 | 512 | 3.90 | 2.48 | 1.42 | 2.75× |
| 16 | 512 | 4.48 | 2.48 | 2.00 | 2.24× |
| 32 | 512 | 6.11 | 2.48 | 3.62 | 1.69× |
| 64 | 512 | 8.91 | 2.48 | 6.43 | 1.39× |
| 1 | 1024 | 3.16 | 2.48 | 0.68 | 4.65× |
| 2 | 1024 | 3.54 | 2.48 | 1.05 | 3.36× |
| 4 | 1024 | 3.74 | 2.48 | 1.26 | 2.97× |
| 8 | 1024 | 4.22 | 2.48 | 1.73 | 2.43× |
| 16 | 1024 | 5.07 | 2.48 | 2.59 | 1.96× |
| 32 | 1024 | 7.36 | 2.48 | 4.88 | 1.51× |
| 64 | 1024 | 11.29 | 2.48 | 8.81 | 1.28× |
| 1 | 2048 | 3.27 | 2.48 | 0.79 | 4.13× |
| 2 | 2048 | 3.71 | 2.48 | 1.23 | 3.02× |
| 4 | 2048 | 4.09 | 2.48 | 1.61 | 2.54× |
| 8 | 2048 | 4.82 | 2.48 | 2.33 | 2.06× |
| 16 | 2048 | 6.34 | 2.48 | 3.86 | 1.64× |
| 32 | 2048 | 9.77 | 2.48 | 7.29 | 1.34× |
| 64 | 2048 | 16.02 | 2.48 | 13.53 | 1.18× |
| 1 | 4096 | 3.58 | 2.48 | 1.10 | 3.26× |
| 2 | 4096 | 4.34 | 2.48 | 1.86 | 2.34× |
| 4 | 4096 | 4.65 | 2.48 | 2.17 | 2.14× |
| 8 | 4096 | 5.86 | 2.48 | 3.38 | 1.73× |
| 16 | 4096 | 8.74 | 2.48 | 6.26 | 1.40× |
| 32 | 4096 | 14.52 | 2.48 | 12.04 | 1.21× |
| 64 | 4096 | 26.26 | 2.48 | 23.77 | 1.10× |
| 1 | 8192 | 3.92 | 2.48 | 1.43 | 2.73× |
| 2 | 8192 | 4.60 | 2.48 | 2.11 | 2.17× |
| 4 | 8192 | 5.71 | 2.48 | 3.23 | 1.77× |
| 8 | 8192 | 7.89 | 2.48 | 5.41 | 1.46× |
| 16 | 8192 | 13.58 | 2.48 | 11.09 | 1.22× |
| 32 | 8192 | 25.04 | 2.48 | 22.56 | 1.11× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

