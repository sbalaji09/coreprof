# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 1.43 ms (spec), 1.94 ms (measured copy peak 635 GB/s), 1.80 ms (measured read peak 688 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.3 %, exit gate 0.156, top-1 next token agrees 94 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.18 | 3.72 | 0.46 | 1.12× |
| 16 | 2048 | 13.94 | 15.08 | -1.14 | 0.92× |
| 16 | 8192 | 39.55 | 40.65 | -1.09 | 0.97× |

Weight time at B=1, C=128: bf16 3.771 ms (654 GB/s) → int8 ≈ 3.311 ms (373 GB/s), i.e. 0.88× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 4.43 ms (compile 94.2 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 13.73 ms (compile 77.6 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 635 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 688 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 3.311 ms |
| effective weight bandwidth | 373 GB/s (54% of read peak) |
| δ (M2) | 4.218 ± 0.085 ms |
| γ (M2) | 0.0809 ± 0.0037 ms per sequence |
| α′ (M2) | 0.2711 ± 0.0021 µs per cached token |
| α′-implied KV bandwidth | 725 GB/s (105% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.502, 16x512: 0.312, 16x2048: 0.302, 16x8192: 0.271 |
| small kernels (copy + other), graph, B=1, C=128 | 1.997 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.054 ms |
| X1 crossover (weight time ÷ α′) | 12,210 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.302 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.352 | 0.302 | 666 |
| 16 | 0.325 | 0.312 | 646 |
| 64 | 0.360 | 0.349 | 577 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 508 | 1.47 | 0.21 | 2.00 | 0.03 | 3.71 |
| int8_graph 1x128 | 510 | 1.48 | 0.19 | 1.97 | 0.03 | 3.67 |
| int8_compiled 16x512 | 483 | 3.11 | 2.57 | 2.37 | 0.03 | 8.07 |
| int8_graph 16x512 | 485 | 3.01 | 2.56 | 2.34 | 0.03 | 7.94 |
| int8_compiled 16x2048 | 483 | 3.12 | 9.91 | 2.00 | 0.03 | 15.06 |
| int8_graph 16x2048 | 485 | 3.02 | 9.91 | 1.97 | 0.03 | 14.93 |
| int8_compiled 16x8192 | 483 | 3.09 | 35.54 | 2.00 | 0.03 | 40.66 |
| int8_graph 16x8192 | 485 | 3.02 | 35.53 | 1.97 | 0.03 | 40.55 |
| compiled_graph 1x128 | 342 | 3.77 | 0.17 | 0.13 | 0.03 | 4.10 |

Top kernels (first row): `std::enable_if<!(false), void>::type internal::gemvx::kernel` 1.46 ms; `triton_poi_fused__to_copy_5` 1.15 ms; `triton_poi_fused__to_copy_1` 0.71 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7` 0.11 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.05 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.205 | 1.6048 | 123 | 18% |
| int8_graph 1x128 | 0.192 | 1.5022 | 131 | 19% |
| int8_compiled 16x512 | 2.565 | 0.3131 | 628 | 91% |
| int8_graph 16x512 | 2.556 | 0.3121 | 630 | 92% |
| int8_compiled 16x2048 | 9.910 | 0.3024 | 650 | 95% |
| int8_graph 16x2048 | 9.906 | 0.3023 | 650 | 95% |
| int8_compiled 16x8192 | 35.539 | 0.2711 | 725 | 105% |
| int8_graph 16x8192 | 35.528 | 0.2711 | 725 | 105% |
| compiled_graph 1x128 | 0.173 | 1.3548 | 145 | 21% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 1.396 ± 1.571 | 0.3558 ± 0.0504 | 0.002195 ± 0.000394 |  | 0.6105 | 175.4 | 20.27 |
| M2 | 4.218 ± 0.085 | 0.0809 ± 0.0037 |  | 0.2711 ± 0.0021 | 0.9983 | -73.9 | 0.77 |
| M3 | 4.022 ± 0.098 | 0.0875 ± 0.0039 | 0.000100 ± 0.000031 | 0.2661 ± 0.0024 | 0.9986 | -82.3 | 0.81 |

M2 KV slope → effective KV bandwidth 725 GB/s (105% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.72 | 3.31 | 0.41 | 9.00× |
| 2 | 128 | 4.29 | 3.31 | 0.98 | 4.36× |
| 4 | 128 | 4.42 | 3.31 | 1.11 | 4.00× |
| 8 | 128 | 4.87 | 3.31 | 1.56 | 3.13× |
| 16 | 128 | 5.88 | 3.31 | 2.57 | 2.29× |
| 32 | 128 | 7.64 | 3.31 | 4.33 | 1.76× |
| 64 | 128 | 11.33 | 3.31 | 8.02 | 1.41× |
| 1 | 256 | 3.76 | 3.31 | 0.45 | 8.36× |
| 2 | 256 | 4.30 | 3.31 | 0.99 | 4.35× |
| 4 | 256 | 4.56 | 3.31 | 1.25 | 3.65× |
| 8 | 256 | 5.14 | 3.31 | 1.83 | 2.81× |
| 16 | 256 | 6.57 | 3.31 | 3.26 | 2.01× |
| 32 | 256 | 9.12 | 3.31 | 5.81 | 1.57× |
| 64 | 256 | 14.00 | 3.31 | 10.69 | 1.31× |
| 1 | 512 | 3.81 | 3.31 | 0.50 | 7.63× |
| 2 | 512 | 4.43 | 3.31 | 1.12 | 3.96× |
| 4 | 512 | 4.84 | 3.31 | 1.53 | 3.16× |
| 8 | 512 | 5.84 | 3.31 | 2.53 | 2.31× |
| 16 | 512 | 8.05 | 3.31 | 4.74 | 1.70× |
| 32 | 512 | 11.75 | 3.31 | 8.44 | 1.39× |
| 64 | 512 | 18.38 | 3.31 | 15.07 | 1.22× |
| 1 | 1024 | 3.98 | 3.31 | 0.67 | 5.93× |
| 2 | 1024 | 4.73 | 3.31 | 1.42 | 3.33× |
| 4 | 1024 | 5.50 | 3.31 | 2.19 | 2.51× |
| 8 | 1024 | 7.36 | 3.31 | 4.05 | 1.82× |
| 16 | 1024 | 10.70 | 3.31 | 7.39 | 1.45× |
| 32 | 1024 | 16.17 | 3.31 | 12.86 | 1.26× |
| 64 | 1024 | 27.06 | 3.31 | 23.75 | 1.14× |
| 1 | 2048 | 4.29 | 3.31 | 0.98 | 4.37× |
| 2 | 2048 | 5.44 | 3.31 | 2.13 | 2.55× |
| 4 | 2048 | 7.03 | 3.31 | 3.72 | 1.89× |
| 8 | 2048 | 10.08 | 3.31 | 6.76 | 1.49× |
| 16 | 2048 | 15.08 | 3.31 | 11.77 | 1.28× |
| 32 | 2048 | 24.69 | 3.31 | 21.38 | 1.15× |
| 64 | 2048 | 44.22 | 3.31 | 40.91 | 1.08× |
| 1 | 4096 | 4.98 | 3.31 | 1.67 | 2.98× |
| 2 | 4096 | 6.96 | 3.31 | 3.64 | 1.91× |
| 4 | 4096 | 9.70 | 3.31 | 6.39 | 1.52× |
| 8 | 4096 | 14.36 | 3.31 | 11.05 | 1.30× |
| 16 | 4096 | 23.64 | 3.31 | 20.33 | 1.16× |
| 32 | 4096 | 42.05 | 3.31 | 38.74 | 1.09× |
| 1 | 8192 | 6.49 | 3.31 | 3.18 | 2.04× |
| 2 | 8192 | 9.58 | 3.31 | 6.27 | 1.53× |
| 4 | 8192 | 14.04 | 3.31 | 10.73 | 1.31× |
| 8 | 8192 | 22.96 | 3.31 | 19.65 | 1.17× |
| 16 | 8192 | 40.65 | 3.31 | 37.34 | 1.09× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

