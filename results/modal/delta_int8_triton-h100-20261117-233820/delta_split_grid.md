# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA H100 80GB HBM3

Predicted weight fetch 0.37 ms (spec), 0.41 ms (measured copy peak 3017 GB/s), 0.43 ms (measured read peak 2877 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.2 %, exit gate 0.312, top-1 next token agrees 88 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.07 | 2.42 | -0.35 | 0.86× |
| 16 | 2048 | 5.19 | 4.92 | 0.27 | 1.06× |
| 16 | 8192 | 12.59 | 12.40 | 0.19 | 1.02× |

Weight time at B=1, C=128: bf16 1.502 ms (1642 GB/s) → int8 ≈ 1.853 ms (666 GB/s), i.e. 1.23× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 2.43 ms (compile 18.9 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 4.92 ms (compile 20.7 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| spec bandwidth | 3,350 GB/s |
| copy bandwidth (read + write) | 3,017 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 2,877 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | 1.853 ms |
| effective weight bandwidth | 666 GB/s (23% of read peak) |
| δ (M2) | 2.202 ± 0.050 ms |
| γ (M2) | 0.0252 ± 0.0021 ms per sequence |
| α′ (M2) | 0.0761 ± 0.0007 µs per cached token |
| α′-implied KV bandwidth | 2,585 GB/s (90% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.687, 16x512: 0.102, 16x2048: 0.082, 16x8192: 0.075 |
| small kernels (copy + other), graph, B=1, C=128 | 2.035 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.168 ms |
| X1 crossover (weight time ÷ α′) | 24,365 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.092 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.117 | 0.092 | 2189 |
| 16 | 0.099 | 0.092 | 2199 |
| 64 | 0.109 | 0.099 | 2035 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 340 | 0.00 | 0.22 | 0.15 | 1.90 | 2.27 |
| int8_graph 1x128 | 342 | 0.00 | 0.22 | 0.14 | 1.89 | 2.26 |
| int8_compiled 16x512 | 315 | 0.00 | 0.84 | 0.16 | 1.93 | 2.93 |
| int8_graph 16x512 | 317 | 0.00 | 0.84 | 0.15 | 1.91 | 2.90 |
| int8_compiled 16x2048 | 315 | 0.00 | 2.73 | 0.16 | 1.94 | 4.83 |
| int8_graph 16x2048 | 317 | 0.00 | 2.68 | 0.15 | 1.92 | 4.76 |
| int8_compiled 16x8192 | 315 | 0.00 | 9.88 | 0.16 | 1.91 | 11.94 |
| int8_graph 16x8192 | 317 | 0.00 | 9.87 | 0.15 | 1.91 | 11.93 |
| compiled_graph 1x128 | 342 | 1.50 | 0.22 | 0.15 | 0.03 | 1.89 |

Top kernels (first row): `_int8_wo_kernel` 1.87 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.13 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_3` 0.06 ms; `void pytorch_flash::flash_fwd_splitkv_combine_kernel<pytorch` 0.06 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6` 0.05 ms; `triton_poi_fused__scaled_dot_product_flash_attention_add_cat` 0.03 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.223 | 1.7405 | 113 | 4% |
| int8_graph 1x128 | 0.216 | 1.6873 | 117 | 4% |
| int8_compiled 16x512 | 0.842 | 0.1028 | 1913 | 66% |
| int8_graph 16x512 | 0.837 | 0.1022 | 1923 | 67% |
| int8_compiled 16x2048 | 2.730 | 0.0833 | 2360 | 82% |
| int8_graph 16x2048 | 2.683 | 0.0819 | 2401 | 83% |
| int8_compiled 16x8192 | 9.878 | 0.0754 | 2609 | 91% |
| int8_graph 16x8192 | 9.871 | 0.0753 | 2611 | 91% |
| compiled_graph 1x128 | 0.216 | 1.6871 | 117 | 4% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.451 ± 0.714 | 0.1478 ± 0.0219 | 0.000947 ± 0.000171 |  | 0.6059 | 110.6 | 10.61 |
| M2 | 2.202 ± 0.050 | 0.0252 ± 0.0021 |  | 0.0761 ± 0.0007 | 0.9971 | -125.5 | 0.68 |
| M3 | 2.248 ± 0.065 | 0.0239 ± 0.0024 | -0.000021 ± 0.000019 | 0.0768 ± 0.0010 | 0.9972 | -124.8 | 0.59 |

M2 KV slope → effective KV bandwidth 2585 GB/s (90% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.42 | 1.85 | 0.57 | 4.25× |
| 2 | 128 | 2.43 | 1.85 | 0.57 | 4.23× |
| 4 | 128 | 2.45 | 1.85 | 0.59 | 4.12× |
| 8 | 128 | 2.48 | 1.85 | 0.63 | 3.95× |
| 16 | 128 | 2.57 | 1.85 | 0.71 | 3.60× |
| 32 | 128 | 2.97 | 1.85 | 1.11 | 2.67× |
| 64 | 128 | 4.65 | 1.85 | 2.80 | 1.66× |
| 1 | 256 | 2.43 | 1.85 | 0.58 | 4.19× |
| 2 | 256 | 2.46 | 1.85 | 0.61 | 4.05× |
| 4 | 256 | 2.50 | 1.85 | 0.65 | 3.87× |
| 8 | 256 | 2.56 | 1.85 | 0.71 | 3.61× |
| 16 | 256 | 2.73 | 1.85 | 0.88 | 3.11× |
| 32 | 256 | 3.29 | 1.85 | 1.44 | 2.29× |
| 64 | 256 | 5.27 | 1.85 | 3.42 | 1.54× |
| 1 | 512 | 2.46 | 1.85 | 0.61 | 4.04× |
| 2 | 512 | 2.50 | 1.85 | 0.64 | 3.88× |
| 4 | 512 | 2.58 | 1.85 | 0.72 | 3.57× |
| 8 | 512 | 2.70 | 1.85 | 0.85 | 3.18× |
| 16 | 512 | 3.01 | 1.85 | 1.16 | 2.60× |
| 32 | 512 | 3.92 | 1.85 | 2.06 | 1.90× |
| 64 | 512 | 6.55 | 1.85 | 4.69 | 1.39× |
| 1 | 1024 | 2.52 | 1.85 | 0.67 | 3.78× |
| 2 | 1024 | 2.59 | 1.85 | 0.73 | 3.52× |
| 4 | 1024 | 2.71 | 1.85 | 0.86 | 3.16× |
| 8 | 1024 | 2.98 | 1.85 | 1.13 | 2.64× |
| 16 | 1024 | 3.64 | 1.85 | 1.79 | 2.04× |
| 32 | 1024 | 5.18 | 1.85 | 3.33 | 1.56× |
| 64 | 1024 | 8.93 | 1.85 | 7.07 | 1.26× |
| 1 | 2048 | 2.61 | 1.85 | 0.76 | 3.44× |
| 2 | 2048 | 2.71 | 1.85 | 0.86 | 3.16× |
| 4 | 2048 | 2.98 | 1.85 | 1.13 | 2.64× |
| 8 | 2048 | 3.49 | 1.85 | 1.64 | 2.13× |
| 16 | 2048 | 4.92 | 1.85 | 3.06 | 1.61× |
| 32 | 2048 | 7.54 | 1.85 | 5.69 | 1.33× |
| 64 | 2048 | 13.84 | 1.85 | 11.98 | 1.15× |
| 1 | 4096 | 2.88 | 1.85 | 1.03 | 2.80× |
| 2 | 4096 | 3.31 | 1.85 | 1.45 | 2.28× |
| 4 | 4096 | 3.52 | 1.85 | 1.67 | 2.11× |
| 8 | 4096 | 4.61 | 1.85 | 2.75 | 1.67× |
| 16 | 4096 | 7.39 | 1.85 | 5.53 | 1.33× |
| 32 | 4096 | 12.57 | 1.85 | 10.72 | 1.17× |
| 64 | 4096 | 24.41 | 1.85 | 22.55 | 1.08× |
| 1 | 8192 | 3.17 | 1.85 | 1.31 | 2.41× |
| 2 | 8192 | 3.53 | 1.85 | 1.68 | 2.10× |
| 4 | 8192 | 4.60 | 1.85 | 2.75 | 1.67× |
| 8 | 8192 | 6.71 | 1.85 | 4.85 | 1.38× |
| 16 | 8192 | 12.40 | 1.85 | 10.54 | 1.18× |
| 32 | 8192 | 22.97 | 1.85 | 21.12 | 1.09× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

## Errors

- **static_vs_hf_check**: `AssertionError: decode kernel: one BLOCK_M covers M <= 64`
