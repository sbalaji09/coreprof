# Week 5 v1: int8 weight-only + torch.compile (default) + CUDA-graph static-cache step over B x C on NVIDIA L40S

Predicted weight fetch 1.43 ms (spec), 1.95 ms (measured copy peak 633 GB/s), 1.81 ms (measured read peak 681 GB/s).

## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)

Looped-block weights: 2.467 GB bf16 → 1.235 GB int8 + scales (168 linears). Embedding, LM head, norms, exit gate stay bf16.

Accuracy, int8 vs bf16 step (eager, B=16, C=64): hidden state 6.3 %, exit gate 0.156, top-1 next token agrees 94 %. (int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state. GSM8K accuracy is the Week 6 test.)

| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.67 | 2.15 | 1.52 | 1.71× |
| 16 | 2048 | 13.37 | n/a | n/a | n/a× |
| 16 | 8192 | 39.02 | n/a | n/a | n/a× |

Weight time at B=1, C=128: bf16 0.006 ms (429670 GB/s) → int8 ≈ -1.517 ms (n/a GB/s), i.e. -264.23× the bf16 time for 0.50× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).

- int8, max-autotune-no-cudagraphs at B=1, C=128: 2.17 ms (compile 39.3 s)
- int8, max-autotune-no-cudagraphs at B=16, C=2048: 13.58 ms (compile 84.4 s)

## Calibration inputs

| quantity | value |
|---|---|
| GPU | NVIDIA L40S |
| spec bandwidth | 864 GB/s |
| copy bandwidth (read + write) | 633 GB/s |
| read bandwidth (torch.sum, 2 GiB) | 681 GB/s |
| weight time (int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128) | -1.517 ms |
| effective weight bandwidth | -814 GB/s (-120% of read peak) |
| δ (M2) | -0.063 ± 0.141 ms |
| γ (M2) | 2.0523 ± 0.0921 ms per sequence |
| α′ (M2) | 0.3252 ± 0.0103 µs per cached token |
| α′-implied KV bandwidth | 605 GB/s (89% of read peak) |
| attention kernels, µs per cached token (graph, BxC) | 1x128: 1.377, 16x512: 0.314, 16x2048: 0.303, 16x8192: 0.271 |
| small kernels (copy + other), graph, B=1, C=128 | 1.903 ms |
| gaps (graph step − kernel busy), B=1, C=128 | 0.065 ms |
| X1 crossover (weight time ÷ α′) | -4,664 cached tokens |
| LM head, graph, B=1 (once per token, not per loop) | 0.303 ms, 201.3 MB |

## Once per token (not per loop): LM head

lm_head: 100,663,296 params, 201.3 MB; vocab 49,152; tie_word_embeddings=False (shares the embedding tensor: False); embedding: 100,663,296 params.

| B | eager ms | graph ms | graph GB/s |
|---:|---:|---:|---:|
| 1 | 0.348 | 0.303 | 664 |
| 16 | 0.323 | 0.311 | 647 |
| 64 | 0.361 | 0.350 | 576 |

## GPU kernel time per step, by kind (kernel durations only; no host gaps)

| step | kernels | gemm ms | attention ms | copy ms | other ms | busy ms |
|---|---:|---:|---:|---:|---:|---:|
| int8_compiled 1x128 | 220 | 0.01 | 0.19 | 0.56 | 1.37 | 2.12 |
| int8_graph 1x128 | 222 | 0.01 | 0.18 | 0.55 | 1.35 | 2.08 |
| int8_compiled 16x512 | 459 | 1.89 | 2.57 | 2.32 | 0.55 | 7.33 |
| int8_graph 16x512 | 461 | 1.81 | 2.58 | 2.27 | 0.55 | 7.21 |
| int8_compiled 16x2048 | 459 | 1.86 | 9.92 | 1.83 | 0.52 | 14.13 |
| int8_graph 16x2048 | 461 | 1.78 | 9.92 | 1.78 | 0.51 | 13.99 |
| int8_compiled 16x8192 | 459 | 1.86 | 35.53 | 1.83 | 0.52 | 39.75 |
| int8_graph 16x8192 | 461 | 1.80 | 35.53 | 1.78 | 0.51 | 39.62 |
| compiled_graph 1x128 | 222 | 0.01 | 0.18 | 0.92 | 2.51 | 3.61 |

Top kernels (first row): `triton_red_fused_mm_5` 0.75 ms; `triton_red_fused_copy_mm_1` 0.46 ms; `triton_red_fused_mm_6` 0.45 ms; `triton_red_fused_bmm_3` 0.17 ms; `void pytorch_flash::flash_fwd_splitkv_kernel<pytorch_flash::` 0.11 ms; `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7` 0.09 ms

## Attention kernels across context (should match α′)

| step | attention ms | µs per cached token | KV read GB/s | % of read peak |
|---|---:|---:|---:|---:|
| int8_compiled 1x128 | 0.188 | 1.4716 | 134 | 20% |
| int8_graph 1x128 | 0.176 | 1.3773 | 143 | 21% |
| int8_compiled 16x512 | 2.574 | 0.3142 | 626 | 92% |
| int8_graph 16x512 | 2.576 | 0.3144 | 625 | 92% |
| int8_compiled 16x2048 | 9.917 | 0.3026 | 650 | 95% |
| int8_graph 16x2048 | 9.918 | 0.3027 | 650 | 95% |
| int8_compiled 16x8192 | 35.527 | 0.2711 | 725 | 107% |
| int8_graph 16x8192 | 35.528 | 0.2711 | 725 | 107% |
| compiled_graph 1x128 | 0.176 | 1.3740 | 143 | 21% |

## Fits

| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | -1.164 ± 0.556 | 2.8076 ± 0.3394 | 0.000474 ± 0.000062 |  | 0.9197 | -10.1 | 1.22 |
| M2 | -0.063 ± 0.141 | 2.0523 ± 0.0921 |  | 0.3252 ± 0.0103 | 0.9945 | -47.6 | 0.35 |
| M3 | 0.263 ± 0.111 | 1.8568 ± 0.0705 | -0.000140 ± 0.000031 | 0.4094 ± 0.0197 | 0.9982 | -61.1 | 0.23 |

M2 KV slope → effective KV bandwidth 605 GB/s (89% of measured read peak).

## Upper bound if weights were free (weight term = int8 weight time, estimated from step times)

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 2.15 | -1.52 | 3.67 | 0.59× |
| 2 | 128 | 4.04 | -1.52 | 5.55 | 0.73× |
| 1 | 256 | 2.16 | -1.52 | 3.67 | 0.59× |
| 2 | 256 | 4.10 | -1.52 | 5.61 | 0.73× |
| 1 | 512 | 2.24 | -1.52 | 3.75 | 0.60× |
| 2 | 512 | 4.25 | -1.52 | 5.77 | 0.74× |
| 1 | 1024 | 2.42 | -1.52 | 3.94 | 0.62× |
| 2 | 1024 | 4.60 | -1.52 | 6.12 | 0.75× |
| 1 | 2048 | 2.70 | -1.52 | 4.22 | 0.64× |
| 2 | 2048 | 5.48 | -1.52 | 7.00 | 0.78× |
| 1 | 4096 | 3.24 | -1.52 | 4.75 | 0.68× |
| 2 | 4096 | 6.99 | -1.52 | 8.50 | 0.82× |
| 1 | 8192 | 4.31 | -1.52 | 5.82 | 0.74× |
| 2 | 8192 | 9.40 | -1.52 | 10.92 | 0.86× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

