# Week 5: δ split on NVIDIA L40S

Looped-block weights: 2.467 GB. Predicted weight fetch: **2.85 ms** at spec bandwidth (864 GB/s), **3.74 ms** at the measured copy peak (659 GB/s).

Static-cache step vs HF step (same cache contents): relative error 0.00e+00, gate error 0.00e+00 → OK.

Launch cost: eager 4.78 µs of CPU per kernel launch; inside a CUDA graph 0.83 µs per op.

## Decomposition at B=1, C=128 (KV traffic negligible)

| implementation | step time (ms) |
|---|---:|
| eager + HF cache (LoopProf) | 11.75 |
| eager + static cache | 15.09 |
| CUDA graph + static cache (best-case GPU) | 5.95 |

| component | ms |
|---|---:|
| HF cache handling (eager_hf − eager_static) | -3.35 |
| launch + Python (eager_static − graph) | 9.15 |
| **software overhead total** | **5.80** (49 % of the eager step) |
| GPU time (graph) | 5.95 |
| … of which ideal weight streaming (measured peak) | 3.74 |
| … GPU time beyond ideal streaming | 2.20 |

**Layer sweep (graph_static)**: t = 0.03 ms + weight term; full-block weight term 6.25 ms → effective weight bandwidth 395 GB/s (R² 1.0000, n=5).

**Layer sweep (eager_static)**: t = 0.13 ms + weight term; full-block weight term 15.33 ms → effective weight bandwidth 161 GB/s (R² 1.0000, n=5).

## KV cost per cached token (slope of step time vs B·C)

| implementation | µs per cached token | effective KV bandwidth (GB/s) | R² |
|---|---:|---:|---:|
| eager_hf | 0.799 | 246 | 0.9974 |
| eager_static | 0.175 | 1121 | 0.2535 |
| graph_static | 0.346 | 568 | 0.8881 |

## Upper bound: removing only the weight term from the best-case GPU step

| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.95 | 6.25 | -0.30 | 5945296.05× |
| 1 | 2048 | 6.51 | 6.25 | 0.26 | 24.80× |
| 8 | 512 | 7.51 | 6.25 | 1.26 | 5.95× |
| 16 | 2048 | 15.53 | 6.25 | 9.28 | 1.67× |
| 32 | 1024 | 15.89 | 6.25 | 9.65 | 1.65× |
| 4 | 8192 | 14.98 | 6.25 | 8.73 | 1.72× |
| 1 | 128 | 0.82 | 6.25 | -5.43 | 818288.00× |
| 1 | 128 | 1.60 | 6.25 | -4.65 | 1597360.01× |
| 1 | 128 | 3.15 | 6.25 | -3.10 | 3149168.01× |
| 1 | 128 | 4.72 | 6.25 | -1.53 | 4717855.93× |
| 1 | 128 | 6.28 | 6.25 | 0.04 | 165.70× |

Upper bound on the step-time gain from removing ONLY the weight-streaming term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this.

