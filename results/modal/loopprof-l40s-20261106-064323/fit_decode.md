# LoopProf fit: decode_step time

39 configs; B in [1, 2, 4, 8, 16, 32, 64]; C in [128, 256, 512, 1024, 2048, 4096, 8192].

## M1: δ + γB + αC

R² = 0.4257, adjusted R² = 0.3937, AIC = 269.0 (Δ = 152.6), max |residual| = 80.45 ms (at B=16, C=2048), mean |residual| = 27.1 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 35.605 | 7.703 | [20.044, 51.165] | [24.775, 43.926] | [35.533, 35.682] |
| γ (ms per sequence) | 1.223 | 0.291 | [0.634, 1.811] | [0.622, 2.190] | [1.221, 1.224] |
| α (ms per context token) | 0.00939 | 0.00228 | [0.00479, 0.01398] | [0.00460, 0.01740] | [0.00937, 0.00940] |

## M2: δ + γB + α'·BC

R² = 0.9869, adjusted R² = 0.9861, AIC = 121.7 (Δ = 5.3), max |residual| = 9.34 ms (at B=1, C=8192), mean |residual| = 7.3 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 32.906 | 1.025 | [30.836, 34.977] | [30.619, 35.182] | [32.846, 32.969] |
| γ (ms per sequence) | -0.03942 | 0.04619 | [-0.133, 0.05389] | [-0.127, 0.03449] | [-0.04009, -0.03883] |
| α' (ms per sequence·context token) | 0.00369 | 7.724e-05 | [0.00353, 0.00384] | [0.00351, 0.00377] | [0.00368, 0.00369] |

## M3: δ + γB + αC + α'·BC  ← preferred (lowest AIC, or simplest within ΔAIC < 2)

R² = 0.9891, adjusted R² = 0.9882, AIC = 116.4 (Δ = 0.0), max |residual| = 7.48 ms (at B=2, C=128), mean |residual| = 6.7 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 34.283 | 1.077 | [32.108, 36.459] | [31.810, 36.698] | [34.213, 34.360] |
| γ (ms per sequence) | -0.117 | 0.05149 | [-0.221, -0.01273] | [-0.282, -0.05243] | [-0.118, -0.116] |
| α (ms per context token) | -0.00108 | 4.022e-04 | [-0.00189, -2.665e-04] | [-0.00191, -3.601e-04] | [-0.00109, -0.00107] |
| α' (ms per sequence·context token) | 0.00384 | 9.018e-05 | [0.00365, 0.00402] | [0.00366, 0.00395] | [0.00383, 0.00384] |

## Bandwidth utilization (bytes model ÷ measured copy peak)

min 34.8 %, median 41.5 %, max 47.2 %

## Time relative to the smallest batch at the same context length

| C | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.00 | 1.05 | 0.95 | 0.97 | 0.98 | 1.10 | 1.47 |
| 256 | 1.00 | 1.05 | 0.95 | 0.97 | 1.09 | 1.43 | 2.27 |
| 512 | 1.00 | 1.01 | 0.95 | 1.07 | 1.43 | 2.28 | 3.91 |
| 1024 | 1.00 | 1.01 | 1.07 | 1.43 | 2.27 | 3.92 |  |
| 2048 | 1.00 | 0.92 | 1.23 | 1.98 | 3.35 |  |  |
| 4096 | 1.00 | 1.24 | 1.98 | 3.39 |  |  |  |
| 8192 | 1.00 | 1.70 | 2.90 |  |  |  |  |

Plots: fit_decode_heat_time.png, fit_decode_heat_util.png, fit_decode_resid_M1.png, fit_decode_resid_M2.png, fit_decode_resid_M3.png, fit_decode_lines.png
