# LoopProf fit: ut_step time

39 configs; B in [1, 2, 4, 8, 16, 32, 64]; C in [128, 256, 512, 1024, 2048, 4096, 8192].

## M1: δ + γB + αC

R² = 0.4299, adjusted R² = 0.3983, AIC = 160.8 (Δ = 156.2), max |residual| = 20.10 ms (at B=16, C=2048), mean |residual| = 26.9 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 8.643 | 1.925 | [4.754, 12.532] | [5.901, 10.720] | [8.612, 8.689] |
| γ (ms per sequence) | 0.308 | 0.07284 | [0.161, 0.456] | [0.158, 0.549] | [0.307, 0.309] |
| α (ms per context token) | 0.00237 | 5.691e-04 | [0.00122, 0.00351] | [0.00117, 0.00439] | [0.00236, 0.00237] |

## M2: δ + γB + α'·BC

R² = 0.9882, adjusted R² = 0.9875, AIC = 9.7 (Δ = 5.0), max |residual| = 2.27 ms (at B=1, C=8192), mean |residual| = 7.2 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 7.990 | 0.244 | [7.498, 8.483] | [7.468, 8.536] | [7.962, 8.033] |
| γ (ms per sequence) | -0.00860 | 0.01099 | [-0.03080, 0.01360] | [-0.02919, 0.00855] | [-0.00921, -0.00823] |
| α' (ms per sequence·context token) | 9.248e-04 | 1.838e-05 | [8.876e-04, 9.619e-04] | [8.848e-04, 9.446e-04] | [9.233e-04, 9.258e-04] |

## M3: δ + γB + αC + α'·BC  ← preferred (lowest AIC, or simplest within ΔAIC < 2)

R² = 0.9901, adjusted R² = 0.9893, AIC = 4.7 (Δ = 0.0), max |residual| = 2.06 ms (at B=1, C=2048), mean |residual| = 6.6 %

| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |
|---|---:|---:|---|---|---|
| δ (ms) | 8.312 | 0.257 | [7.793, 8.832] | [7.720, 8.870] | [8.280, 8.358] |
| γ (ms per sequence) | -0.02667 | 0.01229 | [-0.05150, -0.00184] | [-0.06588, -0.01137] | [-0.02750, -0.02608] |
| α (ms per context token) | -2.522e-04 | 9.604e-05 | [-4.462e-04, -5.821e-05] | [-4.365e-04, -7.743e-05] | [-2.573e-04, -2.484e-04] |
| α' (ms per sequence·context token) | 9.593e-04 | 2.153e-05 | [9.159e-04, 0.00100] | [9.163e-04, 9.861e-04] | [9.580e-04, 9.604e-04] |

## Bandwidth utilization (bytes model ÷ measured copy peak)

min 34.8 %, median 41.5 %, max 47.2 %

## Time relative to the smallest batch at the same context length

| C | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.00 | 0.96 | 1.01 | 0.95 | 0.99 | 1.13 | 1.51 |
| 256 | 1.00 | 1.06 | 1.07 | 1.06 | 1.18 | 1.55 | 2.48 |
| 512 | 1.00 | 1.10 | 0.94 | 1.08 | 1.45 | 2.32 | 3.98 |
| 1024 | 1.00 | 0.98 | 1.10 | 1.47 | 2.35 | 4.07 |  |
| 2048 | 1.00 | 0.90 | 1.19 | 1.93 | 3.27 |  |  |
| 4096 | 1.00 | 1.26 | 2.02 | 3.45 |  |  |  |
| 8192 | 1.00 | 1.70 | 2.92 |  |  |  |  |

Plots: fit_ut_heat_time.png, fit_ut_heat_util.png, fit_ut_resid_M1.png, fit_ut_resid_M2.png, fit_ut_resid_M3.png, fit_ut_lines.png
