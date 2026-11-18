# int8 probe on NVIDIA L40S

bf16 compiled: 1x128 4.09 ms, 16x2048 13.91 ms, 64x128 10.61 ms; bf16 weight kernels at 1x128: 3.768 ms

| variant | 1x128 ms | 16x2048 ms | 64x128 ms | int8 weight est. ms (× bf16) | to_copy ms |
|---|---:|---:|---:|---:|---:|
| cast, default | 3.73 | 14.97 | 11.23 | 3.406 (0.90×) | 1.934 |
| cast, mixed_mm_choice=triton | 3.72 | 14.99 | 11.26 | 3.395 (0.90×) | 1.934 |
| cast, coordinate_descent_tuning | 2.17 | 14.12 | 8.87 | 1.846 (0.49×) | 0.091 |
| cast, both | 2.17 | 14.12 | 8.87 | 1.848 (0.49×) | 0.091 |
| triton int8 kernel | 2.58 | 11.28 | 5.31 | 2.255 (0.60×) | 0.101 |

Triton kernel check: {'max_rel_err': 0.0017814630409702659, 'ok': True}. Ideal: int8 weight time ≈ 0.5× bf16.
