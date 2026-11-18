# int8 probe on NVIDIA H100 80GB HBM3

bf16 compiled: 1x128 2.02 ms, 16x2048 5.19 ms, 64x128 6.07 ms; bf16 weight kernels at 1x128: 1.496 ms

| variant | 1x128 ms | 16x2048 ms | 64x128 ms | int8 weight est. ms (× bf16) | to_copy ms |
|---|---:|---:|---:|---:|---:|
| cast, default | 3.11 | 6.37 | 7.00 | 2.588 (1.73×) | 1.334 |
| cast, mixed_mm_choice=triton | 3.12 | 6.38 | 7.00 | 2.593 (1.73×) | 1.350 |
| cast, coordinate_descent_tuning | 1.25 | 6.17 | 5.82 | 0.724 (0.48×) | 0.103 |
| cast, both | 1.22 | 6.16 | 5.82 | 0.702 (0.47×) | 0.104 |
| triton int8 kernel | 2.47 | 4.97 | 4.87 | 1.951 (1.30×) | 0.115 |

Triton kernel check: {'max_rel_err': 0.0017067150911316276, 'ok': True}. Ideal: int8 weight time ≈ 0.5× bf16.
