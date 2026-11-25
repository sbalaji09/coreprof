# LoopCore chip estimate (inputs and labels at the top of chip_cost.py)
area 310-753 mm² = (SRAM 132-321 + compute 123-247 + SerDes 27-60) x (1 + other 0.10-0.20); MB/mm² 0.66-1.61
power 310-576 W = (SRAM 144-264 + MACs 46 + SerDes 48-50) x 1.3-1.6 (full load)
cost: LoopCore 225-1049 USD; LoopCore + 1 HBM stack 993-2189 USD; H100 2193-3603 USD (die + 80 GB HBM + CoWoS + assembly)
calibration, GC200: predicted 508-914 mm² vs actual 823 mm²
ratio area      0.38-0.93
ratio area_hbm  0.40-0.96
ratio power     0.44-0.82
ratio power_hbm 0.47-0.87
ratio cost      0.08-0.38
ratio cost_hbm  0.39-0.67

## speedup per chip vs H100
| design | chips 1x / 16x | (b) B=1 | (c) B=1 | B=16 |
|---|---|---|---|---|
| per-loop, HBM | 3 / 3 | 1.27-2.37 (above 1) | 1.93-3.30 (above 1) | 1.03-1.43 (above 1) |
| per-loop, on-chip, near | 5 / 29 | 0.88-1.98 (straddles) | 2.18-4.78 (above 1) | 0.39-0.74 (below 1) |
| per-loop, on-chip, stream | 5 / 29 | 0.20-1.04 (straddles) | 0.18-1.22 (straddles) | 0.01-0.07 (below 1) |
| per-loop, on-chip, co-located | 5 / 29 | 1.02-2.20 (above 1) | 2.52-5.24 (above 1) | does not fit |
| shared, on-chip, near | 4 / 10 | 1.23-2.70 (above 1) | 3.05-6.40 (above 1) | 1.15-2.15 (above 1) |
| shared, on-chip, stream | 4 / 10 | 0.70-2.23 (straddles) | 0.78-3.77 (straddles) | 0.03-0.20 (below 1) |
| shared, on-chip, co-located | 4 / 12 | 1.27-2.75 (above 1) | 3.15-6.55 (above 1) | 1.08-1.90 (above 1) |

## speedup per area vs H100 (LoopCore/H100 ratio 0.38-0.93; HBM design 0.40-0.96)
| design | chips 1x / 16x | (b) B=1 | (c) B=1 | B=16 |
|---|---|---|---|---|
| per-loop, HBM | 3 / 3 | 1.32-5.93 (above 1) | 2.01-8.26 (above 1) | 1.07-3.59 (above 1) |
| per-loop, on-chip, near | 5 / 29 | 0.95-5.20 (straddles) | 2.36-12.55 (above 1) | 0.42-1.94 (straddles) |
| per-loop, on-chip, stream | 5 / 29 | 0.22-2.73 (straddles) | 0.19-3.20 (straddles) | 0.01-0.17 (below 1) |
| per-loop, on-chip, co-located | 5 / 29 | 1.10-5.78 (above 1) | 2.72-13.76 (above 1) | does not fit |
| shared, on-chip, near | 4 / 10 | 1.32-7.09 (above 1) | 3.30-16.80 (above 1) | 1.24-5.64 (above 1) |
| shared, on-chip, stream | 4 / 10 | 0.76-5.84 (straddles) | 0.84-9.91 (straddles) | 0.03-0.53 (below 1) |
| shared, on-chip, co-located | 4 / 12 | 1.38-7.22 (above 1) | 3.40-17.20 (above 1) | 1.17-4.99 (above 1) |

## speedup per power vs H100 (LoopCore/H100 ratio 0.44-0.82; HBM design 0.47-0.87)
| design | chips 1x / 16x | (b) B=1 | (c) B=1 | B=16 |
|---|---|---|---|---|
| per-loop, HBM | 3 / 3 | 1.46-5.00 (above 1) | 2.23-6.98 (above 1) | 1.19-3.03 (above 1) |
| per-loop, on-chip, near | 5 / 29 | 1.07-4.47 (above 1) | 2.65-10.79 (above 1) | 0.48-1.67 (straddles) |
| per-loop, on-chip, stream | 5 / 29 | 0.24-2.35 (straddles) | 0.22-2.75 (straddles) | 0.01-0.15 (below 1) |
| per-loop, on-chip, co-located | 5 / 29 | 1.24-4.97 (above 1) | 3.06-11.83 (above 1) | does not fit |
| shared, on-chip, near | 4 / 10 | 1.49-6.09 (above 1) | 3.71-14.44 (above 1) | 1.40-4.85 (above 1) |
| shared, on-chip, stream | 4 / 10 | 0.85-5.02 (straddles) | 0.94-8.52 (straddles) | 0.04-0.45 (below 1) |
| shared, on-chip, co-located | 4 / 12 | 1.55-6.21 (above 1) | 3.83-14.78 (above 1) | 1.32-4.29 (above 1) |

## speedup per cost vs H100 (LoopCore/H100 ratio 0.08-0.38; HBM design 0.39-0.67)
| design | chips 1x / 16x | (b) B=1 | (c) B=1 | B=16 |
|---|---|---|---|---|
| per-loop, HBM | 3 / 3 | 1.88-6.04 (above 1) | 2.87-8.42 (above 1) | 1.53-3.66 (above 1) |
| per-loop, on-chip, near | 5 / 29 | 2.29-26.13 (above 1) | 5.67-63.08 (above 1) | 1.02-9.74 (above 1) |
| per-loop, on-chip, stream | 5 / 29 | 0.52-13.72 (straddles) | 0.47-16.10 (straddles) | 0.03-0.86 (below 1) |
| per-loop, on-chip, co-located | 5 / 29 | 2.65-29.03 (above 1) | 6.56-69.15 (above 1) | does not fit |
| shared, on-chip, near | 4 / 10 | 3.19-35.63 (above 1) | 7.93-84.45 (above 1) | 2.99-28.37 (above 1) |
| shared, on-chip, stream | 4 / 10 | 1.82-29.36 (above 1) | 2.02-49.81 (above 1) | 0.08-2.64 (straddles) |
| shared, on-chip, co-located | 4 / 12 | 3.32-36.29 (above 1) | 8.19-86.43 (above 1) | 2.82-25.07 (above 1) |
