# Week 1: where the decode time goes (RTX A6000, sweep `20261023-064159`)

Setup: Ouro-1.4B at revision `574fa66`, bf16, SDPA, HF eager with the HF
`UniversalTransformerCache`. Context C = 512, 10 warmup + 50 reps. Run on an
RTX A6000, not the target A10 (README, "Results").

| B | one loop step, median ms | full token, ms | loop-step tokens/s |
|---|---|---|---|
| 1 | 16.9 | 72.3 | 59 |
| 2 | 17.1 | 69.5 | 117 |
| 4 | 18.3 | 69.6 | 218 |
| 8 | 17.9 | 70.0 | 448 |
| 16 | 18.2 | 73.2 | 880 |
| 32 | 27.9 | 112.1 | 1148 |
| 64 | 48.4 | 193.1 | 1322 |

A full token costs 4.0–4.3 loop steps at every B, so everything below is
stated per loop step (24 layers, norm and exit gate).

## The weight-read floor

The recurrent core is all 24 decoder layers. Each layer has q/k/v/o
(4 × 2048²) and a SwiGLU MLP (3 × 2048 × 5632), which is 51,388,416
parameters. Counted from the loaded model:

- **24 layers = 2,466,643,968 bytes in bf16** (2.47 GB).
- Final norm and gate: 8 KB.
- Embedding and lm_head: 0.2 GB each. They run once per token, not per loop.

The 6 MB L2 cache can't hold the weights between loops, so every loop step
streams all 2.47 GB from DRAM.

The RTX A6000 has **768 GB/s** GDDR6 bandwidth (NVIDIA RTX A6000 datasheet,
https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf).

**Floor = 2.467 GB / 768 GB/s = 3.21 ms per loop step**, independent of B
while decode is memory-bound. The measured 16.9 ms is **5.3× the floor**.

## Flat from B = 1 to 16: not the weight read

The latency is flat as expected for decode, but it sits at ~17 ms, not
~3 ms. The one-off `torch.profiler` run (README, "Sanity check") shows why:

- **At B = 1, the GPU is busy 7.2 of 17.8 ms (40%).** The weight GEMMs
  take 4.2 ms, about 76% of spec bandwidth, which is close to the floor.
- **The other ~10 ms is the GPU waiting for the CPU.** A loop step issues
  about 1,400 small kernels from eager Python (≈12 µs each on this 4-core
  pod): 24 layers × (4 RMSNorms, rotary, 7 GEMMs, SDPA, cache `cat`,
  residuals…).
- **Adding rows adds almost no CPU time and only a little GPU time.** So
  the step stays at the CPU's pace until the GPU work catches up.

## The rise at B = 32 and 64: KV traffic

Per sequence, one loop step's KV at C = 512 is 24 layers × K+V ×
16 heads × 512 × 128 × 2 B = **101 MB**. That KV is touched about three
times per step:

- attention reads it;
- HF's `torch.cat` cache growth (NOTES.md §3) reads it again;
- `torch.cat` then writes a new copy.

So each sequence adds about 0.3 GB of traffic, or ≥0.39 ms per sequence at
spec bandwidth. The measured slope between B = 32 and 64 is
**0.64 ms per sequence** (20.5 ms / 32), about 470 GB/s effective. In the
profile at B = 64, `cat` takes 29.0 ms and attention 10.1 ms, against 4.7 ms
for the weight GEMMs.

The model predicts the knee. GPU work is about 3.2 ms + B × 0.6 ms, which
overtakes the ~17 ms of CPU launch time at **B ≈ 22**. That falls between the
flat B = 16 point and the B = 32 rise. At B = 16, GPU work is 16.9 ms, already
level with the launch time.

## Conclusion

- **Most of the flat ~17 ms cost is probably launch/framework overhead,
  not hardware.** About 3.2 ms is the unavoidable weight read and about 4 ms
  the GEMMs actually take. The remaining ~10–13 ms is Python/kernel-launch
  time with the GPU idle.
- **Above B ≈ 20 the cost is dominated by KV traffic.** Most of that is the
  HF cache's `cat` copy, an implementation artifact that a static or paged
  cache removes.
- Our Week 2 batchers already use a static store, but they gather instead.

**Week 5 has to split the flat part properly.** Run the same loop step:

- under CUDA graphs (removes launches);
- with a static cache (removes `cat`);
- with an NVTX/profiler breakdown of CPU time per op.

That separates three components: launch overhead, framework (Python/HF)
overhead and real GPU time. The 5.3× gap is the upper bound on what that
work can recover at small B.

The 40% figure and the per-op split come from one profiling run (B = 16
measured 22.6 ms there against 18.2 ms in the sweep), so treat them as
approximate.
