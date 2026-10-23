# LoopCore: Ouro-1.4B decode timing harness

Measures, on one GPU, how long **one recurrent step** of Ouro-1.4B takes during
decode compared with **one full decode token** (all 4 steps, exit gate and
lm_head), across batch sizes. NOTES.md explains what each one is, with line
references into the model code.

## Files

| file | what it is |
|---|---|
| `smoke_test.py` | loads the model and greedily generates 50 tokens; prints load time and peak memory |
| `harness.py` | the timing sweep. Writes `results/<timestamp>.csv` and `.json` (metadata and raw per-rep times) |
| `plot.py` | turns a CSV into `<timestamp>_latency.png` and `<timestamp>_tokens_per_s.png` |
| `selfcheck.py` | run before any sweep: GPU repeatability, plus the check that calling the block directly reproduces the model's forward |
| `NOTES.md` | how the Ouro loop, exit gate and per-loop KV cache work |

## Running on the NRP JupyterHub pod from scratch

```bash
# 1. Keep large files off the 5 GB home volume.
#    The image already sets these; set them yourself if `env | grep -E "HF_HOME|PIP_CACHE"` is empty.
export HF_HOME=/tmp/hf            # model weights (~2.8 GB) and remote code
export PIP_CACHE_DIR=/tmp/pip

# 2. Install. Packages go to /opt/conda (container storage, not home).
cd ~/loopcore
pip install -r requirements.txt

# 3. Check the GPU is healthy and the harness is correct (must print ALL OK),
#    then run the smoke test.
nvidia-smi
python selfcheck.py
python smoke_test.py

# 4. Sweep (defaults: B = 1 2 4 8 16 32 64, C = 512, 10 warmup + 50 reps, bf16).
python harness.py
python harness.py --batch-sizes 1 4 16 --context-lens 512 2048 --reps 100
python harness.py --targets ut_step --ut-step 3

# 5. Plot.
python plot.py results/<timestamp>.csv
```

**What a pod restart wipes:** `/tmp` (the HF cache, so weights download again,
about a minute) and `/opt/conda` (rerun step 2). Only `~/loopcore` survives,
including `results/`.

## What gets measured

- **`ut_step`**: one pass of the loop body (24 layers, final norm and exit
  gate) on one new token per sequence, with that step's KV cache prefilled to
  C tokens. It is called directly on the model's layers rather than through
  `generate()`.
- **`decode_step`**: `model(...)` for one token, the call `generate()` makes
  per step.
- **Method**: random prompts (fixed seed) of length C are prefilled for B
  sequences. Before every repetition the cache is sliced back to C tokens,
  untimed. Each rep is timed with CUDA events after draining the queue. The
  first `--warmup` reps are discarded. The whole sweep runs under
  `torch.inference_mode()`.
- **CSV columns**: `target, B, C, status, median_ms, p10_ms, p90_ms, iqr_ms,
  mean_ms, tokens_per_s (= B / median), peak_mem_gb, sm_clock_mhz, temp_c`.
  `status` is `oom` or `oom_prefill` when a config doesn't fit. For
  `ut_step`, `tokens_per_s` is the throughput of a single loop, not of real
  decoding.
- **Metadata** (`.json`): GPU, driver, torch, CUDA, transformers, model
  revision, attention implementation, dtype, timestamp, git commit and dirty
  flag, and all CLI args.

**Expected out-of-memory configs:** Ouro keeps a separate KV cache for each of
its 4 loops, which costs about 0.79 MB per token in bf16. B=64 at C=512 needs
25.8 GB for the cache alone, so on a 24 GB A10 it is recorded as
`oom_prefill`. It fits on a 48 GB A6000.

## Extending (LoopProf)

- A new thing to time: write `build_<name>(model, state, args) -> callable`
  in `harness.py` and add it to `TARGETS`.
- A new sweep axis: add a CLI arg and extend `sweep()`.
- A new metric: compute it in `run_config()` and add the column to
  `CSV_FIELDS`.

## Results

### `results/20261023-064159` — NVIDIA RTX A6000 (48 GB), not the A10

**This run is on an RTX A6000, not the target A10.** On 2026-10-22 the
original A10 pod started failing plain PyTorch matmuls (cuBLAS errors, illegal
memory accesses, silently wrong outputs). The pod was restarted, and no A10
was free, so it landed on an A6000. The A6000 has more memory bandwidth
(~768 vs ~600 GB/s), more SMs and 48 GB of memory, so its numbers do **not**
carry over to the A10. Rerun on an A10 before quoting A10 figures.

Setup: driver 615.71.10, torch 2.5.1+cu124, transformers 4.57.1, bf16, SDPA,
C=512, 10 warmup + 50 reps, commit `ecbadcd`. `selfcheck.py` printed
`ALL OK`: `ut_step` matches the model's forward bit for bit at all 4 steps.

| B | one loop step, median (p10–p90) ms | full decode token, median ms | full / loop | tokens/s | peak GB |
|---|---|---|---|---|---|
| 1 | 16.9 (16.4–17.3) | 72.3 | 4.3 | 13.8 | 3.1 |
| 2 | 17.1 (16.7–17.4) | 69.5 | 4.1 | 28.8 | 3.4 |
| 4 | 18.3 (17.8–19.0) | 69.6 | 3.8 | 57.5 | 4.2 |
| 8 | 17.9 (17.4–18.6) | 70.0 | 3.9 | 114 | 5.7 |
| 16 | 18.2 (18.1–18.3) | 73.2 | 4.0 | 219 | 8.8 |
| 32 | 27.9 (27.8–28.0) | 112.1 | 4.0 | 285 | 14.8 |
| 64 | 48.4 (48.2–48.6) | 193.1 | 4.0 | 332 | 27.0 |

- A full token costs about 4 loop steps, as expected: Ouro always runs all
  4 steps (NOTES.md §2), and embedding, lm_head and exit selection add
  little.
- B=64 fits on the A6000. On a 24 GB A10 it would not, because the 4
  per-loop KV caches need 25.8 GB at C=512.
- The SM clock fell from 1950 MHz (B=1) to 1695–1725 MHz (B=64) under
  sustained load. Check `sm_clock_mhz` in the CSV when comparing runs.

### Sanity check: flat at small B, but launch-bound, not memory-bound

Latency is flat from B=1 to 16, but not for the expected reason. A loop step
reads about 2.5 GB of weights, which takes about 3.2 ms at the A6000's
bandwidth, yet it measures about 17 ms. A one-off `torch.profiler` run
(ut_step, C=512) shows why:

| B | CUDA-event time | GPU kernel busy | weight GEMMs | attention | KV `torch.cat` |
|---|---|---|---|---|---|
| 1 | 17.8 ms | 7.2 ms (40%) | 4.2 ms | 0.3 ms | 0.7 ms |
| 16 | 22.6 ms | 16.9 ms (75%) | 4.5 ms | 2.9 ms | 6.9 ms |
| 64 | 48.5 ms | 47.2 ms (97%) | 4.7 ms | 10.1 ms | 29.0 ms |

- **Small B is CPU-launch-bound.** A loop step issues about 1,400 kernels
  from eager PyTorch on a 4-core pod, and the GPU idles 60% of the time at
  B=1. The memory-bound part, the weight GEMMs, is flat at about 4.2–4.7 ms
  across all B, as expected for decode. It is hidden under launch overhead.
- **Large B is dominated by the HF cache's `torch.cat` copy** (NOTES.md §3):
  60% of GPU time at B=64. A preallocated or paged KV cache would remove it.
  Attention reading the cache is the next-largest cost.
- Run-to-run variation: B=16 measured 22.6 ms in the profiling run against
  18.2 ms in the sweep, which is wider than the p10–p90 band within one
  run. Repeat sweeps before reading small differences.
- So the current numbers describe *HF eager Ouro on this pod*, not the
  hardware floor of a loop step. Removing launch overhead (CUDA graphs /
  `torch.compile`) and the cat copy (a static cache) would show the
  memory-bound cost. Both are natural LoopProf additions.

## Known issue (2026-10-22, resolved by moving node)

The original A10 pod's GPU began failing plain PyTorch matmuls
(`CUBLAS_STATUS_INTERNAL_ERROR`, `illegal memory access`) with no model code
involved. Repeated forward passes also gave different results. Always run
`python selfcheck.py` (must print `ALL OK`) on a new pod before trusting any
numbers.
