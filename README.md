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
`oom_prefill`.

## Extending (LoopProf)

- A new thing to time: write `build_<name>(model, state, args) -> callable`
  in `harness.py` and add it to `TARGETS`.
- A new sweep axis: add a CLI arg and extend `sweep()`.
- A new metric: compute it in `run_config()` and add the column to
  `CSV_FIELDS`.

## Known issue (2026-10-22)

During setup the pod's A10 began failing plain PyTorch matmuls
(`CUBLAS_STATUS_INTERNAL_ERROR` and `illegal memory access`, with no model
code involved). Repeated forward passes also gave different results. No
timing data has been collected yet. Restart the pod (or move to another node)
and make sure `python selfcheck.py` prints `ALL OK` before running the sweep.
