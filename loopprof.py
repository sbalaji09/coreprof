"""LoopProf v0: time one recurrent step of Ouro-1.4B over a batch x context grid, and say
how many bytes each step moves and what fraction of the GPU's memory bandwidth that is.

Built on harness.py (same prefill, same ut_step/decode_step targets, same CUDA-event timing).
For every (B, C) config it records:
  - ut_step and decode_step time: median, p10, p90, IQR over --reps (after --warmup)
  - tokens/s = B / decode_step time
  - bytes moved per ut_step, from an explicit byte model (see byte_model() and NOTES.md):
      weights of the looped block (24 layers + final norm + exit gate), read once per step
      + KV read: B * C * layers * 2 (K,V) * kv_heads * head_dim * dtype bytes   (one loop's cache)
      + KV write: the new token's K,V for each layer
    Activations are ignored (decode: B x hidden per layer, < 0.1 % of the total here).
  - achieved bandwidth = bytes / ut_step time, as a fraction of
      (a) the GPU's spec peak (SPEC_GBPS, sources in the comments) and
      (b) a measured achievable peak: a large device-to-device copy (read + write bytes / time).
Configs whose estimated memory won't fit are skipped (status=skipped_mem) instead of attempted.
Each finished config is appended to <out>/loopprof.csv immediately; re-running with the same
--out-dir skips configs already there, so a killed job resumes where it stopped.

Usage: python loopprof.py --out-dir results/loopprof_<gpu> [--batch-sizes ...] [--context-lens ...]
Then:  python fit_loopprof.py results/loopprof_<gpu>/loopprof.csv
"""
import argparse
import csv
import datetime as dt
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import harness

# Spec peak DRAM bandwidth in GB/s (1e9 bytes/s).
SPEC_GBPS = {
    "L40S": 864,        # NVIDIA L40S datasheet: 864 GB/s GDDR6
    "L40": 864,         # NVIDIA L40 datasheet
    "RTX A6000": 768,   # NVIDIA RTX A6000 datasheet: 768 GB/s GDDR6
    "A10": 600,         # NVIDIA A10 datasheet: 600 GB/s
    "A100-SXM4-80GB": 2039, "A100 80GB PCIe": 1935, "A100-SXM4-40GB": 1555, "A100-PCIE-40GB": 1555,
    "H100 80GB HBM3": 3350, "H100 PCIe": 2000,
}

FIELDS = ["B", "C", "status", "ut_median_ms", "ut_p10_ms", "ut_p90_ms", "ut_iqr_ms",
          "decode_median_ms", "decode_p10_ms", "decode_p90_ms", "tokens_per_s",
          "weight_bytes", "kv_read_bytes", "kv_write_bytes", "bytes_per_step",
          "achieved_gbps", "util_spec", "util_measured", "peak_mem_gb", "sm_clock_mhz", "temp_c",
          "wall_s"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--context-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=harness.DTYPES, default="bf16")
    p.add_argument("--ut-step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem-frac", type=float, default=0.80,
                   help="skip configs whose estimated memory exceeds this fraction of GPU memory")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- byte model

def byte_model(model, dtype):
    m, cfg = model.model, model.model.config
    elem = torch.tensor([], dtype=dtype).element_size()
    L = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    looped = list(m.layers[:L]) + [m.norm, m.early_exit_gate]
    weight_bytes = sum(p.numel() * p.element_size() for mod in looped for p in mod.parameters())
    kv_per_token_per_loop = L * 2 * kv_heads * head_dim * elem    # bytes for one token, one loop
    return SimpleNamespace(
        elem=elem, layers=L, kv_heads=kv_heads, head_dim=head_dim, loops=m.total_ut_steps,
        weight_bytes=weight_bytes, kv_per_token_per_loop=kv_per_token_per_loop,
        total_param_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
    )


def step_bytes(bm, B, C):
    kv_read = B * C * bm.kv_per_token_per_loop
    kv_write = B * bm.kv_per_token_per_loop
    return bm.weight_bytes, kv_read, kv_write


def est_mem_bytes(bm, B, C):
    """Rough peak: all weights + the prefilled cache for every loop + prefill activations."""
    kv_all_loops = B * (C + 1) * bm.kv_per_token_per_loop * bm.loops
    prefill_act = B * C * (2048 + 2 * 5632) * bm.elem * 3   # hidden + MLP intermediates, generous
    return bm.total_param_bytes + kv_all_loops + prefill_act


# ---------------------------------------------------------------- measured peak bandwidth

def measured_peak_gbps(nbytes=2 << 30, reps=20):
    """Device-to-device copy: counts read + write bytes. Best of reps (achievable peak)."""
    n = nbytes // 2
    src = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for i in range(reps + 3):
        torch.cuda.synchronize()
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        if i >= 3:
            best = min(best, start.elapsed_time(end))
    del src, dst
    harness.free_cuda()
    return 2 * nbytes / (best / 1e3) / 1e9


def spec_gbps(name):
    for key in sorted(SPEC_GBPS, key=len, reverse=True):
        if key in name:
            return SPEC_GBPS[key]
    return None


# ---------------------------------------------------------------- sweep

def done_configs(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(int(r["B"]), int(r["C"])) for r in csv.DictReader(f)}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "loopprof.csv")
    meta_path = os.path.join(args.out_dir, "loopprof_meta.json")
    raw_path = os.path.join(args.out_dir, "loopprof_raw.jsonl")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = harness.load_model(args.dtype)
    dtype = harness.DTYPES[args.dtype]
    bm = byte_model(model, dtype)
    gpu = torch.cuda.get_device_name()
    total_mem = torch.cuda.get_device_properties(0).total_memory
    peak_meas = measured_peak_gbps()
    peak_spec = spec_gbps(gpu)

    hargs = SimpleNamespace(targets=["ut_step", "decode_step"], ut_step=args.ut_step, seed=args.seed,
                            warmup=args.warmup, reps=args.reps)
    meta = harness.run_metadata(hargs, model)
    meta.update(
        tool="loopprof v0", args=vars(args), gpu_total_mem_gb=total_mem / 2**30,
        spec_peak_gbps=peak_spec, measured_peak_gbps=peak_meas,
        byte_model={k: v for k, v in vars(bm).items()},
        byte_model_notes=("bytes/ut_step = weight_bytes (looped block: 24 layers + norm + exit gate) "
                          "+ B*C*kv_per_token_per_loop (KV read, one loop) + B*kv_per_token_per_loop "
                          "(KV write); activations ignored"),
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"GPU {gpu}: spec peak {peak_spec} GB/s, measured copy peak {peak_meas:.0f} GB/s, "
          f"{total_mem / 2**30:.1f} GiB")
    print(f"byte model: looped-block weights {bm.weight_bytes / 1e9:.3f} GB, "
          f"KV {bm.kv_per_token_per_loop / 1e3:.1f} kB per token per loop")

    done = done_configs(csv_path)
    if done:
        print(f"resuming: {len(done)} configs already in {csv_path}")
    new_file = not os.path.exists(csv_path)
    with torch.inference_mode(), open(csv_path, "a", newline="") as f, open(raw_path, "a") as rawf:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for C in args.context_lens:
            for B in args.batch_sizes:
                if (B, C) in done:
                    continue
                wb, kvr, kvw = step_bytes(bm, B, C)
                row = {"B": B, "C": C, "weight_bytes": wb, "kv_read_bytes": kvr,
                       "kv_write_bytes": kvw, "bytes_per_step": wb + kvr + kvw}
                if est_mem_bytes(bm, B, C) > args.mem_frac * total_mem:
                    row["status"] = "skipped_mem"
                    w.writerow(row)
                    f.flush()
                    print(f"B={B:<3} C={C:<5} skipped (est. {est_mem_bytes(bm, B, C) / 2**30:.0f} GiB)")
                    continue
                t0 = time.time()
                rows, raw = harness.run_config(model, hargs, C, B)
                byt = {r["target"]: r for r in rows}
                ut, dec = byt["ut_step"], byt["decode_step"]
                row["wall_s"] = round(time.time() - t0, 1)
                if ut.get("status") != "ok":
                    row["status"] = ut.get("status", "error")
                else:
                    gbps = row["bytes_per_step"] / (ut["median_ms"] / 1e3) / 1e9
                    row.update(
                        status="ok", ut_median_ms=ut["median_ms"], ut_p10_ms=ut["p10_ms"],
                        ut_p90_ms=ut["p90_ms"], ut_iqr_ms=ut["iqr_ms"],
                        peak_mem_gb=ut["peak_mem_gb"], sm_clock_mhz=ut["sm_clock_mhz"], temp_c=ut["temp_c"],
                        achieved_gbps=gbps, util_measured=gbps / peak_meas,
                        util_spec=(gbps / peak_spec) if peak_spec else None,
                    )
                    if dec.get("status") == "ok":
                        row.update(decode_median_ms=dec["median_ms"], decode_p10_ms=dec["p10_ms"],
                                   decode_p90_ms=dec["p90_ms"], tokens_per_s=dec["tokens_per_s"])
                    rawf.write(json.dumps({"B": B, "C": C, **{k: v for k, v in raw.items()}}) + "\n")
                    rawf.flush()
                w.writerow(row)
                f.flush()
                if row["status"] == "ok":
                    print(f"B={B:<3} C={C:<5} ut={row['ut_median_ms']:8.3f} ms "
                          f"[{row['ut_p10_ms']:.2f}-{row['ut_p90_ms']:.2f}]  "
                          f"decode={row.get('decode_median_ms', float('nan')):8.2f} ms  "
                          f"{row['bytes_per_step'] / 1e9:6.2f} GB/step  {gbps:6.0f} GB/s  "
                          f"util {100 * row['util_measured']:5.1f}% of measured peak  ({row['wall_s']} s)")
                else:
                    print(f"B={B:<3} C={C:<5} {row['status']}")
    print(f"wrote {csv_path}, {meta_path}, {raw_path}")


if __name__ == "__main__":
    main()
