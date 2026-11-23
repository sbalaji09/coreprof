"""GPU fingerprint for the unified step-time formula (results/NOTEBOOK_CALIBRATION.md, "Unified formula").

    t_step = sum over stages s of [ L_s + max(bytes_s / BW_s, flops_s / Thr_s) ] + N_stages * g

The default mode measures the per-GPU inputs with generic microbenchmarks: no Ouro weights, no Ouro shapes,
no step timings. It writes fingerprint.json:
  F3 tiny-kernel chain in a CUDA graph, as the FIRST graph of this process and as later graphs: L_tiny, g
  F1 DRAM read sweep 64 KB .. 512 MB (distinct buffers, no L2 reuse): BW_read (slope), t0 (intercept)
  F2 L2-resident re-read of one buffer: BW_L2
     v2: F1 and F2 use read_sum, ONE Triton kernel per read (v1 used torch.sum, which is 2 kernels from 256 KB up)
  F4 pointer chase (dependent loads, L1 bypassed), 1 MB .. 4 GB, checked against the same chain followed on the host
  F5 dense bf16 GEMM (tensor cores) and fp32 GEMM with TF32 off (CUDA cores): Thr_tc, Thr_fp32
Fits pre-registered in the notebook: BW_read, t0 = least squares of busy time on bytes over 4..256 MB;
BW_L2 = slope over 2..16 MB.

--kernel-list: kernel names and counts of the fused + coordinate-descent step per config. Durations are
discarded in the process and never written (option K2, a secondary prediction).
--existing-checks (laptop): the 2026-11-22 diagnostics on existing L40S/H100 profiles. Those are seen data,
not inputs.
"""
import argparse
import json
import os
import subprocess
from collections import Counter

MB = 1 << 20

try:   # GPU container only; the laptop has no torch/triton (--existing-checks does not need them)
    import triton
    import triton.language as tl

    @triton.jit
    def chase(p, out, steps):
        """One thread follows a cyclic permutation; .cg skips L1, so each step is one L2 or DRAM round trip."""
        i = tl.load(p)
        for _ in range(steps):
            i = tl.load(p + i, cache_modifier=".cg")
        tl.store(out, i)

    @triton.jit
    def read_sum(x, out, n, BLOCK: tl.constexpr):
        """Streaming read: each program reads BLOCK elements once and writes one fp32 partial sum."""
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x + offs, mask=offs < n, other=0.0)
        tl.store(out + pid, tl.sum(v.to(tl.float32), axis=0))
except ImportError:
    chase = read_sum = None

READ_BLOCK = 4096   # bf16 elements per program (8 KB)


def smi():
    q = "name,memory.total,clocks.max.sm,clocks.max.mem,power.limit,clocks.sm,clocks.mem"
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"], capture_output=True,
                              text=True).stdout.strip()
    except OSError as e:
        return f"nvidia-smi failed: {e}"


def fingerprint(out_dir):
    import numpy as np
    import torch
    import delta_split as ds

    bf = torch.bfloat16
    res = dict(version=2, read_probe=f"triton read_sum, BLOCK={READ_BLOCK}", gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, smi_start=smi())
    path = os.path.join(out_dir, "fingerprint.json")

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)

    def per_op(fns, n=None):
        """Graph of fns: slot (graph time / n), busy (profiled kernel time / n), kernels per op."""
        n = n or len(fns)
        g, _ = ds.capture_graph(lambda: [f() for f in fns])
        slot = ds.stats(ds.gpu_times(g.replay, 5, 30))["median_ms"] * 1e3 / n
        bd = ds.safe_breakdown(g.replay) or {}
        del g
        return dict(slot_us=slot, busy_us=bd["busy_ms"] * 1e3 / n if bd else None,
                    kernels_per_op=bd["kernels"] / n if bd else None,
                    kernel_names=[k[:80] for k in bd.get("top_kernels_ms", {})][:3])

    def reader(b):
        out = torch.empty(triton.cdiv(b.numel(), READ_BLOCK), dtype=torch.float32, device="cuda")
        return lambda: read_sum[(out.numel(),)](b, out, b.numel(), BLOCK=READ_BLOCK)

    with torch.inference_mode():
        # F3 must run first: E6 found the first CUDA graph of a process has ~0 gap and later graphs do not.
        x = torch.zeros(1, device="cuda")
        chain = [lambda: x.add_(1)] * 256          # each add depends on the previous one
        res["tiny_chain"] = []
        for position in ("first", "second", "third"):
            r = per_op(chain)
            r["position"], r["gap_us"] = position, r["slot_us"] - r["busy_us"] if r["busy_us"] else None
            res["tiny_chain"].append(r)
            print("F3", r)
        save()

        # F1: one read_sum per buffer. Enough distinct buffers per graph that the set is >= 384 MB (beyond any L2).
        res["read_sweep"] = {}
        for k in range(16, 30):
            nbytes = 1 << k
            n = min(4096, max(8, (384 * MB) // nbytes))
            bufs = [torch.ones(nbytes // 2, dtype=bf, device="cuda") for _ in range(n)]
            r = per_op([reader(b) for b in bufs])
            r["copies"] = n
            res["read_sweep"][str(nbytes)] = r
            del bufs
            torch.cuda.empty_cache()
            print("F1", nbytes, r)
            save()
        pts = [(int(b), r["busy_us"]) for b, r in res["read_sweep"].items() if 4 * MB <= int(b) <= 256 * MB]
        slope, t0 = np.polyfit([b for b, _ in pts], [t for _, t in pts], 1)
        res["read_fit"] = dict(bw_gbps=1e-3 / slope, t0_us=t0, range="4..256 MB, busy_us")

        # F2: the same buffer read 64 times per graph, so every read after the first hits L2.
        res["l2_read"] = {}
        for nbytes in (1 * MB, 2 * MB, 4 * MB, 8 * MB, 16 * MB):
            b = torch.ones(nbytes // 2, dtype=bf, device="cuda")
            res["l2_read"][str(nbytes)] = per_op([reader(b)] * 64)
            del b
        pts = [(int(b), r["busy_us"]) for b, r in res["l2_read"].items() if int(b) >= 2 * MB]
        slope, t0 = np.polyfit([b for b, _ in pts], [t for _, t in pts], 1)
        res["l2_fit"] = dict(bw_gbps=1e-3 / slope, t0_us=t0, range="2..16 MB, busy_us")
        print("F2", res["l2_read"], res["l2_fit"])
        save()

        # F4: pointer chase, size sweep. Check: after k steps the kernel must hold the index the host reaches by
        # following the same chain (v1 gave 1 GB as fast as 1 MB, so v1's chase is not trusted).
        res["pointer_chase"] = {}
        steps, k = 20_000, 2_000
        for nbytes in (1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, 1 << 30, 4 << 30):
            n = nbytes // 8
            perm = torch.randperm(n, device="cuda")
            nxt = torch.empty(n, dtype=torch.int64, device="cuda")
            nxt[perm] = perm.roll(-1)
            del perm
            out = torch.empty(1, dtype=torch.int64, device="cuda")
            i, lines = nxt[0].item(), set()
            for _ in range(k):
                lines.add(i // 16)
                i = nxt[i].item()
            chase[(1,)](nxt, out, k)
            matches = out.item() == i
            t = ds.stats(ds.gpu_times(lambda: chase[(1,)](nxt, out, steps), 2, 5))["median_ms"]
            res["pointer_chase"][str(nbytes)] = dict(ns_per_load=t * 1e6 / steps, steps=steps,
                                                     kernel_matches_host=matches,
                                                     distinct_128B_lines_in_first_k=len(lines), k=k)
            del nxt
            torch.cuda.empty_cache()
        print("F4", res["pointer_chase"])
        save()

        # F5: large GEMMs, best of reps.
        res["gemm_peak"] = {}
        for name, dtype, tf32 in (("bf16_tensor_core", bf, False), ("fp32_cuda_core", torch.float32, False)):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a = torch.randn(8192, 8192, device="cuda", dtype=dtype)
            bm = torch.randn(8192, 8192, device="cuda", dtype=dtype)
            t = min(ds.gpu_times(lambda: a @ bm, 3, 10))
            res["gemm_peak"][name] = dict(tflops=2 * 8192 ** 3 / (t * 1e-3) / 1e12)
            del a, bm
        print("F5", res["gemm_peak"])

        r = per_op(chain)
        r["position"], r["gap_us"] = "last", r["slot_us"] - r["busy_us"] if r["busy_us"] else None
        res["tiny_chain"].append(r)
    res["smi_end"] = smi()
    save()
    print(json.dumps({k: res[k] for k in ("gpu", "read_fit", "l2_fit", "pointer_chase", "gemm_peak")}, default=str))


def kernel_list(out_dir, configs):
    from types import SimpleNamespace
    import torch
    import torch._inductor.config as ind
    from torch.profiler import ProfilerActivity, profile
    import delta_split as ds
    import harness

    ind.triton.unique_kernel_names = True
    ind.coordinate_descent_tuning = True
    args = SimpleNamespace(dtype="bf16", seed=0)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, coord_descent=True,
               note="kernel names and counts only; durations discarded before writing", points={})
    with torch.inference_mode():
        model = harness.load_model("bf16")
        L = model.model.config.num_hidden_layers
        for cfg in configs.split(","):
            B, C = map(int, cfg.split("x"))
            step, cache, _ = ds.make_step(model, B, C, L, args, "default")
            g, _ = ds.capture_graph(step)
            g.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                g.replay()
                torch.cuda.synchronize()
            names = Counter(e.name for e in prof.events()
                            if getattr(e, "device_type", None) is not None and e.device_type.name == "CUDA")
            del prof
            cats = Counter()
            for name, n in names.items():
                cats[next((c for c, keys in ds.KERNEL_CATS if any(k in name.lower() for k in keys)), "other")] += n
            res["points"][cfg] = dict(kernels=dict(names), categories=dict(cats), total=sum(names.values()))
            del g, step, cache
            torch.cuda.empty_cache()
            print(cfg, dict(cats))
            with open(os.path.join(out_dir, "kernel_list.json"), "w") as f:
                json.dump(res, f, indent=1)


def existing_checks():
    """Diagnostics on seen data (L40S design GPU and H100), recorded in the notebook on 2026-11-22."""
    import numpy as np
    import loopcore_model as lm
    from pathlib import Path
    root = Path(__file__).resolve().parent
    m = lm.OURO_1_4B
    d, ff, kv = m.d_model, m.d_ff, m.n_kv_heads * m.head_dim
    # the cd step's four weight kernels at 1x128 (names identical in both runs; NOTEBOOK O1)
    kern = {"triton_red_fused_copy_mm_1": ("qkv", d * (d + 2 * kv) * 2),
            "triton_red_fused_bmm_3": ("o", d * d * 2),
            "triton_red_fused_mm_5": ("gate+up", d * 2 * ff * 2),
            "triton_red_fused_mm_6": ("down", ff * d * 2)}
    for gpu in ("l40s", "h100"):
        kb = json.load(open(root / lm.CD_RUN[gpu] / "delta_split.json"))["kernel_breakdown"]
        top = kb["compiled_graph 1x128"]["top_kernels_ms"]
        us = {kern[k][0]: (kern[k][1], top[k] * 1e3 / m.n_layers) for k in kern}
        fit = [us[n] for n in ("qkv", "o", "gate+up")]
        slope, t0 = np.polyfit([b for b, _ in fit], [t for _, t in fit], 1)
        print(f"{gpu}: weight kernels t = t0 + bytes/BW over qkv, o, gate+up: BW {1e-3 / slope:.0f} GB/s, "
              f"t0 {t0:.2f} us")
        for n, (b, t) in us.items():
            print(f"   {n:8s} {b / 1e6:6.2f} MB  in-step {t:6.2f} us  line {t0 + b * slope:6.2f} us "
                  f"({(t0 + b * slope) / t - 1:+.0%})")
        o = lambda key: sum(v for k, v in kb[key]["top_kernels_ms"].items() if "fused_bmm_" in k) * 1e3 / m.n_layers
        o16 = np.mean([o(f"compiled_graph 16x{c}") for c in (512, 2048, 8192)])
        print(f"   o_proj per call B=1 {o('compiled_graph 1x128'):.1f} us, B=16 {o16:.1f} us; "
              f"implied re-read rate at B=16 (16 x W / (t - t0)) {16 * d * d * 2 / ((o16 - t0) * 1e-6) / 1e12:.1f} TB/s")
        att = sum(v for k, v in top.items() if "flash" in k) * 1e3 / m.n_layers
        kv_bytes = 128 * 2 * kv * 2
        bw = json.load(open(root / lm.KERNEL_BENCH_V12[gpu] / "kernel_bench.json"))["measured_read_gbps"]
        print(f"   attention 1x128 per layer {att:.2f} us; KV bytes / read peak {kv_bytes / bw / 1e3:.2f} us")
        gemm = json.load(open(root / lm.KERNEL_BENCH_V12[gpu] / "kernel_bench.json"))["gemm_compiled"]["2048x2048"]
        print(f"   lone 2048x2048 GEMM busy us by B: " + ", ".join(f"{b}: {v['busy_us']:.1f}" for b, v in gemm.items())
              + f"; + flops/datasheet at B=64 would add {2 * d * d * 64 / lm.DATASHEET[gpu]['compute'] * 1e6:.2f} us")
        for key in ("compiled_graph 1x128", "compiled_graph 16x2048"):
            print(f"   kernels {key}: {kb[key]['kernels']:.0f} (gemm {kb[key]['gemm_kernels']:.0f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir")
    p.add_argument("--kernel-list", action="store_true")
    p.add_argument("--configs", default="1x128,2x128,4x128,8x128,16x128,32x128,64x128,16x2048,16x8192")
    p.add_argument("--existing-checks", action="store_true")
    a = p.parse_args()
    if a.existing_checks:
        return existing_checks()
    if not a.out_dir:
        p.error("--out-dir is required")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.kernel_list:
        kernel_list(a.out_dir, a.configs)
    else:
        fingerprint(a.out_dir)


if __name__ == "__main__":
    main()
