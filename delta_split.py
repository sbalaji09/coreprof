"""Week 5: split δ (the fixed cost of one recurrent step) into weight transfer vs software overhead.

Everything runs in ONE process on ONE GPU, because Week 4 showed host overhead differs between
machines. Four measurements:

1. Launch microbenchmarks
   - eager: wall-clock cost per tiny kernel launch (x.add_(1) on 1 element), host-bound
   - graph: the same ops captured in a CUDA graph and replayed (per-op cost inside a graph)
   - kernels per step: torch.profiler count of GPU kernels / launches in one recurrent step

2. Three implementations of the SAME recurrent step (24 layers + norm + exit gate, loop 0):
   - eager_hf      harness.build_ut_step: HF UniversalTransformerCache (what LoopProf measured;
                   HF appends each token with torch.cat, so every step copies the cache)
   - eager_static  same layers, but a static KV buffer written in place (no cache copy)
   - graph_static  eager_static captured once as a CUDA graph and replayed (no per-kernel launch,
                   no Python): the best-case GPU time for this step
   at several (B, C) points (small B·C isolates δ; larger B·C re-measures the KV slope α').

3. Layer-count sweep: graph_static and eager_static with the first L' of the 24 layers
   (L' in --layers) at B=1, C=128. Weights scale with L', so the slope (ms per layer) divided into
   bytes per layer gives the effective weight-streaming bandwidth, and the intercept is the
   per-step cost that does not depend on weights.

4. Predicted weight-fetch time = looped-block weight bytes / bandwidth, for both the spec peak and
   a measured copy peak (Vaibhav's hardware-sheet number should match the spec version).

Outputs (in --out-dir): delta_split.json (everything), delta_split_steps.csv, delta_split.md
(the decomposition table and a first, explicitly bounded, estimate of what on-chip weights save).

--grid (v1) also measures: a read-only bandwidth peak (the reference for weight/KV reads; the copy
peak pays for writes too), the kernel breakdown at B=1,C=128 and B=16,C in {512,2048,8192}
(attention µs per cached token should match the fitted α'), and the once-per-token LM head.
delta_split_grid.md starts with a "Calibration inputs" table for the analytical model.

Usage: python delta_split.py --out-dir results/delta_split_<gpu> [--quick]
"""
import argparse
import csv
import json
import math
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import harness
import loopprof


GRID_MEM_FRAC = 0.65   # 80 GB GPUs: 51.6 GB KV for B=64,C=4096 and B=32,C=8192 fits; 48 GB L40S: same 46 points


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="1x128,1x2048,8x512,16x2048,32x1024,4x8192",
                   help="comma-separated BxC points for the three-implementation comparison")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 12, 18, 24])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=harness.DTYPES, default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="few reps/configs: a cheap end-to-end check")
    p.add_argument("--grid", action="store_true",
                   help="v1: skip the launch/3-impl/layer parts; profile kernels and sweep the CUDA-graph "
                        "static-cache step over --grid-batch x --grid-ctx, then fit δ + γB + α'BC")
    p.add_argument("--grid-batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--grid-ctx", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--grid-mem-gb", type=float, default=None,
                   help=f"skip grid points whose static KV exceeds this (always capped at {GRID_MEM_FRAC} x GPU memory; "
                        "default: that cap)")
    p.add_argument("--compile", action="store_true",
                   help="with --grid: torch.compile the static step (kernel fusion), then CUDA-graph it "
                        "(impl compiled_graph); also times the uncompiled graph at a few points, same container")
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode for the grid (default keeps cuBLAS GEMMs, so the breakdown classifies them)")
    p.add_argument("--int8", action="store_true",
                   help="with --grid --compile: int8 weight-only linears in the looped block (per-output-channel "
                        "scales, gpt-fast style; the cast is fused into the matmul by torch.compile); also times the "
                        "bf16 compiled step at a few points in the same container")
    p.add_argument("--int8-impl", choices=["cast", "triton"], default="cast",
                   help="cast: F.linear(x, w.to(bf16)) * s, fusion left to torch.compile (not fused in torch 2.5.1); "
                        "triton: int8_kernel.py, reads int8 weights directly")
    p.add_argument("--coord-descent", action="store_true",
                   help="with --compile: inductor coordinate_descent_tuning (gpt-fast's setting; fuses the int8 cast "
                        "into a reduction at B=1). Applies to every compile in the run, incl. the bf16 reference")
    p.add_argument("--profile-configs", default=None,
                   help="BxC list: step time + full kernel breakdown per point (step_profile.json); honours --coord-descent")
    p.add_argument("--noise-check", action="store_true",
                   help="only: fused (torch.compile + CUDA graph) step at B=1, C=128: step time, weight-kernel time and "
                        "their difference, for measuring container-to-container noise; writes noise.json")
    p.add_argument("--kernel-bench", action="store_true",
                   help="only: per-kernel microbenchmarks in CUDA graphs (weight GEMV/GEMM per Ouro shape and batch, "
                        "small fused kernels, decode attention) for loopcore_model's GPU term; writes kernel_bench.json")
    p.add_argument("--int8-probe", action="store_true",
                   help="only: bf16 compiled vs int8 variants (inductor settings, Triton kernel) at a few points; "
                        "writes int8_probe.json")
    p.add_argument("--compile-check", action="store_true",
                   help="only: compare eager-bf16 and compiled-bf16 static steps against an fp32 reference "
                        "(is the compiled step's difference rounding, or a bug?); writes compile_numerics.json")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    if a.compile and not a.grid:
        p.error("--compile needs --grid")
    if a.int8 and not a.compile:
        p.error("--int8 needs --grid --compile (without compile the int8->bf16 cast is not fused and reads MORE bytes)")
    if a.quick:
        a.configs, a.layers, a.reps, a.warmup = "1x128,8x512", [6, 24], 10, 3
    a.configs = [tuple(int(v) for v in c.lower().split("x")) for c in a.configs.split(",")]
    return a


# ---------------------------------------------------------------- timing helpers

def gpu_times(fn, warmup, reps):
    """CUDA-event time per call (ms). The queue is drained before each start, as in harness.py."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for i in range(warmup + reps):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            out.append(s.elapsed_time(e))
    return out


def stats(t):
    a = np.asarray(t, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(a, [10, 25, 50, 75, 90])
    return dict(median_ms=float(p50), p10_ms=float(p10), p90_ms=float(p90), iqr_ms=float(p75 - p25),
                n=len(a))


def measured_read_gbps(nbytes=2 << 30, reps=20):
    """Read-only peak: torch.sum over a 2 GiB bf16 tensor, bytes read / time, best of reps after 3 warmups.
    The right reference for weight and KV reads (loopprof.measured_peak_gbps is a copy: read + write)."""
    x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for i in range(reps + 3):
        torch.cuda.synchronize()
        s.record()
        torch.sum(x)
        e.record()
        torch.cuda.synchronize()
        if i >= 3:
            best = min(best, s.elapsed_time(e))
    del x
    harness.free_cuda()
    return nbytes / (best / 1e3) / 1e9


# ---------------------------------------------------------------- 1. launch microbenchmarks

def launch_microbench(n=20000, graph_ops=2000):
    x = torch.zeros(1, device="cuda")
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    # eager: host wall time per launch (the CPU side is the bottleneck for tiny kernels)
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    host_us = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    total_us = (time.perf_counter() - t0) / n * 1e6
    # eager, GPU-side view: events around the burst
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        x.add_(1)
    e.record()
    torch.cuda.synchronize()
    eager_gpu_us = s.elapsed_time(e) * 1e3 / n
    # graph: capture graph_ops tiny ops, replay
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        for _ in range(graph_ops):
            x.add_(1)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    reps = 20
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    graph_us = s.elapsed_time(e) * 1e3 / (reps * graph_ops)
    del g
    return dict(eager_host_us_per_launch=host_us, eager_total_us_per_launch=total_us,
                eager_gpu_us_per_launch=eager_gpu_us, graph_us_per_op=graph_us,
                note="eager_host = CPU time to issue one tiny kernel; graph = per-op cost when replayed "
                     "from a CUDA graph (GPU-side, no CPU involvement)")


def count_kernels(fn, steps=5):
    """GPU kernel launches per call of fn, via torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    ev = prof.events()
    gpu = [e for e in ev if getattr(e, "device_type", None) is not None and e.device_type.name == "CUDA"]
    launches = [e for e in ev if e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC",
                                              "cudaGraphLaunch", "cudaMemcpyAsync", "cudaMemsetAsync")]
    return dict(gpu_ops_per_step=len(gpu) / steps, launch_calls_per_step=len(launches) / steps)


KERNEL_CATS = [   # order matters: flash-attention kernel names contain "cutlass", so match attention first
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas", "sm80_", "sm89_", "sm90_", "ampere_", "ada_",
              "_mm_", "fused_bmm_")),   # "_mm_": inductor's fused matmul reductions (coordinate_descent_tuning),
    # e.g. triton_red_fused_mm_5; "fused_bmm_": o_proj lowered as bmm (triton_red_fused_bmm_3), added 2026-11-22.
    # Not plain "_bmm_": RMSNorm kernels are named triton_red_fused__to_copy_add_bmm_mean_... (they stay "copy").
    ("copy", ("copy", "memcpy", "memset", "cat", "index")),
]
KERNEL_CATS_VERSION = 2   # 1 = before "fused_bmm_"; calibrate.weight_kernel_ms re-sorts v1 breakdowns


def kernel_breakdown(fn, steps=5):
    """GPU-side kernel time per call of fn, grouped by kind (from kernel durations, so it does not
    include host gaps). gemm ≈ weight streaming; the rest is norms, activations, rotary, etc."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    tot, n, top = {}, {}, {}
    for e in prof.events():
        if getattr(e, "device_type", None) is None or e.device_type.name != "CUDA":
            continue
        us = e.time_range.elapsed_us()
        name = e.name.lower()
        cat = next((c for c, keys in KERNEL_CATS if any(k in name for k in keys)), "other")
        tot[cat] = tot.get(cat, 0.0) + us
        n[cat] = n.get(cat, 0) + 1
        top[e.name] = top.get(e.name, 0.0) + us
    res = {f"{c}_ms": tot[c] / steps / 1e3 for c in tot}
    res.update({f"{c}_kernels": n[c] / steps for c in n})
    res["busy_ms"] = sum(tot.values()) / steps / 1e3
    res["kernels"] = sum(n.values()) / steps
    res["kernel_cats_version"] = KERNEL_CATS_VERSION
    res["top_kernels_ms"] = {k: v / steps / 1e3 for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:12]}
    return res


def safe_breakdown(fn):
    try:
        return kernel_breakdown(fn)
    except Exception as e:
        print("kernel breakdown failed:", e)
        return None


def safe_count(fn):
    try:
        return count_kernels(fn)
    except Exception as e:  # profiler/CUPTI unavailable: timing still counts
        print("kernel count failed:", e)
        return {}


# ---------------------------------------------------------------- 2. static-cache step

class StaticCache:
    """Minimal stand-in for HF's cache inside OuroAttention for ONE recurrent step at loop 0.

    Per layer it holds K, V of shape [B, H, C+1, D]: positions 0..C-1 are the prefilled context,
    position C is where the new token's k/v is written IN PLACE. update() returns views of the full
    C+1 positions, so attention reads exactly the same bytes as with HF's cache, but nothing is
    reallocated or copied. Values are random: decode timing does not depend on them.
    Ouro calls update(k, v, current_ut * n_layers + layer_idx); we run loop 0 only.
    """

    def __init__(self, n_layers, B, C, kv_heads, head_dim, dtype, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        shape = (B, kv_heads, C + 1, head_dim)
        self.k = [torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)
                  for _ in range(n_layers)]
        self.v = [torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)
                  for _ in range(n_layers)]
        self.C, self.n_layers = C, n_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        l = layer_idx % self.n_layers
        self.k[l][:, :, self.C:self.C + 1].copy_(key_states)
        self.v[l][:, :, self.C:self.C + 1].copy_(value_states)
        return self.k[l], self.v[l]


def build_static_step(model, B, C, n_layers, dtype, seed):
    m = model.model
    L = m.config.num_hidden_layers
    bm = loopprof.byte_model(model, harness.DTYPES[dtype])
    cache = StaticCache(L, B, C, bm.kv_heads, bm.head_dim, harness.DTYPES[dtype], seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    ids = torch.randint(0, model.config.vocab_size, (B, 1), device="cuda", generator=g)
    hidden_in = m.embed_tokens(ids).detach().clone()          # static input buffer [B, 1, hidden]
    position_ids = torch.full((B, 1), C, device="cuda", dtype=torch.long)
    position_embeddings = m.rotary_emb(hidden_in, position_ids)
    layers = m.layers[:n_layers]

    def step():
        h = hidden_in
        for layer in layers:
            h = layer(h, attention_mask=None, position_ids=position_ids, past_key_value=cache,
                      use_cache=True, cache_position=None, position_embeddings=position_embeddings,
                      current_ut=0)
        h = m.norm(h)
        return h, m.early_exit_gate(h)

    return step, cache


def capture_graph(step):
    """Capture step() into a CUDA graph. Returns replay(); raises if capture is not possible."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):          # warm up allocator / cuBLAS handles off the capture stream
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    torch.cuda.synchronize()
    return g, out


def make_step(model, B, C, L, args, mode=None):
    """build_static_step, optionally torch.compile'd (mode=None: plain eager). Compiles on the first call
    here, so later timing never includes compile time. dynamo is reset per shape: each (B, C) gets its own
    static-shape compile, and the recompile limit (default 8) can never silently fall back to eager."""
    step, cache = build_static_step(model, B, C, L, args.dtype, args.seed)
    info = {}
    if mode:
        import torch._dynamo
        torch._dynamo.reset()
        step = torch.compile(step, mode=mode, dynamic=False)
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        info["compile_s"] = round(time.perf_counter() - t0, 1)
        try:
            from torch._dynamo.utils import counters
            info["graph_breaks"] = sum(counters["graph_break"].values())
        except Exception:
            pass
    return step, cache, info


class Int8WeightOnlyLinear(torch.nn.Module):
    """int8 weights + one bf16 scale per output channel (symmetric). forward() casts the int8 weight to the
    activation dtype; under torch.compile that cast is fused into the matmul, so only int8 bytes are read.
    Parameters (not buffers) so loopprof.byte_model counts their bytes.
    impl (class-wide): "cast" leaves fusion to torch.compile; "triton" calls int8_kernel's custom op."""
    impl = "cast"

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
        self.weight = torch.nn.Parameter((w / scale).round().clamp(-128, 127).to(torch.int8), requires_grad=False)
        self.scales = torch.nn.Parameter(scale.squeeze(1).to(lin.weight.dtype), requires_grad=False)
        self.bias = lin.bias

    def forward(self, x):
        if Int8WeightOnlyLinear.impl == "triton":
            y = torch.ops.loopcore.int8_wo_linear(x, self.weight, self.scales)
        else:
            y = torch.nn.functional.linear(x, self.weight.to(x.dtype)) * self.scales
        return y if self.bias is None else y + self.bias


def set_int8_impl(impl):
    if impl == "triton":
        import int8_kernel  # noqa: F401  (registers torch.ops.loopcore.int8_wo_linear)
    Int8WeightOnlyLinear.impl = impl


def noise_check(args):
    """One sample of the B=1, C=128 fused step and its weight/non-weight split (one container = one sample)."""
    import torch._inductor.config as ind
    ind.triton.unique_kernel_names = True
    with torch.inference_mode():
        model = harness.load_model(args.dtype)
        L = model.model.config.num_hidden_layers
        step, cache, info = make_step(model, 1, 128, L, args, "default")
        g, _ = capture_graph(step)
        t = stats(gpu_times(g.replay, args.warmup, args.reps))
        bd = safe_breakdown(g.replay) or {}
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, step=t, gemm_ms=bd.get("gemm_ms"),
               busy_ms=bd.get("busy_ms"), kernels=bd.get("kernels"),
               non_weight_ms=t["median_ms"] - bd["gemm_ms"] if bd.get("gemm_ms") else None, **info)
    with open(os.path.join(args.out_dir, "noise.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, default=str))


def step_profile(args):
    """Step time + full kernel breakdown of the fused step at each --profile-configs point (one container), with
    SM clock and power sampled by nvidia-smi while the graph replays (tests clock throttling at large B)."""
    import subprocess
    import torch._inductor.config as ind
    ind.triton.unique_kernel_names = True
    if args.coord_descent:
        ind.coordinate_descent_tuning = True
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, coord_descent=args.coord_descent, points={})
    path = os.path.join(args.out_dir, "step_profile.json")
    with torch.inference_mode():
        model = harness.load_model(args.dtype)
        L = model.model.config.num_hidden_layers
        for cfg in args.profile_configs.split(","):
            B, C = map(int, cfg.split("x"))
            step, cache, info = make_step(model, B, C, L, args, "default")
            g, _ = capture_graph(step)
            t = stats(gpu_times(g.replay, args.warmup, args.reps))
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw",
                                    "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
            t_end = time.time() + 2.0
            while time.time() < t_end:
                g.replay()
            torch.cuda.synchronize()
            smi.terminate()
            samples = [[float(x) for x in ln.split(",")] for ln in smi.communicate()[0].splitlines() if ln.strip()]
            res["points"][cfg] = dict(step=t, breakdown=safe_breakdown(g.replay), smi_sm_mhz_max_mhz_watts=samples,
                                      **info)
            del g, step, cache
            torch.cuda.empty_cache()
            with open(path, "w") as f:
                json.dump(res, f, indent=1, default=str)
            print(cfg, f"{t['median_ms']:.3f} ms", samples[-1:] if samples else "no nvidia-smi samples")


def kernel_bench(args, copies=24, batches=(1, 2, 4, 8, 16, 32, 64)):
    """Per-kernel costs measured in isolation, the inputs of loopcore_model's kernel-sum GPU model.

    Each op is captured `copies` times in one CUDA graph on distinct tensors (like 24 layers: no L2 reuse of
    weights or KV), replayed, and reported per op as slot_us (graph time / ops: kernel + launch gap) and busy_us
    (profiled kernel time / ops). No model weights or step timings are involved: synthetic tensors with Ouro's
    shapes (hidden 2048, MLP 5632, 16 heads x 128, bf16)."""
    import torch._inductor.config as ind
    ind.triton.unique_kernel_names = True
    dev, bf = "cuda", torch.bfloat16
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, copies=copies,
               gemm={}, small={}, attention={}, errors={})
    path = os.path.join(args.out_dir, "kernel_bench.json")

    def per_op(fns, n=None, top=False):
        """n: ops to divide by (default len(fns))."""
        n = n or len(fns)

        def run():
            for fn in fns:
                fn()
        g, _ = capture_graph(run)
        slot = stats(gpu_times(g.replay, args.warmup, args.reps))["median_ms"] * 1e3 / n
        bd = safe_breakdown(g.replay)
        busy = bd["busy_ms"] * 1e3 / n if bd else None
        kernels = bd["kernels"] / n if bd else None
        del g
        r = dict(slot_us=slot, busy_us=busy, kernels_per_op=kernels)
        if top and bd:
            r["top_kernels"] = [k[:90] for k in list(bd["top_kernels_ms"])[:3]]
        return r

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1)

    with torch.inference_mode():
        res["measured_peak_gbps"] = loopprof.measured_peak_gbps()
        res["measured_read_gbps"] = measured_read_gbps()
        # weight matmuls: q/k/v/o 2048x2048, gate/up 5632x2048, down 2048x5632 (out x in), as F.linear on [B, 1, in]
        for N, K in ((2048, 2048), (5632, 2048), (2048, 5632)):
            key = f"{N}x{K}"
            try:
                ws = [torch.randn(N, K, device=dev, dtype=bf) for _ in range(copies)]
                for B in batches:
                    x = torch.randn(B, 1, K, device=dev, dtype=bf)
                    res["gemm"].setdefault(key, {})[str(B)] = per_op([lambda w=w, x=x: torch.nn.functional.linear(x, w)
                                                                      for w in ws])
                del ws
            except Exception:
                res["errors"][f"gemm {key}"] = traceback.format_exc().strip().splitlines()[-1]
            harness.free_cuda()
            print("gemm", key, json.dumps(res["gemm"].get(key), default=str)[:300])
            save()
        # v1.2 (pre-registered): weight matmuls as torch.compile issues them, compiled fresh per (shape, B)
        import torch._dynamo as dynamo   # (not `import torch._dynamo`: that would make `torch` local here)
        for N, K in ((2048, 2048), (5632, 2048), (2048, 5632)):
            key = f"{N}x{K}"
            try:
                ws = [torch.randn(N, K, device=dev, dtype=bf) for _ in range(copies)]
                for B in batches:
                    dynamo.reset()
                    lin = torch.compile(lambda x, w: torch.nn.functional.linear(x, w), dynamic=False)
                    x = torch.randn(B, 1, K, device=dev, dtype=bf)
                    res.setdefault("gemm_compiled", {}).setdefault(key, {})[str(B)] = per_op(
                        [lambda w=w, x=x: lin(x, w) for w in ws], top=B in (1, 16, 64))
                del ws
            except Exception:
                res["errors"][f"gemm_compiled {key}"] = traceback.format_exc().strip().splitlines()[-1]
            harness.free_cuda()
            print("gemm_compiled", key, json.dumps(res.get("gemm_compiled", {}).get(key), default=str)[:300])
            save()
        # small fused kernels: compiled RMSNorm over hidden (one Triton reduction) and SiLU*mul over the MLP width
        # (one Triton pointwise kernel): representative of the fused step's ~123 non-GEMM, non-attention kernels
        rms = torch.compile(lambda x, w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6))
                            .to(bf) * w, dynamic=False)
        silu = torch.compile(lambda a, b: torch.nn.functional.silu(a) * b, dynamic=False)
        for B in (1, 16, 64):
            try:
                w = torch.randn(2048, device=dev, dtype=bf)
                xs = [torch.randn(B, 1, 2048, device=dev, dtype=bf) for _ in range(copies)]
                a_s = [torch.randn(B, 1, 5632, device=dev, dtype=bf) for _ in range(copies)]
                b_s = [torch.randn(B, 1, 5632, device=dev, dtype=bf) for _ in range(copies)]
                res["small"].setdefault("rmsnorm_2048", {})[str(B)] = per_op([lambda x=x: rms(x, w) for x in xs])
                res["small"].setdefault("silu_mul_5632", {})[str(B)] = per_op(
                    [lambda a=a, b=b: silu(a, b) for a, b in zip(a_s, b_s)])
                # v1.2 (pre-registered): small kernels in context, alternating with the 2048x2048 GEMV as in the
                # step; per small kernel = (pairs - GEMV alone) / copies
                ws = [torch.randn(2048, 2048, device=dev, dtype=bf) for _ in range(copies)]
                xg = torch.randn(B, 1, 2048, device=dev, dtype=bf)
                alone = per_op([lambda w=w: torch.nn.functional.linear(xg, w) for w in ws])
                for name, fns in (("rmsnorm_2048", [lambda x=x: rms(x, w) for x in xs]),
                                  ("silu_mul_5632", [lambda a=a, b=b: silu(a, b) for a, b in zip(a_s, b_s)])):
                    seq = [f for pair in zip([lambda w=w: torch.nn.functional.linear(xg, w) for w in ws], fns) for f in pair]
                    pr = per_op(seq, n=copies)
                    res.setdefault("small_after_gemv", {}).setdefault(name, {})[str(B)] = dict(
                        slot_us=pr["slot_us"] - alone["slot_us"],
                        busy_us=(pr["busy_us"] - alone["busy_us"]) if pr["busy_us"] and alone["busy_us"] else None,
                        pair_slot_us=pr["slot_us"], gemv_alone_slot_us=alone["slot_us"])
                del xs, a_s, b_s, ws
            except Exception:
                res["errors"][f"small B={B}"] = traceback.format_exc().strip().splitlines()[-1]
            harness.free_cuda()
        print("small", json.dumps(res["small"], default=str)[:400])
        save()
        # decode attention: SDPA (flash) with one query over C+1 cached positions, 16 heads x 128, per layer
        for B, C in ((1, 128), (16, 128), (64, 128), (1, 2048), (16, 2048)):
            try:
                q = torch.randn(B, 16, 1, 128, device=dev, dtype=bf)
                kv = [(torch.randn(B, 16, C + 1, 128, device=dev, dtype=bf), torch.randn(B, 16, C + 1, 128, device=dev, dtype=bf))
                      for _ in range(copies)]
                res["attention"][f"{B}x{C}"] = per_op([lambda k=k, v=v: torch.nn.functional.scaled_dot_product_attention(q, k, v)
                                                      for k, v in kv])
                del kv
            except Exception:
                res["errors"][f"attention {B}x{C}"] = traceback.format_exc().strip().splitlines()[-1]
            harness.free_cuda()
        print("attention", json.dumps(res["attention"], default=str)[:400])
        save()
    return res


def int8_probe(args, points=((1, 128), (16, 2048), (64, 128))):
    """Which int8 path actually reads int8 bytes? bf16 compiled reference vs int8 variants, same container.
    Weight time per variant at B=1, C=128 is estimated as t_int8 - (t_bf16 - w_bf16)."""
    import torch._inductor.config as ind
    ind.triton.unique_kernel_names = True
    variants = [("cast, default", "cast", {}),
                ("cast, mixed_mm_choice=triton", "cast", {"mixed_mm_choice": "triton"}),
                ("cast, coordinate_descent_tuning", "cast", {"coordinate_descent_tuning": True}),
                ("cast, both", "cast", {"mixed_mm_choice": "triton", "coordinate_descent_tuning": True}),
                ("triton int8 kernel", "triton", {})]
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, points=[f"{B}x{C}" for B, C in points],
               bf16={}, variants={}, errors={})
    path = os.path.join(args.out_dir, "int8_probe.json")

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)

    def time_point(model, B, C, L, breakdown):
        step, cache, info = make_step(model, B, C, L, args, "default")
        g, _ = capture_graph(step)
        t = stats(gpu_times(g.replay, args.warmup, args.reps))["median_ms"]
        bd = safe_breakdown(g.replay) if breakdown else None
        del g, step, cache
        harness.free_cuda()
        return t, bd, info

    with torch.inference_mode():
        model = harness.load_model(args.dtype)
        L = model.model.config.num_hidden_layers
        for B, C in points:
            t, bd, _ = time_point(model, B, C, L, (B, C) == (1, 128))
            res["bf16"][f"{B}x{C}"] = t
            if bd:
                res["bf16_breakdown_1x128"] = bd
        res["bf16_weight_bytes"] = loopprof.byte_model(model, harness.DTYPES[args.dtype]).weight_bytes
        res["linears_quantized"] = quantize_int8(model)
        res["int8_weight_bytes"] = loopprof.byte_model(model, harness.DTYPES[args.dtype]).weight_bytes
        try:
            set_int8_impl("triton")
            import int8_kernel
            res["triton_kernel_check"] = int8_kernel.check()
        except Exception:
            res["errors"]["triton kernel check"] = traceback.format_exc()
        save()
        w16 = (res.get("bf16_breakdown_1x128") or {}).get("gemm_ms")
        for name, impl, cfg in variants:
            old = {k: getattr(ind, k) for k in cfg}
            v = res["variants"][name] = dict(impl=impl, inductor=cfg, ms={})
            try:
                for k, val in cfg.items():
                    setattr(ind, k, val)
                set_int8_impl(impl)
                for B, C in points:
                    t, bd, info = time_point(model, B, C, L, (B, C) == (1, 128))
                    v["ms"][f"{B}x{C}"] = t
                    v.setdefault("compile_s", {})[f"{B}x{C}"] = info.get("compile_s")
                    if bd:
                        v["top_kernels_1x128"] = dict(list(bd["top_kernels_ms"].items())[:6])
                        v["to_copy_ms_1x128"] = sum(ms for n, ms in bd["top_kernels_ms"].items() if "to_copy" in n)
                t1, b1 = v["ms"].get("1x128"), res["bf16"].get("1x128")
                if t1 and b1 and w16:
                    v["w_int8_est_ms"] = t1 - (b1 - w16)
                    v["weight_time_ratio"] = v["w_int8_est_ms"] / w16
                    v["int8_gbps"] = res["int8_weight_bytes"] / (v["w_int8_est_ms"] * 1e-3) / 1e9
                v["speedup_vs_bf16"] = {k: res["bf16"][k] / t for k, t in v["ms"].items()}
            except Exception:
                res["errors"][name] = traceback.format_exc()
                harness.free_cuda()
            finally:
                for k, val in old.items():
                    setattr(ind, k, val)
            print(name, json.dumps({k: v.get(k) for k in ("ms", "w_int8_est_ms", "weight_time_ratio", "to_copy_ms_1x128")},
                                   default=str))
            save()
    res["w_bf16_ms"] = w16
    lines = [f"# int8 probe on {res['gpu']}\n", f"bf16 compiled: " + ", ".join(f"{k} {t:.2f} ms" for k, t in res["bf16"].items())
             + f"; bf16 weight kernels at 1x128: {w16 if w16 is None else round(w16, 3)} ms\n",
             "| variant | " + " | ".join(f"{k} ms" for k in res["points"]) + " | int8 weight est. ms (× bf16) | to_copy ms |",
             "|---|" + "---:|" * (len(points) + 2)]
    for name, v in res["variants"].items():
        f2 = lambda x, nd=2: "n/a" if x is None else f"{x:.{nd}f}"
        lines.append(f"| {name} | " + " | ".join(f2(v["ms"].get(k)) for k in res["points"]) +
                     f" | {f2(v.get('w_int8_est_ms'), 3)} ({f2(v.get('weight_time_ratio'))}×) | {f2(v.get('to_copy_ms_1x128'), 3)} |")
    lines.append(f"\nTriton kernel check: {res.get('triton_kernel_check')}. Ideal: int8 weight time ≈ 0.5× bf16.")
    save()
    with open(os.path.join(args.out_dir, "int8_probe.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def quantize_int8(model):
    """Replace every nn.Linear inside the looped layers (q/k/v/o, gate/up/down) with Int8WeightOnlyLinear.
    Embedding, LM head, norms and the exit gate stay bf16. Returns the number of linears replaced."""
    n = 0
    for layer in model.model.layers:
        for mod in list(layer.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mod, name, Int8WeightOnlyLinear(child))
                    n += 1
    harness.free_cuda()
    return n


INT4_GROUP = 128


def quantize_int4_fake(model):
    """Week 6 accuracy only (no int4 kernel exists here): round every looped-layer linear's weight to symmetric
    int4, one scale per 128 inputs per output channel, and store it back in bf16. Same layers as quantize_int8.
    Returns the number of linears rounded."""
    n = 0
    for layer in model.model.layers:
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight.data.float()
                g = w.reshape(w.shape[0], -1, INT4_GROUP)
                scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
                mod.weight.data = ((g / scale).round().clamp(-8, 7) * scale).reshape(w.shape).to(mod.weight.dtype)
                n += 1
    return n


def int8_quantize_and_check(model, args, B=16, C=64):
    """Quantize the model in place, and report how far the int8 step moves from the bf16 step (same cache and
    input; eager, so no compile effects): hidden state, exit gate, top-1 next token via the bf16 LM head."""
    L = model.model.config.num_hidden_layers
    bf16_bytes = loopprof.byte_model(model, harness.DTYPES[args.dtype]).weight_bytes

    def run():
        step, cache, _ = make_step(model, B, C, L, args)
        h, g = (x.float().clone() for x in step())
        top = model.lm_head(h.to(model.lm_head.weight.dtype)).argmax(-1)
        del step, cache
        return h, g, top

    with torch.inference_mode():
        h0, g0, t0 = run()
        n = quantize_int8(model)
        h1, g1, t1 = run()
    return dict(B=B, C=C, linears_quantized=n, bf16_weight_bytes=bf16_bytes,
                rel_err_hidden=float((h1 - h0).norm() / h0.norm()), max_abs_err_gate=float((g1 - g0).abs().max()),
                top1_agreement=float((t1 == t0).float().mean()),
                note="int8 vs bf16, eager. For scale: eager bf16 vs fp32 is ~2.4 % on the hidden state.")


def check_compiled_matches_eager(model, args, mode, B=2, C=64):
    """Same static cache, same input: the compiled step must compute what the eager static step computes."""
    m = model.model
    step, cache, _ = make_step(model, B, C, m.config.num_hidden_layers, args)
    h_ref, gate_ref = (x.float().clone() for x in step())
    del step, cache
    step, cache, info = make_step(model, B, C, m.config.num_hidden_layers, args, mode)   # same seed: same cache
    h, gate = (x.float() for x in step())
    rel = float((h - h_ref).norm() / h_ref.norm())
    gate_err = float((gate - gate_ref).abs().max())
    del step, cache
    harness.free_cuda()
    return dict(B=B, C=C, mode=mode, rel_err_hidden=rel, max_abs_err_gate=gate_err,
                ok=bool(rel < 2e-2 and gate_err < 5e-2), **info)


def compile_numerics(args, B=16, C=64):
    """Is compiled-bf16 vs eager-bf16 disagreement just bf16 rounding? Measure both against an fp32 step on
    the same (bf16-rounded) cache and the same token ids. Rounding: compiled is about as close to fp32 as
    eager is. Bug: compiled is clearly further away. Also top-1 next-token agreement via the LM head."""
    res = dict(B=B, C=C, compile_mode=args.compile_mode, runs={})
    with torch.inference_mode():
        m32 = harness.load_model("fp32")
        L = m32.model.config.num_hidden_layers
        step, cache, _ = make_step(m32, B, C, L, SimpleNamespace(dtype="fp32", seed=args.seed))
        for t in cache.k + cache.v:                 # same cache values as the bf16 runs
            t.copy_(t.to(torch.bfloat16).float())
        h32, g32 = (x.float().clone() for x in step())
        top32 = m32.lm_head(h32).argmax(-1)
        del step, cache, m32
        harness.free_cuda()
        m16 = harness.load_model("bf16")
        for name, mode in (("eager_bf16", None), ("compiled_bf16", args.compile_mode)):
            step, cache, info = make_step(m16, B, C, L, SimpleNamespace(dtype="bf16", seed=args.seed), mode)
            h, g = (x.float().clone() for x in step())
            top = m16.lm_head(h.to(torch.bfloat16)).argmax(-1)
            res["runs"][name] = dict(rel_err_hidden_vs_fp32=float((h - h32).norm() / h32.norm()),
                                     max_abs_err_gate_vs_fp32=float((g - g32).abs().max()),
                                     top1_agreement_vs_fp32=float((top == top32).float().mean()), **info)
            res["runs"][name]["_h"], res["runs"][name]["_g"] = h, g
            del step, cache
            harness.free_cuda()
    e, c = res["runs"]["eager_bf16"], res["runs"]["compiled_bf16"]
    res["compiled_vs_eager"] = dict(rel_err_hidden=float((c["_h"] - e["_h"]).norm() / e["_h"].norm()),
                                    max_abs_err_gate=float((c["_g"] - e["_g"]).abs().max()))
    for r in res["runs"].values():
        del r["_h"], r["_g"]
    ratio = c["rel_err_hidden_vs_fp32"] / e["rel_err_hidden_vs_fp32"]
    res["compiled_over_eager_error_ratio"] = ratio
    res["verdict"] = ("rounding: compiled bf16 is as close to fp32 as eager bf16 (ratio <= 1.5)" if ratio <= 1.5
                      else "SUSPECT: compiled bf16 is further from fp32 than eager bf16 (ratio > 1.5)")
    with open(os.path.join(args.out_dir, "compile_numerics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return res


def check_static_matches_hf(model, dtype, seed):
    """Sanity check: with the SAME cache contents, the static step must equal the HF step."""
    m = model.model
    L = m.config.num_hidden_layers
    B, C = 2, 64
    state = harness.prefill(model, B, C, seed)
    hf_step = harness.build_ut_step(model, state, SimpleNamespace(ut_step=0))
    state.reset()
    h_hf, gate_hf = hf_step()
    state.reset()
    step, cache = build_static_step(model, B, C, L, dtype, seed)
    for l in range(L):                      # copy loop-0 HF cache into the static buffers
        cache.k[l][:, :, :C].copy_(state.cache.key_cache[l][:, :, :C])
        cache.v[l][:, :, :C].copy_(state.cache.value_cache[l][:, :, :C])
    emb = m.embed_tokens(state.next_ids)
    # feed the same decode token as the HF step (build_static_step's own input is random)
    position_ids = torch.full((B, 1), C, device="cuda", dtype=torch.long)
    pe = m.rotary_emb(emb, position_ids)
    h = emb
    for layer in m.layers[:L]:
        h = layer(h, attention_mask=None, position_ids=position_ids, past_key_value=cache, use_cache=True,
                  cache_position=None, position_embeddings=pe, current_ut=0)
    h = m.norm(h)
    gate = m.early_exit_gate(h)
    rel = float((h.float() - h_hf.float()).norm() / h_hf.float().norm())
    gate_err = float((gate.float() - gate_hf.float()).abs().max())
    del state, cache
    harness.free_cuda()
    return dict(B=B, C=C, rel_err_hidden=rel, max_abs_err_gate=gate_err,
                ok=bool(rel < 2e-2 and gate_err < 5e-2))


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.compile_check:
        compile_numerics(args)
        return
    if args.profile_configs:
        step_profile(args)
        return
    if args.noise_check:
        noise_check(args)
        return
    if args.kernel_bench:
        kernel_bench(args)
        return
    if args.int8_probe:
        int8_probe(args)
        return
    if args.int8:
        set_int8_impl(args.int8_impl)
    t_start = time.time()
    if args.compile:
        import torch._inductor.config as ind
        ind.triton.unique_kernel_names = True      # readable kernel names in the breakdown
        if args.coord_descent:
            ind.coordinate_descent_tuning = True
    model = harness.load_model(args.dtype)
    int8_check = None
    if args.int8:
        int8_check = int8_quantize_and_check(model, args)
        print("int8 vs bf16 (eager):", int8_check)
    dtype = harness.DTYPES[args.dtype]
    bm = loopprof.byte_model(model, dtype)
    L = model.model.config.num_hidden_layers
    gpu = torch.cuda.get_device_name()
    peak_meas = loopprof.measured_peak_gbps()
    peak_read = measured_read_gbps()
    peak_spec = loopprof.spec_gbps(gpu)
    print(f"GPU name: {gpu!r} -> spec {peak_spec} GB/s"
          + ("" if peak_spec else "  (NO ENTRY in loopprof.SPEC_GBPS: spec-based numbers will be n/a)"))
    out = dict(tool="delta_split v0", gpu=gpu, args={k: v for k, v in vars(args).items()},
               torch=torch.__version__, weight_bytes_looped_block=bm.weight_bytes,
               kv_bytes_per_token_per_loop=bm.kv_per_token_per_loop,
               spec_peak_gbps=peak_spec, measured_peak_gbps=peak_meas, measured_read_gbps=peak_read, errors={})
    if int8_check:
        out["int8_vs_bf16_check"] = int8_check
        out["int8_impl"] = args.int8_impl
    if args.compile:
        import torch._inductor.config as ind
        out["inductor_config"] = {k: getattr(ind, k, None) for k in
                                  ("use_mixed_mm", "mixed_mm_choice", "force_mixed_mm", "coordinate_descent_tuning")}
    hm = harness.run_metadata(SimpleNamespace(dtype=args.dtype, targets=[], ut_step=0, seed=args.seed,
                                              warmup=args.warmup, reps=args.reps), model)
    out["meta"] = {k: hm[k] for k in ("nvidia_smi", "torch", "cuda", "transformers", "model_revision",
                                      "attn_implementation") if k in hm}
    w_ms_spec = bm.weight_bytes / (peak_spec * 1e9) * 1e3 if peak_spec else None
    w_ms_meas = bm.weight_bytes / (peak_meas * 1e9) * 1e3
    w_ms_read = bm.weight_bytes / (peak_read * 1e9) * 1e3
    out["predicted_weight_fetch_ms"] = dict(spec=w_ms_spec, measured_peak=w_ms_meas, read_peak=w_ms_read)
    print(f"GPU {gpu}: spec {peak_spec} GB/s, measured copy peak {peak_meas:.0f} GB/s, read peak {peak_read:.0f} GB/s; "
          f"looped-block weights {bm.weight_bytes / 1e9:.3f} GB -> predicted weight fetch "
          f"{'n/a' if w_ms_spec is None else f'{w_ms_spec:.2f}'} ms (spec), {w_ms_meas:.2f} ms (copy peak), "
          f"{w_ms_read:.2f} ms (read peak)")

    def save():
        with open(os.path.join(args.out_dir, "delta_split.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)

    # 0. static step must compute the same thing as the HF step
    try:
        with torch.inference_mode():
            out["static_vs_hf_check"] = check_static_matches_hf(model, args.dtype, args.seed)
        print("static vs HF step:", out["static_vs_hf_check"])
    except Exception:
        out["errors"]["static_vs_hf_check"] = traceback.format_exc()
        print(out["errors"]["static_vs_hf_check"])
    save()

    if args.grid:
        run_grid(args, model, bm, L, out, save)
        return

    # 1. launch microbenchmarks
    try:
        out["launch"] = launch_microbench()
        print("launch:", {k: round(v, 3) for k, v in out["launch"].items() if isinstance(v, float)})
    except Exception:
        out["errors"]["launch"] = traceback.format_exc()
        print(out["errors"]["launch"])
    save()

    rows = []
    csv_path = os.path.join(args.out_dir, "delta_split_steps.csv")

    def record(**r):
        rows.append(r)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}))
            w.writeheader()
            w.writerows(rows)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    # 2. three implementations at each (B, C)
    with torch.inference_mode():
        for B, C in args.configs:
            tag = f"{B}x{C}"
            # eager_hf (LoopProf's path)
            try:
                state = harness.prefill(model, B, C, args.seed)
                fn = harness.build_ut_step(model, state, SimpleNamespace(ut_step=0))

                def hf_call():
                    state.reset()
                    fn()
                t = gpu_times(hf_call, args.warmup, args.reps)
                kc = safe_count(hf_call) if (B, C) == args.configs[0] else {}
                record(impl="eager_hf", B=B, C=C, layers=L, **stats(t), **kc)
                del state, fn
                harness.free_cuda()
            except Exception:
                out["errors"][f"eager_hf {tag}"] = traceback.format_exc()
                print(out["errors"][f"eager_hf {tag}"])
                harness.free_cuda()
            # eager_static and graph_static
            try:
                step, cache = build_static_step(model, B, C, L, args.dtype, args.seed)
                t = gpu_times(step, args.warmup, args.reps)
                kc = safe_count(step) if (B, C) == args.configs[0] else {}
                record(impl="eager_static", B=B, C=C, layers=L, **stats(t), **kc)
                out.setdefault("kernel_breakdown", {})[f"eager_static {tag}"] = safe_breakdown(step)
                try:
                    g, _ = capture_graph(step)
                    t = gpu_times(g.replay, args.warmup, args.reps)
                    record(impl="graph_static", B=B, C=C, layers=L, **stats(t))
                    del g
                except Exception:
                    out["errors"][f"graph_static {tag}"] = traceback.format_exc()
                    print(out["errors"][f"graph_static {tag}"])
                del step, cache
                harness.free_cuda()
            except Exception:
                out["errors"][f"static {tag}"] = traceback.format_exc()
                print(out["errors"][f"static {tag}"])
                harness.free_cuda()
            out["steps"] = rows
            save()

        # 3. layer-count sweep at B=1, C=128 (KV traffic ~0.1 % of weight bytes)
        sweep = []
        for nl in args.layers:
            for impl in ("eager_static", "graph_static"):
                try:
                    step, cache = build_static_step(model, 1, 128, nl, args.dtype, args.seed)
                    if impl == "graph_static":
                        g, _ = capture_graph(step)
                        t = gpu_times(g.replay, args.warmup, args.reps)
                        del g
                    else:
                        t = gpu_times(step, args.warmup, args.reps)
                    r = dict(impl=impl, B=1, C=128, layers=nl,
                             weight_bytes=bm.weight_bytes * nl / L, **stats(t))
                    sweep.append(r)
                    record(**r)
                    del step, cache
                    harness.free_cuda()
                except Exception:
                    out["errors"][f"layers {impl} {nl}"] = traceback.format_exc()
                    print(out["errors"][f"layers {impl} {nl}"])
                    harness.free_cuda()
        out["layer_sweep"] = sweep
        save()

    out["analysis"] = analyze(out, bm, L)
    out["wall_s"] = round(time.time() - t_start, 1)
    save()
    md = report(out)
    with open(os.path.join(args.out_dir, "delta_split.md"), "w") as f:
        f.write(md)
    print(md)


# ---------------------------------------------------------------- v1: CUDA-graph B x C grid

def fit_models(rows):
    """OLS fits of t = δ+γB+αC (M1), δ+γB+α'BC (M2), δ+γB+αC+α'BC (M3) with standard errors."""
    B = np.array([r["B"] for r in rows], float)
    C = np.array([r["C"] for r in rows], float)
    y = np.array([r["median_ms"] for r in rows], float)
    one = np.ones_like(B)
    designs = {"M1": (["delta", "gamma", "alpha"], [one, B, C]),
               "M2": (["delta", "gamma", "alpha_prime"], [one, B, B * C]),
               "M3": (["delta", "gamma", "alpha", "alpha_prime"], [one, B, C, B * C])}
    out = {}
    for name, (names, cols) in designs.items():
        A = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        n, k = A.shape
        s2 = float(r @ r) / max(n - k, 1)
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
        tss = float(((y - y.mean()) ** 2).sum())
        aic = n * math.log(max(float(r @ r), 1e-12) / n) + 2 * k
        out[name] = dict(params={p: float(b) for p, b in zip(names, beta)},
                         se={p: float(e) for p, e in zip(names, se)},
                         r2=1 - float(r @ r) / tss, aic=aic, max_abs_resid_ms=float(np.abs(r).max()), n=n)
    return out


def run_grid(args, model, bm, L, out, save):
    out["tool"] = "delta_split v1 (grid)"
    mode = args.compile_mode if args.compile else None
    eager_name, graph_name = (("int8_compiled", "int8_graph") if args.int8 else ("compiled", "compiled_graph")) if mode \
        else ("eager_static", "graph_static")
    out["grid_impl"], out["compile_mode"] = graph_name, mode
    rows, csv_path = [], os.path.join(args.out_dir, "graph_grid.csv")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    cap_gb = GRID_MEM_FRAC * total_mem / 1e9
    mem_gb = min(args.grid_mem_gb, cap_gb) if args.grid_mem_gb else cap_gb
    print(f"grid: static KV limit {mem_gb:.1f} GB (GPU memory {total_mem / 1e9:.1f} GB)")
    out["grid_kv_limit_gb"] = mem_gb
    with torch.inference_mode():
        if mode:
            try:
                out["compiled_vs_eager_check"] = check_compiled_matches_eager(model, args, mode)
                print("compiled vs eager static step:", out["compiled_vs_eager_check"])
            except Exception:
                out["errors"]["compiled_vs_eager_check"] = traceback.format_exc()
                print(out["errors"]["compiled_vs_eager_check"])
                harness.free_cuda()
            save()
        # where GPU time goes: weights-dominated (1x128) and attention across context at B=16
        for (B, C) in [(1, 128), (16, 512), (16, 2048), (16, 8192)]:
            try:
                step, cache, _ = make_step(model, B, C, L, args, mode)
                out.setdefault("kernel_breakdown", {})[f"{eager_name} {B}x{C}"] = safe_breakdown(step)
                g, _ = capture_graph(step)
                out["kernel_breakdown"][f"{graph_name} {B}x{C}"] = safe_breakdown(g.replay)
                del g, step, cache
                harness.free_cuda()
            except Exception:
                out["errors"][f"breakdown {B}x{C}"] = traceback.format_exc()
                harness.free_cuda()
            print(f"kernel breakdown {B}x{C}:", json.dumps(out.get("kernel_breakdown", {}).get(f"{graph_name} {B}x{C}"),
                                                           default=str)[:600])
        save()
        try:
            out["lm_head"] = lm_head_timing(model, harness.DTYPES[args.dtype], args.warmup, args.reps)
            print("lm_head:", json.dumps(out["lm_head"], default=str))
        except Exception:
            out["errors"]["lm_head"] = traceback.format_exc()
            print(out["errors"]["lm_head"])
            harness.free_cuda()
        save()
        for C in args.grid_ctx:
            for B in args.grid_batch:
                kv_gb = B * (C + 1) * bm.kv_per_token_per_loop / 1e9
                if kv_gb > mem_gb:
                    print(f"B={B} C={C} skipped (static KV {kv_gb:.1f} GB)")
                    continue
                try:
                    step, cache, info = make_step(model, B, C, L, args, mode)
                    g, _ = capture_graph(step)
                    t = gpu_times(g.replay, args.warmup, args.reps)
                    r = dict(impl=graph_name, B=B, C=C, layers=L, **stats(t), **info)
                    by = bm.weight_bytes + B * (C + 1) * bm.kv_per_token_per_loop
                    r["achieved_gbps"] = by / (r["median_ms"] * 1e-3) / 1e9
                    r["util_measured"] = r["achieved_gbps"] / out["measured_peak_gbps"]
                    r["util_read"] = r["achieved_gbps"] / out["measured_read_gbps"]
                    rows.append(r)
                    print(f"B={B:<3} C={C:<5} graph {r['median_ms']:7.3f} ms  {r['achieved_gbps']:5.0f} GB/s "
                          f"({100 * r['util_read']:.0f}% of read peak)")
                    del g, step, cache
                except Exception:
                    out["errors"][f"grid {B}x{C}"] = traceback.format_exc().strip().splitlines()[-1]
                    print(f"B={B} C={C} error: {out['errors'][f'grid {B}x{C}']}")
                harness.free_cuda()
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["B"])
                    w.writeheader()
                    w.writerows(rows)
        out["grid"] = rows
        save()
        if args.int8:
            int8_references(L, args, mode, rows, out, save)
            compile_variants(model, L, args, out, save)
        elif mode:
            compile_references(model, L, args, mode, rows, out, save)
    analyze_grid(out, rows)
    save()
    md = report_grid(out)
    with open(os.path.join(args.out_dir, "delta_split_grid.md"), "w") as f:
        f.write(md)
    print(md)


def compile_references(model, L, args, mode, rows, out, save, points=((1, 128), (16, 2048), (16, 8192))):
    """Same container as the compiled grid: the uncompiled CUDA-graph step (δ varies ~±0.25 ms between
    containers, so cross-run comparisons are not enough), its kernel breakdown at 1x128, and a
    max-autotune variant at two points."""
    ref = []
    for B, C in points:
        try:
            step, cache, _ = make_step(model, B, C, L, args)
            g, _ = capture_graph(step)
            if (B, C) == (1, 128):
                out["kernel_breakdown"]["graph_static 1x128"] = safe_breakdown(g.replay)
            base = stats(gpu_times(g.replay, args.warmup, args.reps))["median_ms"]
            comp = next((r["median_ms"] for r in rows if (r["B"], r["C"]) == (B, C)), None)
            ref.append(dict(B=B, C=C, graph_static_ms=base, compiled_graph_ms=comp,
                            speedup=base / comp if comp else None, saved_ms=base - comp if comp else None))
            print("reference:", ref[-1])
            del g, step, cache
        except Exception:
            out["errors"][f"reference {B}x{C}"] = traceback.format_exc().strip().splitlines()[-1]
        harness.free_cuda()
    out["uncompiled_reference"] = ref
    save()
    compile_variants(model, L, args, out, save, points[:2])


def int8_references(L, args, mode, rows, out, save, points=((1, 128), (16, 2048), (16, 8192))):
    """Same container as the int8 grid: the bf16 compiled CUDA-graph step at a few points (a fresh bf16 model),
    and its 1x128 breakdown. Weight time of the int8 step is estimated from step times, assuming the
    non-weight work is unchanged: w_int8 = t_int8 - (t_bf16 - w_bf16). (Fused int8 matmul kernels are Triton
    kernels, which the name-based breakdown cannot reliably tell apart from the other fused kernels.)"""
    ref = []
    try:
        m16 = harness.load_model(args.dtype)
    except Exception:
        out["errors"]["int8 reference model"] = traceback.format_exc().strip().splitlines()[-1]
        return
    for B, C in points:
        try:
            step, cache, _ = make_step(m16, B, C, L, args, mode)
            g, _ = capture_graph(step)
            if (B, C) == (1, 128):
                out["kernel_breakdown"]["compiled_graph 1x128"] = safe_breakdown(g.replay)
            base = stats(gpu_times(g.replay, args.warmup, args.reps))["median_ms"]
            q = next((r["median_ms"] for r in rows if (r["B"], r["C"]) == (B, C)), None)
            ref.append(dict(B=B, C=C, bf16_ms=base, int8_ms=q, speedup=base / q if q else None,
                            saved_ms=base - q if q else None))
            print("int8 reference:", ref[-1])
            del g, step, cache
        except Exception:
            out["errors"][f"int8 reference {B}x{C}"] = traceback.format_exc().strip().splitlines()[-1]
        harness.free_cuda()
    del m16
    harness.free_cuda()
    out["int8_reference"] = ref
    r1 = next((r for r in ref if (r["B"], r["C"]) == (1, 128) and r["int8_ms"]), None)
    w16 = ((out["kernel_breakdown"].get("compiled_graph 1x128") or {}).get("gemm_ms"))
    if r1 and w16:
        w8 = r1["int8_ms"] - (r1["bf16_ms"] - w16)
        b8, b16 = out["weight_bytes_looped_block"], out["int8_vs_bf16_check"]["bf16_weight_bytes"]
        out["int8_weight_estimate"] = dict(
            w_bf16_ms=w16, t_bf16_ms=r1["bf16_ms"], t_int8_ms=r1["int8_ms"], w_int8_est_ms=w8,
            int8_weight_bytes=b8, bf16_weight_bytes=b16, int8_gbps=b8 / (w8 * 1e-3) / 1e9 if w8 > 0 else None,
            bf16_gbps=b16 / (w16 * 1e-3) / 1e9, weight_time_ratio=w8 / w16, byte_ratio=b8 / b16)
    save()


def compile_variants(model, L, args, out, save, points=((1, 128), (16, 2048))):
    """max-autotune (Triton GEMM templates, slow compile) at a couple of points, timing only."""
    variants = []
    for B, C in points:
        try:
            step, cache, info = make_step(model, B, C, L, args, "max-autotune-no-cudagraphs")
            g, _ = capture_graph(step)
            variants.append(dict(mode="max-autotune-no-cudagraphs", B=B, C=C,
                                 graph_ms=stats(gpu_times(g.replay, args.warmup, args.reps))["median_ms"], **info))
            print("variant:", variants[-1])
            del g, step, cache
        except Exception:
            out["errors"][f"max-autotune {B}x{C}"] = traceback.format_exc().strip().splitlines()[-1]
        harness.free_cuda()
    out["compile_variants"] = variants
    save()


def lm_head_timing(model, dtype, warmup, reps, batches=(1, 16, 64)):
    """The LM head runs once per generated token (not per loop): time it alone, eager and CUDA graph."""
    head, cfg = model.lm_head, model.config
    params = list(head.parameters())
    nbytes = sum(p.numel() * p.element_size() for p in params)
    emb = model.model.embed_tokens.weight
    res = dict(params=sum(p.numel() for p in params), bytes=nbytes, vocab_size=cfg.vocab_size,
               hidden_size=cfg.hidden_size, tie_word_embeddings=getattr(cfg, "tie_word_embeddings", None),
               shares_embedding_tensor=head.weight.data_ptr() == emb.data_ptr(),
               embedding_params=emb.numel(), rows=[])
    for B in batches:
        x = torch.randn(B, 1, cfg.hidden_size, device="cuda", dtype=dtype)
        fn = lambda: head(x)
        r = dict(B=B, eager_ms=stats(gpu_times(fn, warmup, reps))["median_ms"])
        g, _ = capture_graph(fn)
        r["graph_ms"] = stats(gpu_times(g.replay, warmup, reps))["median_ms"]
        r["graph_gbps"] = nbytes / (r["graph_ms"] * 1e-3) / 1e9
        res["rows"].append(r)
        del g, x
    harness.free_cuda()
    return res


def attention_per_token(out):
    """attention kernel ms / (B·C) per breakdown row: should match the fitted α'. Plus achieved KV read BW."""
    kvb, rows = out["kv_bytes_per_token_per_loop"], []
    for k, v in (out.get("kernel_breakdown") or {}).items():
        if not v or not v.get("attention_ms"):
            continue
        impl, bc = k.split()
        B, C = (int(x) for x in bc.split("x"))
        ms = v["attention_ms"]
        rows.append(dict(impl=impl, B=B, C=C, attention_ms=ms, us_per_cached_token=ms * 1e3 / (B * C),
                         read_gbps=B * C * kvb / (ms * 1e-3) / 1e9))
    return rows


def calibration_inputs(out, rows):
    """One flat dict of the numbers the analytical model is calibrated against (B=1, C=128 for per-step terms)."""
    impl = out.get("grid_impl", "graph_static")
    g1 = (out.get("kernel_breakdown") or {}).get(f"{impl} 1x128") or {}
    m2 = (out.get("grid_fit") or {}).get("M2") or {}
    p, se = m2.get("params", {}), m2.get("se", {})
    read = out.get("measured_read_gbps")
    pct = lambda bw: 100 * bw / read if bw and read else None
    est = out.get("int8_weight_estimate") or {}
    w_ms = est.get("w_int8_est_ms") or g1.get("gemm_ms")
    ap = p.get("alpha_prime")                          # ms per cached token
    t1 = next((r["median_ms"] for r in rows if (r["B"], r["C"]) == (1, 128)), None)
    w_bw = out["weight_bytes_looped_block"] / (w_ms * 1e-3) / 1e9 if w_ms else None
    kv_bw = out["kv_bytes_per_token_per_loop"] / (ap * 1e-3) / 1e9 if ap and ap > 0 else None
    lm = out.get("lm_head") or {}
    lm1 = next((r for r in lm.get("rows", []) if r["B"] == 1), {})
    return dict(
        gpu=out["gpu"], spec_gbps=out.get("spec_peak_gbps"), copy_gbps=out.get("measured_peak_gbps"), read_gbps=read,
        weight_ms=w_ms, weight_gbps=w_bw, weight_pct_read=pct(w_bw),
        weight_ms_source="int8 estimate: t_int8 - (t_bf16 - w_bf16), B=1, C=128" if est.get("w_int8_est_ms")
        else "GEMV/GEMM kernels, graph, B=1, C=128",
        delta_ms=p.get("delta"), delta_se=se.get("delta"), gamma_ms=p.get("gamma"), gamma_se=se.get("gamma"),
        alpha_prime_us=ap * 1e3 if ap is not None else None,
        alpha_prime_se_us=se["alpha_prime"] * 1e3 if "alpha_prime" in se else None,
        kv_gbps=kv_bw, kv_pct_read=pct(kv_bw),
        attention_us_per_token={f"{r['B']}x{r['C']}": r["us_per_cached_token"]
                                for r in attention_per_token(out) if r["impl"] == impl},
        small_kernels_ms=g1.get("copy_ms", 0) + g1.get("other_ms", 0) if g1 else None,
        gap_ms=t1 - g1["busy_ms"] if t1 is not None and "busy_ms" in g1 else None,
        x1_crossover_tokens=w_ms / ap if w_ms and ap and ap > 0 else None,
        lm_head_graph_ms_b1=lm1.get("graph_ms"), lm_head_bytes=lm.get("bytes"))


def analyze_grid(out, rows):
    if len(rows) >= 5:
        out["grid_fit"] = fit_models(rows)
    out["analysis"] = {"onchip_weights_upper_bound": onchip_bound(out, rows, out.get("grid_impl", "graph_static")),
                       "attention_per_token": attention_per_token(out),
                       "calibration_inputs": calibration_inputs(out, rows)}


def calibration_md(ci):
    f = lambda v, nd=2: "n/a" if v is None else f"{v:,.{nd}f}"
    pm = lambda v, e, nd: "n/a" if v is None else f"{v:.{nd}f} ± {f(e, nd)}"
    att = ", ".join(f"{k}: {v:.3f}" for k, v in ci["attention_us_per_token"].items()) or "n/a"
    rows = [("GPU", ci["gpu"]), ("spec bandwidth", f"{f(ci['spec_gbps'], 0)} GB/s"),
            ("copy bandwidth (read + write)", f"{f(ci['copy_gbps'], 0)} GB/s"),
            ("read bandwidth (torch.sum, 2 GiB)", f"{f(ci['read_gbps'], 0)} GB/s"),
            (f"weight time ({ci.get('weight_ms_source', 'GEMV/GEMM kernels, graph, B=1, C=128')})", f"{f(ci['weight_ms'], 3)} ms"),
            ("effective weight bandwidth", f"{f(ci['weight_gbps'], 0)} GB/s ({f(ci['weight_pct_read'], 0)}% of read peak)"),
            ("δ (M2)", f"{pm(ci['delta_ms'], ci['delta_se'], 3)} ms"),
            ("γ (M2)", f"{pm(ci['gamma_ms'], ci['gamma_se'], 4)} ms per sequence"),
            ("α′ (M2)", f"{pm(ci['alpha_prime_us'], ci['alpha_prime_se_us'], 4)} µs per cached token"),
            ("α′-implied KV bandwidth", f"{f(ci['kv_gbps'], 0)} GB/s ({f(ci['kv_pct_read'], 0)}% of read peak)"),
            ("attention kernels, µs per cached token (graph, BxC)", att),
            ("small kernels (copy + other), graph, B=1, C=128", f"{f(ci['small_kernels_ms'], 3)} ms"),
            ("gaps (graph step − kernel busy), B=1, C=128", f"{f(ci['gap_ms'], 3)} ms"),
            ("X1 crossover (weight time ÷ α′)", f"{f(ci['x1_crossover_tokens'], 0)} cached tokens"),
            ("LM head, graph, B=1 (once per token, not per loop)",
             f"{f(ci['lm_head_graph_ms_b1'], 3)} ms, {f((ci['lm_head_bytes'] or 0) / 1e6 or None, 1)} MB")]
    return ["## Calibration inputs\n", "| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows] + [""]


def compile_md(out):
    """Compiled vs uncompiled CUDA-graph step, same container, plus the correctness check."""
    if not out.get("compile_mode"):
        return []
    if out.get("int8_vs_bf16_check") is not None:
        return int8_md(out)
    f = lambda v, nd=2: "n/a" if v is None else f"{v:.{nd}f}"
    L = ["## torch.compile vs uncompiled (same container)\n"]
    chk = out.get("compiled_vs_eager_check")
    if chk:
        L.append(f"Compiled vs eager static step (same cache, B={chk['B']}, C={chk['C']}): relative error "
                 f"{chk['rel_err_hidden']:.2e}, gate error {chk['max_abs_err_gate']:.2e} → "
                 f"{'OK' if chk['ok'] else '**MISMATCH: compiled numbers below are not trustworthy**'}; "
                 f"compile {chk.get('compile_s', 'n/a')} s, graph breaks {chk.get('graph_breaks', 'n/a')}.\n")
    else:
        L.append("**Compiled-vs-eager check did not run (see Errors): compiled numbers are unverified.**\n")
    kb = out.get("kernel_breakdown") or {}
    a, b = kb.get("graph_static 1x128"), kb.get("compiled_graph 1x128")
    if a and b:
        L += ["| B=1, C=128, kernel time | kernels | gemm ms | attention ms | copy + other ms | busy ms |",
              "|---|---:|---:|---:|---:|---:|"]
        for name, v in (("uncompiled graph", a), ("compiled graph", b)):
            L.append(f"| {name} | {v.get('kernels', 0):.0f} | {f(v.get('gemm_ms'))} | {f(v.get('attention_ms'))} | "
                     f"{f(v.get('copy_ms', 0) + v.get('other_ms', 0))} | {f(v.get('busy_ms'))} |")
        L.append("")
    ref = out.get("uncompiled_reference") or []
    if ref:
        L += ["| B | C | uncompiled graph ms | compiled graph ms | saved ms | speedup |", "|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r['B']} | {r['C']} | {f(r['graph_static_ms'])} | {f(r['compiled_graph_ms'])} | {f(r['saved_ms'])} | "
              f"{f(r['speedup'])}× |" for r in ref]
        L.append("")
    for v in out.get("compile_variants") or []:
        L.append(f"- {v['mode']} at B={v['B']}, C={v['C']}: {f(v['graph_ms'])} ms (compile {v.get('compile_s', 'n/a')} s)")
    return L + [""]


def int8_md(out):
    """int8 weight-only (compiled) vs bf16 compiled, same container."""
    f = lambda v, nd=2: "n/a" if v is None else f"{v:.{nd}f}"
    c = out["int8_vs_bf16_check"]
    L = ["## int8 weight-only vs bf16, both torch.compile + CUDA graph (same container)\n",
         f"Looped-block weights: {c['bf16_weight_bytes'] / 1e9:.3f} GB bf16 → {out['weight_bytes_looped_block'] / 1e9:.3f} GB "
         f"int8 + scales ({c['linears_quantized']} linears). Embedding, LM head, norms, exit gate stay bf16.\n",
         f"Accuracy, int8 vs bf16 step (eager, B={c['B']}, C={c['C']}): hidden state {100 * c['rel_err_hidden']:.1f} %, "
         f"exit gate {c['max_abs_err_gate']:.3f}, top-1 next token agrees {100 * c['top1_agreement']:.0f} %. "
         f"({c['note']} GSM8K accuracy is the Week 6 test.)\n"]
    ref = out.get("int8_reference") or []
    if ref:
        L += ["| B | C | bf16 compiled ms | int8 compiled ms | saved ms | speedup |", "|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r['B']} | {r['C']} | {f(r['bf16_ms'])} | {f(r['int8_ms'])} | {f(r['saved_ms'])} | {f(r['speedup'])}× |"
              for r in ref]
        L.append("")
    e = out.get("int8_weight_estimate")
    if e:
        L.append(f"Weight time at B=1, C=128: bf16 {f(e['w_bf16_ms'], 3)} ms ({f(e['bf16_gbps'], 0)} GB/s) → int8 ≈ "
                 f"{f(e['w_int8_est_ms'], 3)} ms ({f(e['int8_gbps'], 0)} GB/s), i.e. {f(e['weight_time_ratio'])}× the bf16 time "
                 f"for {f(e['byte_ratio'])}× the bytes (estimate: t_int8 − (t_bf16 − w_bf16)).\n")
    for v in out.get("compile_variants") or []:
        L.append(f"- int8, {v['mode']} at B={v['B']}, C={v['C']}: {f(v['graph_ms'])} ms (compile {v.get('compile_s', 'n/a')} s)")
    return L + [""]


def lm_head_md(lm):
    if not lm:
        return []
    L = ["## Once per token (not per loop): LM head\n",
         f"lm_head: {lm['params']:,} params, {lm['bytes'] / 1e6:.1f} MB; vocab {lm['vocab_size']:,}; "
         f"tie_word_embeddings={lm['tie_word_embeddings']} (shares the embedding tensor: {lm['shares_embedding_tensor']}); "
         f"embedding: {lm['embedding_params']:,} params.\n",
         "| B | eager ms | graph ms | graph GB/s |", "|---:|---:|---:|---:|"]
    L += [f"| {r['B']} | {r['eager_ms']:.3f} | {r['graph_ms']:.3f} | {r['graph_gbps']:.0f} |" for r in lm["rows"]]
    return L + [""]


def attention_md(rows, out):
    if not rows:
        return []
    read = out.get("measured_read_gbps")
    L = ["## Attention kernels across context (should match α′)\n",
         "| step | attention ms | µs per cached token | KV read GB/s | % of read peak |", "|---|---:|---:|---:|---:|"]
    L += [f"| {r['impl']} {r['B']}x{r['C']} | {r['attention_ms']:.3f} | {r['us_per_cached_token']:.4f} | "
          f"{r['read_gbps']:.0f} | {100 * r['read_gbps'] / read:.0f}% |" if read else
          f"| {r['impl']} {r['B']}x{r['C']} | {r['attention_ms']:.3f} | {r['us_per_cached_token']:.4f} | "
          f"{r['read_gbps']:.0f} | n/a |" for r in rows]
    return L + [""]


def breakdown_md(out):
    L = []
    kb = {k: v for k, v in (out.get("kernel_breakdown") or {}).items() if v}
    if kb:
        cats = ["gemm", "attention", "copy", "other"]
        L += ["## GPU kernel time per step, by kind (kernel durations only; no host gaps)\n",
              "| step | kernels | " + " | ".join(f"{c} ms" for c in cats) + " | busy ms |",
              "|---|---:|" + "---:|" * (len(cats) + 1)]
        for k, v in kb.items():
            L.append(f"| {k} | {v.get('kernels', 0):.0f} | " + " | ".join(f"{v.get(c + '_ms', 0):.2f}" for c in cats)
                     + f" | {v.get('busy_ms', 0):.2f} |")
        first = next(iter(kb.values()))
        L += ["", "Top kernels (first row): " + "; ".join(f"`{n[:60]}` {t:.2f} ms"
                                                          for n, t in list(first["top_kernels_ms"].items())[:6]), ""]
    return L


def report_grid(out):
    f = lambda v, nd=2: "n/a" if v is None else f"{v:.{nd}f}"
    what = ((("int8 weight-only + " if out.get("int8_vs_bf16_check") is not None else "")
             + f"torch.compile ({out['compile_mode']}) + CUDA-graph") if out.get("compile_mode") else "CUDA-graph")
    L = [f"# Week 5 v1: {what} static-cache step over B x C on {out['gpu']}\n",
         f"Predicted weight fetch {f(out['predicted_weight_fetch_ms']['spec'])} ms (spec), "
         f"{f(out['predicted_weight_fetch_ms']['measured_peak'])} ms (measured copy peak "
         f"{out['measured_peak_gbps']:.0f} GB/s), "
         f"{f(out['predicted_weight_fetch_ms'].get('read_peak'))} ms (measured read peak "
         f"{f(out.get('measured_read_gbps'), 0)} GB/s).\n"]
    an = out.get("analysis") or {}
    L += compile_md(out)
    if an.get("calibration_inputs"):
        L += calibration_md(an["calibration_inputs"])
    L += lm_head_md(out.get("lm_head"))
    L += breakdown_md(out)
    L += attention_md(an.get("attention_per_token"), out)
    gf = out.get("grid_fit")
    if gf:
        L += ["## Fits\n", "| model | δ ms | γ ms/seq | α ms/token | α' µs per cached token | R² | AIC | max resid ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m, v in gf.items():
            p, se = v["params"], v["se"]
            cell = lambda k, s=1.0, nd=3: (f"{p[k] * s:.{nd}f} ± {se[k] * s:.{nd}f}" if k in p else "")
            L.append(f"| {m} | {cell('delta')} | {cell('gamma', 1, 4)} | {cell('alpha', 1, 6)} | "
                     f"{cell('alpha_prime', 1e3, 4)} | {v['r2']:.4f} | {v['aic']:.1f} | {v['max_abs_resid_ms']:.2f} |")
        m2 = gf["M2"]["params"]
        if m2.get("alpha_prime", 0) > 0:
            bw = out["kv_bytes_per_token_per_loop"] / (m2["alpha_prime"] * 1e-3) / 1e9
            rd = out.get("measured_read_gbps")
            L.append(f"\nM2 KV slope → effective KV bandwidth {bw:.0f} GB/s ("
                     + (f"{100 * bw / rd:.0f}% of measured read peak" if rd else
                        f"{100 * bw / out['measured_peak_gbps']:.0f}% of measured copy peak") + ").\n")
    oc = (out.get("analysis") or {}).get("onchip_weights_upper_bound")
    if oc:
        L += [f"## Upper bound if weights were free (weight term = {oc['weight_term_source']})\n",
              "| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |", "|---:|---:|---:|---:|---:|---:|"]
        for r in oc["rows"]:
            sp = r["speedup_if_weights_free"]
            L.append(f"| {r['B']} | {r['C']} | {f(r['gpu_graph_ms'])} | {f(r['weight_term_ms'])} | "
                     f"{f(r['remaining_ms'])} | {f(sp) + '×' if sp else 'n/a (weight term ≈ whole step)'} |")
        L.append(f"\n{oc['note']}\n")
    if out.get("errors"):
        L.append("## Errors\n")
        for k, v in out["errors"].items():
            L.append(f"- **{k}**: `{str(v).strip().splitlines()[-1]}`")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- analysis

def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    n = len(x)
    if n > 2:
        s2 = float(resid @ resid) / (n - 2)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    else:
        se = np.array([float("nan")] * 2)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return dict(intercept=float(beta[0]), slope=float(beta[1]), se_intercept=float(se[0]),
                se_slope=float(se[1]), r2=r2, n=n)


def analyze(out, bm, L):
    a = {}
    steps = [r for r in out.get("steps", []) if r.get("layers") == L]   # layer-sweep rows excluded
    get = lambda impl, B, C: next((r["median_ms"] for r in steps
                                    if r["impl"] == impl and r["B"] == B and r["C"] == C), None)
    # decomposition at the smallest-B·C config
    B0, C0 = min(((r["B"], r["C"]) for r in steps), key=lambda bc: bc[0] * bc[1], default=(None, None))
    if B0 is not None:
        hf, st, gr = get("eager_hf", B0, C0), get("eager_static", B0, C0), get("graph_static", B0, C0)
        w_meas = out["predicted_weight_fetch_ms"]["measured_peak"]
        d = dict(B=B0, C=C0, eager_hf_ms=hf, eager_static_ms=st, graph_static_ms=gr,
                 predicted_weight_fetch_ms_measured_peak=w_meas,
                 predicted_weight_fetch_ms_spec=out["predicted_weight_fetch_ms"]["spec"])
        if hf is not None and st is not None:
            d["hf_cache_overhead_ms"] = hf - st
        if st is not None and gr is not None:
            d["launch_and_python_overhead_ms"] = st - gr
        if gr is not None:
            d["gpu_time_ms"] = gr
            d["gpu_time_beyond_ideal_weight_stream_ms"] = gr - w_meas
            d["weight_stream_efficiency_in_graph"] = w_meas / gr
        if hf is not None and gr is not None:
            d["software_overhead_total_ms"] = hf - gr
            d["software_overhead_fraction_of_eager_hf"] = (hf - gr) / hf
        a["decomposition_smallest_config"] = d
    # layer sweep: slope -> per-layer time, effective bandwidth
    for impl in ("graph_static", "eager_static"):
        pts = [r for r in out.get("layer_sweep", []) if r["impl"] == impl]
        if len(pts) >= 2:
            f = linfit([r["weight_bytes"] for r in pts], [r["median_ms"] for r in pts])
            f["effective_weight_bw_gbps"] = 1 / (f["slope"] * 1e-3) / 1e9 if f["slope"] > 0 else None
            f["weight_term_ms_full_block"] = f["slope"] * bm.weight_bytes
            f["note"] = "t(L') = intercept + slope * weight_bytes(L'); slope in ms per byte"
            a[f"layer_sweep_fit_{impl}"] = f
    # KV slope per implementation (are HF-cache KV costs inflated?)
    for impl in ("eager_hf", "eager_static", "graph_static"):
        pts = [r for r in steps if r["impl"] == impl]
        if len(pts) >= 3:
            f = linfit([r["B"] * r["C"] for r in pts], [r["median_ms"] for r in pts])
            f["kv_us_per_cached_token"] = f["slope"] * 1e3
            f["effective_kv_bw_gbps"] = (bm.kv_per_token_per_loop / (f["slope"] * 1e-3) / 1e9
                                         if f["slope"] > 0 else None)
            a[f"kv_slope_{impl}"] = f
    # first, explicitly bounded, estimate of what on-chip weights save (best-case GPU baseline)
    # The weight term is the measured GEMM kernel time at the smallest config (kernel durations, no
    # host gaps) when the profile exists, else bytes / measured copy peak. NOT the layer-sweep slope:
    # removing layers removes their norms/rotary/attention kernels too, so that slope over-counts.
    a["onchip_weights_upper_bound"] = onchip_bound(out, steps, "graph_static")
    return a


def onchip_bound(out, steps, impl):
    kb = next((v for k, v in (out.get("kernel_breakdown") or {}).items() if v and "gemm_ms" in v), None)
    wt = kb["gemm_ms"] if kb else out["predicted_weight_fetch_ms"]["measured_peak"]
    src = "GEMM kernel time (profiled)" if kb else "weight bytes / measured copy peak"
    est = (out.get("int8_weight_estimate") or {}).get("w_int8_est_ms")
    if est:
        wt, src = est, "int8 weight time, estimated from step times"
    pts = [r for r in steps if r["impl"] == impl]
    if pts:
        est = []
        for r in pts:
            t_gpu = r["median_ms"]
            rem = t_gpu - wt
            est.append(dict(B=r["B"], C=r["C"], gpu_graph_ms=t_gpu, weight_term_ms=wt,
                            remaining_ms=rem, speedup_if_weights_free=(t_gpu / rem) if rem > 0.05 * t_gpu else None))
        return dict(
            rows=est, weight_term_source=src, note=("Upper bound on the step-time gain from removing ONLY the weight-streaming "
                            "term from the best-case (CUDA-graph, static-cache) GPU step; KV still comes "
                            "from DRAM and SRAM read time is taken as zero. Vaibhav's model replaces this."))


def report(out):
    a = out.get("analysis", {})
    f = lambda v, nd=2: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"
    L = []
    L.append(f"# Week 5: δ split on {out['gpu']}\n")
    p = out["predicted_weight_fetch_ms"]
    L.append(f"Looped-block weights: {out['weight_bytes_looped_block'] / 1e9:.3f} GB. Predicted weight fetch: "
             f"**{f(p['spec'])} ms** at spec bandwidth ({out['spec_peak_gbps']} GB/s), "
             f"**{f(p['measured_peak'])} ms** at the measured copy peak ({out['measured_peak_gbps']:.0f} GB/s), "
             f"**{f(p.get('read_peak'))} ms** at the measured read peak ({f(out.get('measured_read_gbps'), 0)} GB/s).\n")
    chk = out.get("static_vs_hf_check")
    if chk:
        L.append(f"Static-cache step vs HF step (same cache contents): relative error {chk['rel_err_hidden']:.2e}, "
                 f"gate error {chk['max_abs_err_gate']:.2e} → {'OK' if chk['ok'] else '**MISMATCH**'}.\n")
    lc = out.get("launch")
    if lc:
        L.append(f"Launch cost: eager {lc['eager_host_us_per_launch']:.2f} µs of CPU per kernel launch; "
                 f"inside a CUDA graph {lc['graph_us_per_op']:.2f} µs per op.\n")
    d = a.get("decomposition_smallest_config")
    if d:
        L += [f"## Decomposition at B={d['B']}, C={d['C']} (KV traffic negligible)\n",
              "| implementation | step time (ms) |", "|---|---:|",
              f"| eager + HF cache (LoopProf) | {f(d.get('eager_hf_ms'))} |",
              f"| eager + static cache | {f(d.get('eager_static_ms'))} |",
              f"| CUDA graph + static cache (best-case GPU) | {f(d.get('graph_static_ms'))} |", "",
              "| component | ms |", "|---|---:|",
              f"| HF cache handling (eager_hf − eager_static) | {f(d.get('hf_cache_overhead_ms'))} |",
              f"| launch + Python (eager_static − graph) | {f(d.get('launch_and_python_overhead_ms'))} |",
              f"| **software overhead total** | **{f(d.get('software_overhead_total_ms'))}** "
              f"({f(100 * d['software_overhead_fraction_of_eager_hf'], 0) if d.get('software_overhead_fraction_of_eager_hf') is not None else 'n/a'} % of the eager step) |",
              f"| GPU time (graph) | {f(d.get('gpu_time_ms'))} |",
              f"| … of which ideal weight streaming (measured peak) | {f(d.get('predicted_weight_fetch_ms_measured_peak'))} |",
              f"| … GPU time beyond ideal streaming | {f(d.get('gpu_time_beyond_ideal_weight_stream_ms'))} |", ""]
    for impl in ("graph_static", "eager_static"):
        lw = a.get(f"layer_sweep_fit_{impl}")
        if lw:
            L.append(f"**Layer sweep ({impl})**: t = {f(lw['intercept'])} ms + weight term; full-block weight term "
                     f"{f(lw['weight_term_ms_full_block'])} ms → effective weight bandwidth "
                     f"{f(lw['effective_weight_bw_gbps'], 0)} GB/s (R² {f(lw['r2'], 4)}, n={lw['n']}).\n")
    kv = [(i, a.get(f"kv_slope_{i}")) for i in ("eager_hf", "eager_static", "graph_static")]
    if any(v for _, v in kv):
        L += ["## KV cost per cached token (slope of step time vs B·C)\n",
              "| implementation | µs per cached token | effective KV bandwidth (GB/s) | R² |", "|---|---:|---:|---:|"]
        for i, v in kv:
            if v:
                L.append(f"| {i} | {f(v['kv_us_per_cached_token'], 3)} | {f(v['effective_kv_bw_gbps'], 0)} | {f(v['r2'], 4)} |")
        L.append("")
    L += breakdown_md(out)
    oc = a.get("onchip_weights_upper_bound")
    if oc:
        L += [f"## Upper bound: removing only the weight term ({oc['weight_term_source']}) from the best-case GPU step\n",
              "| B | C | GPU (graph) ms | weight term ms | remaining ms | speedup bound |", "|---:|---:|---:|---:|---:|---:|"]
        for r in oc["rows"]:
            L.append(f"| {r['B']} | {r['C']} | {f(r['gpu_graph_ms'])} | {f(r['weight_term_ms'])} | "
                     f"{f(r['remaining_ms'])} | {f(r['speedup_if_weights_free']) + '×' if r['speedup_if_weights_free'] else 'n/a'} |")
        L.append(f"\n{oc['note']}\n")
    if out.get("errors"):
        L.append("## Errors\n")
        for k, v in out["errors"].items():
            L.append(f"- **{k}**: `{v.strip().splitlines()[-1]}`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
