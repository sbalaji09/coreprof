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
    p.add_argument("--grid-mem-gb", type=float, default=30.0, help="skip grid points whose static KV exceeds this")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
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
    ("gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas", "sm80_", "sm89_", "sm90_", "ampere_", "ada_")),
    ("copy", ("copy", "memcpy", "memset", "cat", "index")),
]


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
    t_start = time.time()
    model = harness.load_model(args.dtype)
    dtype = harness.DTYPES[args.dtype]
    bm = loopprof.byte_model(model, dtype)
    L = model.model.config.num_hidden_layers
    gpu = torch.cuda.get_device_name()
    peak_meas = loopprof.measured_peak_gbps()
    peak_spec = loopprof.spec_gbps(gpu)
    out = dict(tool="delta_split v0", gpu=gpu, args={k: v for k, v in vars(args).items()},
               torch=torch.__version__, weight_bytes_looped_block=bm.weight_bytes,
               kv_bytes_per_token_per_loop=bm.kv_per_token_per_loop,
               spec_peak_gbps=peak_spec, measured_peak_gbps=peak_meas, errors={})
    hm = harness.run_metadata(SimpleNamespace(dtype=args.dtype, targets=[], ut_step=0, seed=args.seed,
                                              warmup=args.warmup, reps=args.reps), model)
    out["meta"] = {k: hm[k] for k in ("nvidia_smi", "torch", "cuda", "transformers", "model_revision",
                                      "attn_implementation") if k in hm}
    w_ms_spec = bm.weight_bytes / (peak_spec * 1e9) * 1e3 if peak_spec else None
    w_ms_meas = bm.weight_bytes / (peak_meas * 1e9) * 1e3
    out["predicted_weight_fetch_ms"] = dict(spec=w_ms_spec, measured_peak=w_ms_meas)
    print(f"GPU {gpu}: spec {peak_spec} GB/s, measured copy peak {peak_meas:.0f} GB/s; looped-block weights "
          f"{bm.weight_bytes / 1e9:.3f} GB -> predicted weight fetch {w_ms_spec:.2f} ms (spec), "
          f"{w_ms_meas:.2f} ms (measured peak)")

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
    rows, csv_path = [], os.path.join(args.out_dir, "graph_grid.csv")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    with torch.inference_mode():
        for (B, C) in [(1, 128), (16, 2048)]:          # where does GPU time go, small vs large B·C
            try:
                step, cache = build_static_step(model, B, C, L, args.dtype, args.seed)
                out.setdefault("kernel_breakdown", {})[f"eager_static {B}x{C}"] = safe_breakdown(step)
                g, _ = capture_graph(step)
                out["kernel_breakdown"][f"graph_static {B}x{C}"] = safe_breakdown(g.replay)
                del g, step, cache
                harness.free_cuda()
            except Exception:
                out["errors"][f"breakdown {B}x{C}"] = traceback.format_exc()
                harness.free_cuda()
            print(f"kernel breakdown {B}x{C}:", json.dumps(out["kernel_breakdown"].get(f"eager_static {B}x{C}"),
                                                           default=str)[:600])
        save()
        for C in args.grid_ctx:
            for B in args.grid_batch:
                kv_gb = B * (C + 1) * bm.kv_per_token_per_loop / 1e9
                if kv_gb > args.grid_mem_gb or kv_gb * 1e9 > 0.6 * total_mem:
                    print(f"B={B} C={C} skipped (static KV {kv_gb:.1f} GB)")
                    continue
                try:
                    step, cache = build_static_step(model, B, C, L, args.dtype, args.seed)
                    g, _ = capture_graph(step)
                    t = gpu_times(g.replay, args.warmup, args.reps)
                    r = dict(impl="graph_static", B=B, C=C, layers=L, **stats(t))
                    by = bm.weight_bytes + B * (C + 1) * bm.kv_per_token_per_loop
                    r["achieved_gbps"] = by / (r["median_ms"] * 1e-3) / 1e9
                    r["util_measured"] = r["achieved_gbps"] / out["measured_peak_gbps"]
                    rows.append(r)
                    print(f"B={B:<3} C={C:<5} graph {r['median_ms']:7.3f} ms  {r['achieved_gbps']:5.0f} GB/s "
                          f"({100 * r['util_measured']:.0f}% of copy peak)")
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
    if len(rows) >= 5:
        out["grid_fit"] = fit_models(rows)
    out["analysis"] = {"onchip_weights_upper_bound": onchip_bound(out, rows, "graph_static")}
    save()
    md = report_grid(out)
    with open(os.path.join(args.out_dir, "delta_split_grid.md"), "w") as f:
        f.write(md)
    print(md)


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
    L = [f"# Week 5 v1: CUDA-graph static-cache step over B x C on {out['gpu']}\n",
         f"Predicted weight fetch {f(out['predicted_weight_fetch_ms']['spec'])} ms (spec), "
         f"{f(out['predicted_weight_fetch_ms']['measured_peak'])} ms (measured copy peak "
         f"{out['measured_peak_gbps']:.0f} GB/s).\n"]
    L += breakdown_md(out)
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
            L.append(f"\nM2 KV slope → effective KV bandwidth {bw:.0f} GB/s "
                     f"({100 * bw / out['measured_peak_gbps']:.0f}% of measured copy peak).\n")
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
             f"**{f(p['measured_peak'])} ms** at the measured copy peak ({out['measured_peak_gbps']:.0f} GB/s).\n")
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
