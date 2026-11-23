"""LoopCore analytical model, v1 (Weeks 5-6). v0 by Vaibhav (1a8c036); v1 review fixes below.

Predicts decode time per loop step and per token for a looped LM (Ouro-1.4B) on
  (a) a GPU: fitted on one GPU's measurements, then *predicted* for another GPU from its datasheet clock and
      compute plus one measured bandwidth number (the calibration test in CALIBRATION.md), and
  (b) LoopCore configurations that keep recurrent-core weights in on-chip SRAM.

Terms (the shared definitions in CALIBRATION.md):
  T_step  = combine(T_weights, T_compute, T_kv, T_link) + T_overhead
  T_weights = weight bytes / weight bandwidth + weight kernels x fixed cost per kernel   (GPU)
  T_token = mean_depth * T_step + T_lmhead + T_kv_write        (per decode step of B tokens)

combine:
  "sum" - every term runs one after another (a GPU: kernels are serial; one token through a chip pipeline).
  "max" - terms on different resources overlap (a pipelined chip). Terms that share
          one resource are added first: e.g. non-resident weights and the KV cache
          both come from external memory, so their bytes queue on the same bus.

v1 changes (review of v0, 2026-11-18):
  1. GPU weight time = bytes / bandwidth + a fixed cost per weight kernel (v0: bytes / bandwidth only, which
     predicts H100 weight time 31 % low: its GEMVs reach ~55 % of bandwidth at B=1).
  2. GPU constants are fitted from the raw run files (calibrate.py's "best GPU step" envelope + the fused
     kernel breakdown), not copied from WEEK5.md (whose 3.97 ms "weights" included 0.12 ms of attention).
     The H100 is predicted from the L40S fit: fixed costs scale with clock (cycle-bound), bandwidth terms
     with the H100's own measured bandwidth, nothing fitted on H100 timings.
  3. LoopCore overhead is modeled (per-layer pipeline latency + chip hops), not a fraction of the GPU's.
  4. LoopCore is compared against MEASURED best GPU steps: bf16 and int8, L40S and H100 (decision 4).
  5. Multi-chip pipelines: per-chip SRAM bandwidth and compute for latency (one token visits the chips in
     turn), n_chips hops per loop step (including the return to the first chip), plus per-hop latency.
  6. int8 configs (decision 2); chip figures cited.

All quantities are SI: bytes, seconds, FLOP, B/s, FLOP/s.

Usage:
  python loopcore_model.py --validate                    # L40S fit (in-sample) and H100 prediction
  python loopcore_model.py --validate --write-predictions  # also fill calibration_predictions*.json
  python loopcore_model.py                               # LoopCore configs vs measured best GPUs
  python loopcore_model.py --layer-latency-us 0.25 1 4   # sensitivity to the per-layer latency assumption
  python loopcore_model.py --kv-levers                   # KV design options: fp8/int4 KV, shared caches, on-chip KV
"""
import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import calibrate as cal

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- model under test

@dataclass(frozen=True)
class LoopedModel:
    name: str
    core_params: int     # recurrent block (all layers + final norm): read every loop step
    lmhead_params: int   # read once per decode step (untied in Ouro)
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    total_loops: int

    @property
    def d_model(self):
        return self.n_q_heads * self.head_dim

    def kv_elems_per_token_per_loop(self):
        """K and V for one cached token, all layers, one loop's cache."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim

    def attn_flops_per_cached_token(self):
        """QK^T (2·n_q·hd) + weighted sum of V (2·n_q·hd), per layer, per new token."""
        return 4 * self.n_layers * self.n_q_heads * self.head_dim


# Parameter counts verified with inspect_params.py at revision 574fa66.
OURO_1_4B = LoopedModel("Ouro-1.4B", core_params=1_233_324_032, lmhead_params=100_663_296,
                        n_layers=24, n_q_heads=16, n_kv_heads=16, head_dim=128, total_loops=4)

# Mean exit depth by exit threshold q: results/quality_20261030-170543.json (100 GSM8K questions, A6000).
# The Week 3 throughput bench (different prompts, 128 new tokens) measured ~3.1 (q=0.5) and ~2.4 (q=0.3).
MEAN_DEPTH = {1.0: 4.00, 0.5: 3.22, 0.3: 2.62}


# ---------------------------------------------------------------- hardware

@dataclass(frozen=True)
class Hardware:
    name: str
    weight_bw: float              # B/s at which resident core weights are read (per chip: SRAM; or DRAM on a GPU)
    ext_bw: float                 # B/s of external memory: KV cache, non-resident weights, streamed LM head
    compute: float                # FLOP/s at the operating precision (per chip)
    weight_bytes: float           # bytes per weight: 2 = bf16, 1 = 8-bit, 0.5 = 4-bit
    kv_bytes: float = 2.0         # bytes per KV element: 2 = bf16, 1 = fp8, 0.5 = int4
    kv_caches: int = 0            # distinct KV caches across loops: 0 = one per loop (Ouro as shipped),
                                  # 1 = one cache shared by all loops, 2 = one per pair of loops, ...
    kv_on_chip: bool = False      # KV cache held in SRAM (read/written at weight_bw) instead of external memory
    sram_bytes_per_chip: float = float("inf")
    n_chips: int = 1
    link_bw: float = float("inf")  # chip-to-chip B/s
    hop_latency_s: float = 0.0     # fixed latency per chip-to-chip hop
    resident_fraction: float = 1.0  # fraction of core weights held in SRAM
    lmhead_on_chip: bool = False
    overhead_s: float = 0.0        # fixed cost per loop step not tied to weight kernels
    weight_kernels: int = 0        # GPU: weight GEMV/GEMM kernels per loop step
    weight_kernel_s: float = 0.0   # GPU: fixed cost per weight kernel (launch, ramp-up, tail)
    per_seq_overhead_s: float = 0.0  # cost per sequence per step beyond ideal FLOPs (GPU calibration only)
    clock_hz: float = 0.0          # GPU boost clock; fixed per-kernel costs are scaled by it when transferring
    act_bytes: float = 2.0
    combine: str = "sum"
    source: str = ""


def n_kv_caches(m, hw):
    return hw.kv_caches or m.total_loops


def kv_bw(hw):
    return hw.weight_bw if hw.kv_on_chip else hw.ext_bw


def kv_footprint(m, hw, B, C):
    """KV cache bytes for B sequences of C tokens, all caches."""
    return B * C * m.kv_elems_per_token_per_loop() * hw.kv_bytes * n_kv_caches(m, hw)


def fit_check(m, hw, B=0, C=0):
    """(bytes needed in SRAM, SRAM capacity, fits?). With kv_on_chip, the KV cache at (B, C) counts too."""
    need = hw.resident_fraction * m.core_params * hw.weight_bytes
    if hw.lmhead_on_chip:
        need += m.lmhead_params * hw.weight_bytes
    if hw.kv_on_chip:
        need += kv_footprint(m, hw, B, C)
    cap = hw.sram_bytes_per_chip * hw.n_chips
    return need, cap, need <= cap


def sized(m, hw, B, C):
    """hw with the fewest chips that hold everything it keeps in SRAM at (B, C) (all-resident configs only).
    More chips also means more hops per loop step (step_terms)."""
    need, _, _ = fit_check(m, replace(hw, n_chips=1), B, C)
    return replace(hw, n_chips=max(1, math.ceil(need / hw.sram_bytes_per_chip)))


def step_terms(m, hw, B, C):
    """Per-loop-step terms in seconds, for B sequences each with C cached tokens.

    Multi-chip pipeline, latency view: one step's tokens visit every chip in turn, so the per-chip SRAM
    bandwidth and compute apply to the whole block (each chip reads its own share). Aggregate bandwidth only
    helps throughput with several micro-batches in flight, which this decode-latency model does not assume.
    """
    core = m.core_params * hw.weight_bytes
    f = hw.resident_fraction
    hops = hw.n_chips if hw.n_chips > 1 else 0      # recirculation: last chip sends back to the first
    t = {
        "weights_sram": f * core / hw.weight_bw + hw.weight_kernels * hw.weight_kernel_s,
        "weights_ext": (1 - f) * core / hw.ext_bw,
        "compute": (2 * m.core_params * B + m.attn_flops_per_cached_token() * B * C) / hw.compute
                   + hw.per_seq_overhead_s * B,
        # each executed loop reads one cache of C tokens, whether caches are per loop or shared
        "kv": B * C * m.kv_elems_per_token_per_loop() * hw.kv_bytes / kv_bw(hw),
        "link": hops * (hw.hop_latency_s + B * m.d_model * hw.act_bytes / hw.link_bw),
    }
    # resources: terms on the same one queue behind each other even when overlapped
    t["_by_resource"] = {
        "weight_mem": t["weights_sram"] + (t["kv"] if hw.kv_on_chip else 0),
        "ext_mem": t["weights_ext"] + (0 if hw.kv_on_chip else t["kv"]),
        "compute": t["compute"],
        "link": t["link"],
    }
    return t


def step_time(m, hw, B, C):
    t = step_terms(m, hw, B, C)
    res = t["_by_resource"]
    if hw.combine == "sum":
        busy = sum(res.values())
    elif hw.combine == "max":
        busy = max(res.values())
    else:
        raise ValueError(f"combine must be 'sum' or 'max', got {hw.combine!r}")
    bottleneck = max(res, key=res.get)
    return busy + hw.overhead_s, bottleneck


def per_token_extras(m, hw, B):
    """LM head (once per decode step) + KV writes for all loops (copy-forward, DESIGN_CDB.md §4)."""
    lm_bw = hw.weight_bw if hw.lmhead_on_chip else hw.ext_bw
    t_lmhead = m.lmhead_params * hw.weight_bytes / lm_bw + 2 * m.lmhead_params * B / hw.compute
    # Every cache needs an entry for every position: executed loops write their own KV,
    # skipped loops get a copy-forward. So KV writes per token = number of caches, whatever the exit depth.
    t_kv_write = B * n_kv_caches(m, hw) * m.kv_elems_per_token_per_loop() * hw.kv_bytes / kv_bw(hw)
    return t_lmhead + t_kv_write


def token_time(m, hw, B, C, mean_depth):
    """Time for one decode step that finishes B tokens (assumes CDB keeps all B slots full)."""
    t_step, _ = step_time(m, hw, B, C)
    return mean_depth * t_step + per_token_extras(m, hw, B)


def max_layer_latency(m, hw, B, C, mean_depth, target_s):
    """Largest per-layer latency (s) at which token_time <= target_s; None if unreachable even at 0.
    token_time is linear in overhead_s = n_layers * latency, entered once per executed loop step."""
    slack = target_s - token_time(m, replace(hw, overhead_s=0.0), B, C, mean_depth)
    return slack / (mean_depth * m.n_layers) if slack > 0 else None


# ---------------------------------------------------------------- GPU: fit on one GPU, predict another

# Datasheet values (not fitted): boost clock and dense BF16 tensor throughput.
#   L40S: 2520 MHz boost, 362 TFLOPS dense BF16 (NVIDIA L40S datasheet).
#   H100 SXM5: 1980 MHz boost, 989 TFLOPS dense BF16 (NVIDIA H100 datasheet).
DATASHEET = {"l40s": dict(clock_hz=2520e6, compute=362e12), "h100": dict(clock_hz=1980e6, compute=989e12)}
FUSED_RUN = {"l40s": "results/modal/delta_compile-l40s-20261116-214809",
             "h100": "results/modal/delta_compile-h100-20261116-214809"}


def achievable_bw(gpu):
    """The one measured hardware number used for a GPU: max of the copy peak (read + write bytes / time) and
    the torch.sum read peak, from the fused run's container. Both tests under-read the true read peak somewhat."""
    j = json.load(open(ROOT / FUSED_RUN[gpu] / "delta_split.json"))
    return max(j["measured_peak_gbps"], j["measured_read_gbps"]) * 1e9


def fit_gpu(gpu="l40s", m=OURO_1_4B):
    """GPU constants from raw run files only: the best-GPU-step envelope (calibrate.envelope, M2 fit) and the
    fused step's kernel breakdown at B=1, C=128."""
    env, _ = cal.envelope(gpu)
    delta, gamma, alpha, _ = cal.fit_m2([{"B": B, "C": C, "t": t} for (B, C), (t, _) in env.items()])
    kb = json.load(open(ROOT / FUSED_RUN[gpu] / "delta_split.json"))["kernel_breakdown"]["compiled_graph 1x128"]
    bw, ds = achievable_bw(gpu), DATASHEET[gpu]
    core = m.core_params * 2
    n_w = int(kb["gemm_kernels"])
    return Hardware(
        name=f"{gpu.upper()} (fitted)", weight_bw=bw,
        ext_bw=m.kv_elems_per_token_per_loop() * 2 / (alpha * 1e-3),       # α′ → KV bytes per second
        compute=ds["compute"], weight_bytes=2,
        weight_kernels=n_w, weight_kernel_s=(kb["gemm_ms"] * 1e-3 - core / bw) / n_w,
        overhead_s=(delta - kb["gemm_ms"]) * 1e-3,                        # intercept beyond the weight kernels
        per_seq_overhead_s=gamma * 1e-3 - 2 * m.core_params / ds["compute"],
        clock_hz=ds["clock_hz"], combine="sum",
        source=f"fitted: {gpu} envelope ({', '.join(cal.RUNS[gpu][2])}) + {FUSED_RUN[gpu]} breakdown")


def predict_gpu(src, gpu):
    """Transfer a fitted GPU to another GPU without its timings: fixed costs scale with the clock ratio,
    weight and KV bandwidth with the target's measured bandwidth (same efficiency as on the source)."""
    ds, bw = DATASHEET[gpu], achievable_bw(gpu)
    k = src.clock_hz / ds["clock_hz"]
    return replace(src, name=f"{gpu.upper()} (predicted from {src.name})", weight_bw=bw,
                   ext_bw=src.ext_bw / src.weight_bw * bw, compute=ds["compute"], clock_hz=ds["clock_hz"],
                   weight_kernel_s=src.weight_kernel_s * k, overhead_s=src.overhead_s * k,
                   source=f"predicted from {src.source}; {gpu} datasheet clock/compute; measured bandwidth")


# ---------------------------------------------------------------- GPU, kernel-sum model (v1.1)
#
# Pre-registered 2026-11-18, before the microbenchmarks ran: the fused step is a sequence of kernels whose
# counts come from the profiler (same on both GPUs), each costing its time measured IN ISOLATION on that GPU
# (delta_split.py --kernel-bench: synthetic tensors, CUDA graphs, no L2 reuse). Nothing is fitted on step
# timings and nothing is scaled by clock, so both GPUs are predictions.
#   step(B, C) = sum_shapes n_s * gemm_s(B) + n_small * small(B) + n_calls * (attn(B, 128) + kv * B * (C - 128))
# Counts: fused breakdown at 1x128 (compiled_graph): 170 GEMM kernels = 24 x (4 qkvo 2048x2048 + 2 gate/up
# 5632x2048 + 1 down 2048x5632) + 2 tiny (counted as small); 48 attention kernels = 24 SDPA calls; 123 other
# + 1 copy. Slot times (kernel + graph gap) give the step; kernel-only times give the weight/non-weight split.

KERNEL_BENCH = {"l40s": "results/modal/delta_kernel_bench-l40s-20261118-120529",
                "h100": "results/modal/delta_kernel_bench-h100-20261118-120542"}
# v1.2 (pre-registered 2026-11-18, before running): same sum, but (a) weight matmuls as torch.compile issues them
# (gemm_compiled: compiled lone F.linear per shape and B), for the compiled step's slower batched GEMMs; (b) small
# kernels measured in context, alternating with a 2048x2048 GEMV (small_after_gemv), for launch gaps that hide
# behind long kernels. Attention, KV and counts as v1.1. Recorded whatever it scores.
KERNEL_BENCH_V12 = {"l40s": "results/modal/delta_kernel_bench-l40s-20261118-123411",
                    "h100": "results/modal/delta_kernel_bench-h100-20261118-123424"}
KERNEL_COUNTS = dict(gemm={"2048x2048": 96, "5632x2048": 48, "2048x5632": 24}, small=126, attention_calls=24)
# v1.3 (pre-registered 2026-11-22, before scoring): as v1.2, except the weight term at B=1 is the in-step time of the
# weight kernels the best step actually runs (coordinate-descent fused reductions: qkv, o, gate+up, down), read from the
# cd run's 1x128 profile (calibrate.weight_kernel_ms, incl. the bmm-lowered o_proj), plus one launch gap per weight
# kernel taken from the v1.2 microbenchmark (slot - busy, mean over shapes at B=1). B >= 2 unchanged (the step uses
# tensor-core GEMMs there, like the microbenchmark). Small kernels, attention, KV, counts unchanged.
CD_RUN = {"l40s": "results/modal/delta_bf16_cd-l40s-20261120-010859",
          "h100": "results/modal/delta_bf16_cd-h100-20261120-010911"}


def instep_b1(gpu, m=OURO_1_4B):
    """(busy µs, kernel count) of the weight kernels in the cd step's 1x128 profile."""
    bd = json.load(open(ROOT / CD_RUN[gpu] / "delta_split.json"))["kernel_breakdown"]["compiled_graph 1x128"]
    # older breakdowns filed the o_proj kernels (one per layer) outside gemm_kernels
    n = bd["gemm_kernels"] + (m.n_layers if bd.get("kernel_cats_version", 1) < 2 else 0)
    return cal.weight_kernel_ms(bd) * 1e3, n


def instep_weights_b1(gpu, field):
    """v1.3 weight term at B=1, µs per loop step."""
    busy, n = instep_b1(gpu)
    if field == "busy_us":
        return busy
    gemm = json.load(open(ROOT / KERNEL_BENCH_V12[gpu] / "kernel_bench.json"))["gemm_compiled"]
    return busy + n * statistics.mean(t["1"]["slot_us"] - t["1"]["busy_us"] for t in gemm.values())


# v1.4 (pre-registered 2026-11-22, before scoring): as v1.3, except every kernel costs its busy time plus ONE in-graph
# gap per kernel, g, instead of the microbenchmark's slot time (slot - busy = 0.2-1.2 µs, measured with kernels
# alternating outside the real graph). g = mean over the GPU's delta_noise runs of (step - busy) / kernels: separate
# containers, fused non-cd step, not calibration targets. Runs whose device is not this GPU are dropped (one "h100"
# noise run was an H200). Kernel counts: weights 168 (B >= 2) or the in-step count (B=1), small 126, attention 48.
def ingraph_gap_us(gpu):
    vals = []
    for f in sorted(ROOT.glob(f"results/modal/delta_noise-{gpu}-*/noise.json")):
        j = json.load(open(f))
        if j["gpu"] == cal.RUNS[gpu][0]:
            vals.append((j["step"]["median_ms"] - j["busy_ms"]) / j["kernels"] * 1e3)
    return statistics.mean(vals)


def _interp(table, B, field):
    """Value at batch B from {str(B): {field: v}}, linear in B between measured batches."""
    pts = sorted((int(b), v[field]) for b, v in table.items())
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if b0 <= B <= b1:
            return v0 + (v1 - v0) * (B - b0) / (b1 - b0)
    return pts[0][1] if B < pts[0][0] else pts[-1][1]


# v1.5 (pre-registered 2026-11-22, before scoring): as v1.4, except o_proj at B >= 2. In the best step inductor lowers
# o_proj as a bmm and keeps it a Triton reduction (no tensor cores) at every B, whose time grows with B; the
# microbenchmark (a lone linear -> tensor-core GEMM, flat in B) cannot see that. o_proj per call = straight line through
# its in-step time at B=1 and B=16 (cd run profiles; B=16 = mean of 16x512/2048/8192), extrapolated beyond 16.
def oproj_instep_us(gpu, B, m=OURO_1_4B):
    kb = json.load(open(ROOT / CD_RUN[gpu] / "delta_split.json"))["kernel_breakdown"]
    o = lambda key: sum(v for k, v in kb[key]["top_kernels_ms"].items() if "fused_bmm_" in k.lower()) * 1e3 / m.n_layers
    o1, o16 = o("compiled_graph 1x128"), statistics.mean(o(f"compiled_graph 16x{c}") for c in (512, 2048, 8192))
    return o1 + (o16 - o1) * (B - 1) / 15


def kernel_sum(gpu, B, C, field="slot_us", version="v1.1"):
    """(weights, small, attention) in µs per loop step from the GPU's own microbenchmarks."""
    if version == "v1.5":
        w, s, a = kernel_sum(gpu, B, C, field, "v1.4")
        if B >= 2:   # swap the microbenchmark's o_proj (one of the 2048x2048 kernels per layer) for the in-step one
            gemm = json.load(open(ROOT / KERNEL_BENCH_V12[gpu] / "kernel_bench.json"))["gemm_compiled"]
            w += OURO_1_4B.n_layers * (oproj_instep_us(gpu, B) - _interp(gemm["2048x2048"], B, "busy_us"))
        return w, s, a
    if version == "v1.4":
        w, s, a = kernel_sum(gpu, B, C, "busy_us", "v1.3")
        if field == "busy_us":
            return w, s, a
        g = ingraph_gap_us(gpu)
        n_w = instep_b1(gpu)[1] if B == 1 else sum(KERNEL_COUNTS["gemm"].values())
        return w + n_w * g, s + KERNEL_COUNTS["small"] * g, a + 2 * KERNEL_COUNTS["attention_calls"] * g
    v12 = version in ("v1.2", "v1.3")
    run = KERNEL_BENCH_V12[gpu] if v12 else KERNEL_BENCH[gpu]
    kb = json.load(open(ROOT / run / "kernel_bench.json"))
    gemm = kb["gemm_compiled"] if v12 else kb["gemm"]
    small_tab = kb["small_after_gemv"] if v12 else kb["small"]
    if version == "v1.3" and B == 1:
        w = instep_weights_b1(gpu, field)
    else:
        w = sum(n * _interp(gemm[shape], B, field) for shape, n in KERNEL_COUNTS["gemm"].items())
    small = KERNEL_COUNTS["small"] * statistics.mean(_interp(t, B, field) for t in small_tab.values())
    att = kb["attention"]
    a128 = _interp({b: att[f"{b}x128"] for b in ("1", "16", "64")}, B, field)
    kv = (att["16x2048"][field] - att["16x128"][field]) / (16 * (2048 - 128))     # µs per cached token per call
    return w, small, KERNEL_COUNTS["attention_calls"] * (a128 + kv * B * (C - 128))


def kernel_sum_predictions(gpu, version="v1.5"):
    """calibrate.py predictions from the kernel-sum model."""
    env, _ = cal.envelope(gpu)
    grid = {f"{B}x{C}": sum(kernel_sum(gpu, B, C, version=version)) / 1e3 for (B, C) in env}
    delta, _, _, _ = cal.fit_m2([{"B": B, "C": C, "t": grid[f"{B}x{C}"]} for (B, C) in env])
    run = KERNEL_BENCH_V12[gpu] if version in ("v1.2", "v1.3", "v1.4", "v1.5") else KERNEL_BENCH[gpu]
    kb = json.load(open(ROOT / run / "kernel_bench.json"))
    w_busy = kernel_sum(gpu, 1, 128, "busy_us", version=version)[0] / 1e3
    att = kb["attention"]
    kv = KERNEL_COUNTS["attention_calls"] * (att["16x2048"]["slot_us"] - att["16x128"]["slot_us"]) / (16 * 1920)
    return {"date": __import__("time").strftime("%Y-%m-%d"),
            "source": f"loopcore_model.py {version} kernel-sum: {run}/kernel_bench.json",
            "inputs": {"kernel_counts": KERNEL_COUNTS, "microbenchmarks": run,
                       "measured_copy_peak_GBps": kb["measured_peak_gbps"], "measured_read_peak_GBps": kb["measured_read_gbps"]},
            "predictions": {"weight_ms": w_busy, "kv_us_per_token": kv, "crossover_tokens": w_busy * 1e3 / kv,
                            "step_ms_b1_c128": grid["1x128"], "delta_ms": delta, "overhead_ms": grid["1x128"] - w_busy},
            "grid": grid}


# ---------------------------------------------------------------- measured GPU baselines (decision 4)

INT8_RUNS = {"l40s": ["results/modal/delta_int8_triton-l40s-20261117-233833",
                      "results/modal/delta_int8_cd_small-l40s-20261117-233859"],
             "h100": ["results/modal/delta_int8_triton-h100-20261117-233820",
                      "results/modal/delta_int8_cd_small-h100-20261117-233847",
                      "results/modal/delta_int8_triton-h100-20261120-010924"]}   # small-M kernel added (never chosen)


def measured_steps(gpu, precision):
    """{(B, C): ms}: fastest measured step per point. bf16 = calibrate.envelope; int8 = best int8 path."""
    if precision == "bf16":
        return {bc: t for bc, (t, _) in cal.envelope(gpu)[0].items()}
    best = {}
    for run in INT8_RUNS[gpu]:
        for r in csv.DictReader(open(ROOT / run / "graph_grid.csv")):
            bc, t = (int(r["B"]), int(r["C"])), float(r["median_ms"])
            best[bc] = min(t, best.get(bc, t))
    return best


# ---------------------------------------------------------------- LoopCore configurations
#
# ILLUSTRATIVE, not a design. Per-chip figures of the NVIDIA Groq 3 LPU:
#   500 MB SRAM, 150 TB/s SRAM bandwidth, 2.5 TB/s chip-to-chip (NVIDIA, https://www.nvidia.com/en-gb/data-center/lpx;
#   https://www.storagereview.com/news/nvidia-groq-3-lpx-everything-we-know), ~1.2 PFLOPS FP8 per chip
#   (9.6 PFLOPS per 8-chip tray; https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/; dense vs sparse not
#   stated by NVIDIA). The LPU has no external DRAM, so EXT_BW (KV cache) is an ASSUMPTION: one HBM3 interface
#   at H100-SXM speed (3.35 TB/s datasheet). HOP_LATENCY and LAYER_LATENCY are ASSUMPTIONS too (sweep them).

SRAM_MB, SRAM_BW, COMPUTE_FP8, LINK_BW = 500e6, 150e12, 1.2e15, 2.5e12
EXT_BW = 3.35e12          # ASSUMPTION
HOP_LATENCY = 1e-6        # ASSUMPTION: per chip-to-chip hop
LAYER_LATENCY = 1e-6      # ASSUMPTION (default): fixed pipeline latency per layer per loop step, covering the
                          # non-GEMM work (norms, rotary, activations, softmax) of a fused dataflow chip.
                          # For scale: the fused GPU spends (overhead_s / n_layers) ≈ 23 µs per layer (L40S).

_core_4bit = OURO_1_4B.core_params * 0.5
_chip = dict(weight_bw=SRAM_BW, ext_bw=EXT_BW, compute=COMPUTE_FP8, sram_bytes_per_chip=SRAM_MB,
             link_bw=LINK_BW, hop_latency_s=HOP_LATENCY, combine="sum")

LOOPCORE_CONFIGS = {
    "A: 2 chips, 4-bit, all resident": Hardware(name="A", weight_bytes=0.5, n_chips=2, lmhead_on_chip=True, **_chip),
    "B: 1 chip, 4-bit, partial residency": Hardware(
        name="B", weight_bytes=0.5, resident_fraction=min(1.0, SRAM_MB / _core_4bit), **_chip),
    "C: 1 hypothetical 700 MB chip, 4-bit": Hardware(
        name="C", weight_bytes=0.5, **{**_chip, "sram_bytes_per_chip": 700e6}),
    "D: 3 chips, 8-bit, all resident": Hardware(name="D", weight_bytes=1, n_chips=3, lmhead_on_chip=True, **_chip),
}


# ---------------------------------------------------------------- reports

def grid_errors(m, hw, gpu):
    env, _ = cal.envelope(gpu)
    return {(B, C): 100 * (step_time(m, hw, B, C)[0] * 1e3 - t) / t for (B, C), (t, _) in env.items()}


def predictions(m, hw, gpu):
    """calibrate.py predictions file content for this hardware model."""
    env, _ = cal.envelope(gpu)
    grid = {f"{B}x{C}": step_time(m, hw, B, C)[0] * 1e3 for (B, C) in env}
    delta, _, _, _ = cal.fit_m2([{"B": B, "C": C, "t": grid[f"{B}x{C}"]} for (B, C) in env])
    w = step_terms(m, hw, 1, 128)["weights_sram"] * 1e3
    kv = m.kv_elems_per_token_per_loop() * hw.kv_bytes / hw.ext_bw * 1e6
    return {"date": None, "source": f"loopcore_model.py v1: {hw.name}",
            "inputs": {"weight_bytes": m.core_params * hw.weight_bytes, "weight_bandwidth_GBps": hw.weight_bw / 1e9,
                       "kv_bytes_per_token": m.kv_elems_per_token_per_loop() * hw.kv_bytes,
                       "kv_bandwidth_GBps": hw.ext_bw / 1e9,
                       "overhead_term": f"{hw.weight_kernels} weight kernels x {hw.weight_kernel_s * 1e6:.2f} us "
                                        f"+ {hw.overhead_s * 1e3:.3f} ms per step + {hw.per_seq_overhead_s * 1e6:.1f} us/seq",
                       "fitted_on": hw.source},
            "predictions": {"weight_ms": w, "kv_us_per_token": kv, "crossover_tokens": w * 1e3 / kv,
                            "step_ms_b1_c128": grid["1x128"], "delta_ms": delta, "overhead_ms": grid["1x128"] - w},
            "grid": grid}


def validate(m=OURO_1_4B, write=False):
    l40s = fit_gpu("l40s", m)
    h100 = predict_gpu(l40s, "h100")
    print(f"Fitted on L40S: weights {l40s.weight_kernels} kernels x {l40s.weight_kernel_s * 1e6:.2f} µs + bytes at "
          f"{l40s.weight_bw / 1e9:.0f} GB/s; overhead {l40s.overhead_s * 1e3:.3f} ms/step; "
          f"KV at {l40s.ext_bw / 1e9:.0f} GB/s; {l40s.per_seq_overhead_s * 1e6:.1f} µs/seq.")
    print(f"H100 predicted (no H100 timings used): clock ratio {l40s.clock_hz / h100.clock_hz:.3f}, "
          f"bandwidth {h100.weight_bw / 1e9:.0f} GB/s (measured), compute {h100.compute / 1e12:.0f} TFLOPS (datasheet).\n")
    for label, hw, gpu in (("L40S, in-sample", l40s, "l40s"), ("H100, predicted", h100, "h100")):
        e = grid_errors(m, hw, gpu)
        (wB, wC), we = max(e.items(), key=lambda kv: abs(kv[1]))
        print(f"{label:16s} grid: median |error| {statistics.median(map(abs, e.values())):.1f} %, "
              f"max {we:+.1f} % at B={wB}, C={wC} ({len(e)} points)")
    print("\nCaveat: four GPU-transfer variants were tried against the H100 data before choosing this one (clock-scaled")
    print("fixed costs, max(copy, read) bandwidth). Confirm on a third GPU (e.g. A100) before trusting the transfer rule.")
    print("Known residuals: (1) γ (per-sequence cost) is not transferred physically; it is the fused step's slower")
    print("batched GEMMs, which scale differently per GPU (L40S 0.070, H100 0.051 ms/seq). (2) Charging γ from the first")
    print("sequence over-predicts the B=1 step's non-weight time (calibrate.py: +56 % L40S, +20 % H100), the B=1 -> B=2")
    print("GEMV -> GEMM jump v0 noted. Anchoring the overhead at the measured B=1 step fixes L40S but worsens the H100")
    print("prediction (median 12.6 %), so it is not used; more variants tuned against H100 would overfit it.")
    have_bench = all((ROOT / KERNEL_BENCH_V12[g] / "kernel_bench.json").exists() for g in KERNEL_BENCH_V12)
    for version in ("v1.1", "v1.2", "v1.3", "v1.4", "v1.5"):
        print(f"\n{version} kernel-sum model (microbenchmarks only; both GPUs are predictions):")
        for gpu in ("l40s", "h100"):
            env, _ = cal.envelope(gpu)
            e = {bc: 100 * (sum(kernel_sum(gpu, *bc, version=version)) / 1e3 - t) / t for bc, (t, _) in env.items()}
            (wB, wC), we = max(e.items(), key=lambda kv: abs(kv[1]))
            w, sm, at = kernel_sum(gpu, 1, 128, version=version)
            print(f"  {gpu.upper():5s} grid: median |error| {statistics.median(map(abs, e.values())):.1f} %, max {we:+.1f} % "
                  f"at B={wB}, C={wC}; 1x128: weights {w / 1e3:.3f} + small {sm / 1e3:.3f} + attention {at / 1e3:.3f} ms")
    print("\nKnown misses (2026-11-18): (1) inter-kernel gaps depend on the neighbouring kernels; isolated benchmarks give")
    print("0.2-1.2 µs per kernel vs ~0.25 µs (L40S) / ~0.67 µs (H100) inside the real step, so the B=1 non-weight time is")
    print("+33 % (L40S, v1.2) / -14 % (H100, v1.2). (2) The compiled step's batched GEMMs at B >= 32 are slower than")
    print("isolated cuBLAS, and a lone compiled linear does not reproduce that (same kernels, same times), so the grid max")
    print("misses (-26 % L40S, -46 % H100). (3) The B=1 non-weight target itself varies ~±10-12 % between containers")
    print("(L40S 0.32-0.41 ms, H100 0.52-0.63 ms): H100 v1.1 and v1.2 are inside that spread; L40S is a real miss.")
    if write:
        for gpu, hw in (("l40s", l40s), ("h100", h100)):
            p = ROOT / cal.RUNS[gpu][1]
            # pre-registered: the latest kernel-sum model's (v1.5) predictions are recorded, whatever they score
            pred = kernel_sum_predictions(gpu, "v1.5") if have_bench else \
                {**predictions(m, hw, gpu), "date": __import__("time").strftime("%Y-%m-%d")}
            p.write_text(json.dumps(pred, indent=2) + "\n")
            print(f"wrote {p.name} ({'v1.5 kernel-sum' if have_bench else 'v1 transfer'} model)")
    return l40s, h100


def compare(m, configs, combine, layer_latencies, q, Bs=(1, 8, 64), Cs=(128, 2048)):
    depth = MEAN_DEPTH[q]
    gpus = {"l40s": fit_gpu("l40s", m)}
    gpus["h100"] = predict_gpu(gpus["l40s"], "h100")   # only for its LM-head / KV-write terms
    base = {(g, p): measured_steps(g, p) for g in ("l40s", "h100") for p in ("bf16", "int8")}
    cols = [("l40s", "bf16"), ("l40s", "int8"), ("h100", "bf16"), ("h100", "int8")]
    print(f"Speedup in tokens/s vs MEASURED best GPU steps (CALIBRATION.md decision 4); combine={combine}; q={q} "
          f"(mean depth {depth}).\nGPU token = depth x measured step + LM head + KV writes. int8 = best int8 path; "
          f"H100 int8 at B=2-16 is kernel-limited (a lower bound on that GPU).\n")
    for label, hw in configs.items():
        need, cap, fits = fit_check(m, hw)
        print(f"{label}: SRAM need {need / 1e6:.0f} MB of {cap / 1e6:.0f} MB "
              f"{'OK' if fits else 'DOES NOT FIT'}, resident fraction {hw.resident_fraction:.2f}")
        if not fits:
            print("  skipped\n")
            continue
        for lat in layer_latencies:
            h = replace(hw, combine=combine, overhead_s=m.n_layers * lat)
            print(f"  per-layer latency {lat * 1e6:g} µs -> overhead {h.overhead_s * 1e6:.0f} µs/step")
            print(f"  {'B':>3} {'C':>5}  " + "  ".join(f"{g.upper()} {p}" for g, p in cols) + "   LoopCore ms/token  bottleneck")
            for B in Bs:
                for C in Cs:
                    lc = token_time(m, h, B, C, depth)
                    cells = []
                    for g, p in cols:
                        t = base[(g, p)].get((B, C))
                        cells.append(f"{(depth * t * 1e-3 + per_token_extras(m, gpus[g], B)) / lc:8.1f}x" if t else "     n/a")
                    _, bn = step_time(m, h, B, C)
                    print(f"  {B:>3} {C:>5}  " + "  ".join(cells) + f"   {lc * 1e3:10.3f}       {bn}")
        print()


KV_LEVERS = {   # cumulative, as in HANDOFF.md §7
    "today (bf16 KV, 4 caches, HBM)": {},
    "+ fp8 KV": dict(kv_bytes=1.0),
    "+ 1 shared cache": dict(kv_bytes=1.0, kv_caches=1),
    "+ KV on-chip": dict(kv_bytes=1.0, kv_caches=1, kv_on_chip=True),
}


def kv_levers(m, levers, q=0.5, layer_latency=5e-6, points=((1, 2048), (1, 8192), (16, 2048), (16, 8192)),
              targets=(3, 10)):
    """Config D (8-bit, all resident) with each KV option, sized to fit, vs MEASURED H100 int8 (decision 4)."""
    depth = MEAN_DEPTH[q]
    h100 = predict_gpu(fit_gpu("l40s", m), "h100")      # only for the GPU's LM-head / KV-write terms
    gpu8 = measured_steps("h100", "int8")
    base = replace(LOOPCORE_CONFIGS["D: 3 chips, 8-bit, all resident"], overhead_s=m.n_layers * layer_latency)
    print(f"KV options on config D (8-bit, all resident), vs measured H100 int8; q={q} (depth {depth}), "
          f"{layer_latency * 1e6:g} µs/layer.\nChips = fewest that hold weights + LM head (+ KV if on-chip). "
          f"'need <x' = per-layer latency required for that speedup.\nCaveat: the GPU could use fp8 KV / fewer caches "
          f"too; only the on-chip gain is LoopCore-only.\n")
    print(f"{'option':32s} {'B':>3} {'C':>5}  speedup chips  KV MB   " + "  ".join(f"need <{n}x" for n in targets))
    for label, kw in levers.items():
        for B, C in points:
            hw = sized(m, replace(base, **kw), B, C)
            gpu_t = depth * gpu8[(B, C)] * 1e-3 + per_token_extras(m, h100, B)
            lats = [max_layer_latency(m, hw, B, C, depth, gpu_t / n) for n in targets]
            print(f"{label:32s} {B:>3} {C:>5}  {gpu_t / token_time(m, hw, B, C, depth):6.1f}x {hw.n_chips:>5} "
                  f"{kv_footprint(m, hw, B, C) / 1e6:6.0f}   " +
                  "  ".join(f"{l * 1e6:6.1f} µs" if l else "     n/a" for l in lats))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="L40S fit (in-sample) and H100 prediction")
    ap.add_argument("--write-predictions", action="store_true", help="with --validate: write calibration_predictions*.json")
    ap.add_argument("--combine", choices=["sum", "max"], default="sum")
    ap.add_argument("--layer-latency-us", type=float, nargs="+", default=[LAYER_LATENCY * 1e6])
    ap.add_argument("--q", type=float, choices=sorted(MEAN_DEPTH), default=0.5)
    ap.add_argument("--kv-levers", action="store_true", help="KV design options on config D vs measured H100 int8")
    args = ap.parse_args()
    if args.kv_levers:
        for lat in args.layer_latency_us if args.layer_latency_us != [LAYER_LATENCY * 1e6] else [5.0]:
            kv_levers(OURO_1_4B, KV_LEVERS, args.q, lat * 1e-6)
    elif args.validate:
        validate(write=args.write_predictions)
    else:
        compare(OURO_1_4B, LOOPCORE_CONFIGS, args.combine, [x * 1e-6 for x in args.layer_latency_us], args.q)


if __name__ == "__main__":
    main()
