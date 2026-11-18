"""LoopCore analytical model, v0 (Weeks 5-6).

Predicts decode time per loop step and per token for a looped LM (Ouro-1.4B) on
  (a) a GPU, calibrated from LoopProf measurements, and
  (b) LoopCore configurations that keep recurrent-core weights in on-chip SRAM.

Terms (the shared definitions in CALIBRATION.md):
  T_step  = combine(T_weights, T_compute, T_kv, T_link) + T_overhead
  T_token = mean_depth * T_step + T_lmhead + T_kv_write        (per decode step of B tokens)

combine:
  "sum" - every term runs one after another (a GPU: kernels are serial).
  "max" - terms on different resources overlap (a pipelined chip). Terms that share
          one resource are added first: e.g. non-resident weights and the KV cache
          both come from external memory, so their bytes queue on the same bus.

All quantities are SI: bytes, seconds, FLOP, B/s, FLOP/s.

Usage:
  python loopcore_model.py --validate             # model vs Sid's L40S delta_grid measurements
  python loopcore_model.py                        # LoopCore configs vs calibrated L40S
  python loopcore_model.py --combine max --overhead-frac 0 0.25 1
"""
import argparse
from dataclasses import dataclass, replace


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

# Mean exit depth, 100-question GSM8K quality run (A6000), by exit threshold q.
MEAN_DEPTH = {1.0: 4.00, 0.5: 3.22, 0.3: 2.62}


# ---------------------------------------------------------------- hardware

@dataclass(frozen=True)
class Hardware:
    name: str
    weight_bw: float              # B/s at which resident core weights are read (SRAM; or DRAM on a GPU)
    ext_bw: float                 # B/s of external memory: KV cache, non-resident weights, streamed LM head
    compute: float                # FLOP/s at the operating precision
    weight_bytes: float           # bytes per weight: 2 = bf16, 1 = 8-bit, 0.5 = 4-bit
    kv_bytes: float = 2.0         # bytes per KV element
    sram_bytes_per_chip: float = float("inf")
    n_chips: int = 1
    link_bw: float = float("inf")  # chip-to-chip B/s (activations cross each boundary once per loop step)
    resident_fraction: float = 1.0  # fraction of core weights held in SRAM
    lmhead_on_chip: bool = False
    overhead_s: float = 0.0        # fixed cost per loop step
    per_seq_overhead_s: float = 0.0  # cost per sequence per step beyond ideal FLOPs (GPU calibration only)
    act_bytes: float = 2.0
    combine: str = "sum"
    source: str = ""


def fit_check(m, hw):
    """(bytes needed in SRAM, SRAM capacity, fits?)"""
    need = hw.resident_fraction * m.core_params * hw.weight_bytes
    if hw.lmhead_on_chip:
        need += m.lmhead_params * hw.weight_bytes
    cap = hw.sram_bytes_per_chip * hw.n_chips
    return need, cap, need <= cap


def step_terms(m, hw, B, C):
    """Per-loop-step terms in seconds, for B sequences each with C cached tokens."""
    core = m.core_params * hw.weight_bytes
    f = hw.resident_fraction
    t = {
        "weights_sram": f * core / hw.weight_bw,
        "weights_ext": (1 - f) * core / hw.ext_bw,
        "compute": (2 * m.core_params * B + m.attn_flops_per_cached_token() * B * C) / hw.compute
                   + hw.per_seq_overhead_s * B,
        "kv": B * C * m.kv_elems_per_token_per_loop() * hw.kv_bytes / hw.ext_bw,
        "link": (hw.n_chips - 1) * B * m.d_model * hw.act_bytes / hw.link_bw,
    }
    # resources: terms on the same one queue behind each other even when overlapped
    t["_by_resource"] = {
        "weight_mem": t["weights_sram"],
        "ext_mem": t["weights_ext"] + t["kv"],
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


def token_time(m, hw, B, C, mean_depth):
    """Time for one decode step that finishes B tokens (assumes CDB keeps all B slots full)."""
    t_step, _ = step_time(m, hw, B, C)
    lm_bw = hw.weight_bw if hw.lmhead_on_chip else hw.ext_bw
    t_lmhead = m.lmhead_params * hw.weight_bytes / lm_bw + 2 * m.lmhead_params * B / hw.compute
    # Every loop's cache needs an entry for every position: executed loops write their own KV,
    # skipped loops get a copy-forward. So KV writes per token = total_loops, whatever the exit depth.
    t_kv_write = B * m.total_loops * m.kv_elems_per_token_per_loop() * hw.kv_bytes / hw.ext_bw
    return mean_depth * t_step + t_lmhead + t_kv_write


# ---------------------------------------------------------------- calibrated GPU baseline

# L40S, from delta_split_grid.md (CUDA graph, static cache, bf16):
#   weights: GEMM kernel time 3.97 ms at B=1, C=128  -> 2.4666 GB / 3.97 ms = 621 GB/s
#   KV:      flash-attention 8.79 ms for 32,768 cached tokens -> 733 GB/s read-only
#   overhead: 1.64 ms small kernels + 0.25 ms host gaps per step
#   gamma:   fitted 15.2 us/seq; ideal compute 2*P/362 TFLOPS = 6.8 us; the rest is per_seq_overhead
_W_BF16 = OURO_1_4B.core_params * 2
L40S_MEASURED = Hardware(
    name="L40S (calibrated)",
    weight_bw=_W_BF16 / 3.97e-3,
    ext_bw=733e9,
    compute=362e12,
    weight_bytes=2,
    overhead_s=1.64e-3 + 0.25e-3,
    per_seq_overhead_s=15.2e-6 - 2 * OURO_1_4B.core_params / 362e12,
    combine="sum",
    source="delta_split_grid.md; datasheet 864 GB/s, 362 TFLOPS dense BF16",
)

# (B, C, measured ms) from delta_split_grid.md, "GPU (graph) ms" column.
L40S_GRID = [
    (1, 128, 5.82), (2, 128, 6.28), (4, 128, 6.35), (8, 128, 6.56), (16, 128, 7.06), (32, 128, 7.77), (64, 128, 9.11),
    (1, 256, 5.89), (2, 256, 6.32), (4, 256, 6.52), (8, 256, 6.93), (16, 256, 7.58), (32, 256, 8.87), (64, 256, 11.37),
    (1, 512, 5.93), (2, 512, 6.46), (4, 512, 6.84), (8, 512, 7.38), (16, 512, 8.65), (32, 512, 11.09), (64, 512, 15.67),
    (1, 1024, 6.10), (2, 1024, 6.75), (4, 1024, 7.39), (8, 1024, 8.67), (16, 1024, 10.80), (32, 1024, 15.67),
    (64, 1024, 24.39), (1, 2048, 6.42), (2, 2048, 7.31), (4, 2048, 8.37), (8, 2048, 10.63), (16, 2048, 15.08),
    (32, 2048, 24.00), (64, 2048, 41.70), (1, 4096, 6.92), (2, 4096, 8.39), (4, 4096, 10.53), (8, 4096, 15.18),
    (16, 4096, 23.73), (32, 4096, 41.38), (1, 8192, 8.27), (2, 8192, 10.68), (4, 8192, 14.86), (8, 8192, 23.63),
    (16, 8192, 40.99),
]


# ---------------------------------------------------------------- LoopCore configurations
#
# ILLUSTRATIVE, not a design. Chip numbers are Groq 3 LPU public figures (per chip: 500 MB SRAM,
# 150 TB/s SRAM bandwidth, 1.2 PFLOPS FP8, 2.5 TB/s chip-to-chip). Groq has no external DRAM, so
# EXT_BW is an ASSUMPTION: one H100-SXM-class HBM3 interface (3.35 TB/s datasheet) for the KV cache.
# Replace these with your own chip parameters before quoting any result.

SRAM_MB, SRAM_BW, COMPUTE_FP8, LINK_BW = 500e6, 150e12, 1.2e15, 2.5e12
EXT_BW = 3.35e12  # ASSUMPTION

_core_4bit = OURO_1_4B.core_params * 0.5

LOOPCORE_CONFIGS = {
    "A: 2 chips, 4-bit, all resident": Hardware(
        name="A", weight_bw=SRAM_BW * 2, ext_bw=EXT_BW, compute=COMPUTE_FP8 * 2, weight_bytes=0.5,
        sram_bytes_per_chip=SRAM_MB, n_chips=2, link_bw=LINK_BW, lmhead_on_chip=True, combine="sum"),
    "B: 1 chip, 4-bit, partial residency": Hardware(
        name="B", weight_bw=SRAM_BW, ext_bw=EXT_BW, compute=COMPUTE_FP8, weight_bytes=0.5,
        sram_bytes_per_chip=SRAM_MB, resident_fraction=min(1.0, SRAM_MB / _core_4bit), combine="sum"),
    "C: 1 hypothetical 700 MB chip, 4-bit": Hardware(
        name="C", weight_bw=SRAM_BW, ext_bw=EXT_BW, compute=COMPUTE_FP8, weight_bytes=0.5,
        sram_bytes_per_chip=700e6, combine="sum"),
}


# ---------------------------------------------------------------- reports

def validate(m=OURO_1_4B, hw=L40S_MEASURED):
    print(f"Model vs measured, {hw.name}. NOTE: constants come from this same dataset, so this checks the")
    print("model's structure (B·C scaling, additive terms), not its predictive power. The H100 run is the real test.\n")
    print(f"{'B':>3} {'C':>5} {'meas ms':>8} {'pred ms':>8} {'err %':>6}  bottleneck")
    errs = []
    for B, C, meas in L40S_GRID:
        pred, bn = step_time(m, hw, B, C)
        err = 100 * (pred * 1e3 - meas) / meas
        errs.append(abs(err))
        print(f"{B:>3} {C:>5} {meas:>8.2f} {pred * 1e3:>8.2f} {err:>+6.1f}  {bn}")
    print(f"\nmean |error| {sum(errs) / len(errs):.1f}%, max |error| {max(errs):.1f}% over {len(errs)} configs")
    print("Known residual: B=2-32 are ~0.3 ms (4-6%) slow vs the model. Measured time jumps 0.46 ms from B=1 to")
    print("B=2, which a linear gamma can't produce; likely cuBLAS switching from a GEMV to a GEMM kernel. Not fixed.")
    return errs


def compare(m, gpu, configs, combine, overhead_fracs, q, Bs=(1, 8, 64), Cs=(128, 2048)):
    depth = MEAN_DEPTH[q]
    print(f"Speedup in tokens/s vs {gpu.name}; combine={combine}; exit threshold q={q} (mean depth {depth});")
    print(f"LoopCore overhead = overhead_frac x GPU's {gpu.overhead_s * 1e3:.2f} ms per step.\n")
    for label, hw in configs.items():
        need, cap, fits = fit_check(m, hw)
        print(f"{label}: SRAM need {need / 1e6:.0f} MB of {cap / 1e6:.0f} MB "
              f"{'OK' if fits else 'DOES NOT FIT'}, resident fraction {hw.resident_fraction:.2f}")
        if not fits:
            print("  skipped\n")
            continue
        print(f"  {'B':>3} {'C':>5}  " + "  ".join(f"ovh={x:<4}" for x in overhead_fracs) + "   bottleneck (ovh=0)")
        for B in Bs:
            for C in Cs:
                g = token_time(m, gpu, B, C, depth)
                cells = []
                for x in overhead_fracs:
                    h = replace(hw, combine=combine, overhead_s=x * gpu.overhead_s)
                    cells.append(f"{g / token_time(m, h, B, C, depth):7.1f}x")
                _, bn = step_time(m, replace(hw, combine=combine), B, C)
                print(f"  {B:>3} {C:>5}  " + "  ".join(cells) + f"   {bn}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="model vs L40S delta_grid measurements")
    ap.add_argument("--combine", choices=["sum", "max"], default="sum")
    ap.add_argument("--overhead-frac", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    ap.add_argument("--q", type=float, choices=sorted(MEAN_DEPTH), default=0.5)
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        compare(OURO_1_4B, L40S_MEASURED, LOOPCORE_CONFIGS, args.combine, args.overhead_frac, args.q)


if __name__ == "__main__":
    main()
