"""results/NOTEBOOK_KV_SHARING.md, Step 1.4. KV size and speedup for per-loop / decode-only-shared / fully-shared KV, using loopcore_model.py as-is.
Decode-only sharing: P prompt tokens keep 4 caches, G generated tokens share 1. KV read per loop step is C=P+G
tokens in every design (step_terms), so only placement (on-chip vs HBM), chip count and KV writes differ.
Footprint = kv_footprint(4 caches, P) + kv_footprint(1 cache, G); chips sized to hold weights + LM head + that."""
import math
from dataclasses import replace

import loopcore_model as lm

m, q, C = lm.OURO_1_4B, 0.5, 2048
depth = lm.MEAN_DEPTH[q]
D = lm.LOOPCORE_CONFIGS["D: 3 chips, 8-bit, all resident"]
gpu = "h100"
st = lm.loopspec_stats(gpu, "1:2", lm.LOOPSPEC_LONG_RUNS)
hwg = lm.predict_gpu(lm.fit_gpu("l40s", m), gpu)
i8 = lm.measured_steps(gpu, "int8")
gpu_i8 = {B: depth * i8[(B, C)] * 1e-3 + lm.gpu_extras(m, hwg, gpu, B) for B in (1, 16)}
gpu_ls = st["ms_per_token"] * 1e-3
base = replace(D, kv_bytes=1.0)   # fp8 KV (passed Week 6)
WINDOW = 0   # --window N: shared designs also keep N own-loop tokens per later loop (sharefirst_w64, NOTEBOOK_CHIP_COST.md)

def footprint(design, B, P, G):
    f4 = lm.kv_footprint(m, replace(base, kv_caches=0), B, 1)   # bytes per token, 4 caches
    f1 = lm.kv_footprint(m, replace(base, kv_caches=1), B, 1)
    win = lm.kv_footprint(m, replace(base, kv_caches=1, kv_window=WINDOW), B, P + G) - f1 * (P + G)
    return {"per-loop": f4 * (P + G), "decode-only": f4 * P + f1 * G, "full": f1 * (P + G) + win}[design]

def chips_for(B, kv):
    w, _, _ = lm.fit_check(m, replace(base, n_chips=1))
    return max(1, math.ceil((w + kv) / base.sram_bytes_per_chip))

def hw_for(design, B, P, G, level, on_chip):
    kv = footprint(design, B, P, G)
    caches = {"per-loop": 0, "decode-only": 1, "full": 1}[design]   # KV writes per new token
    n = chips_for(B, kv) if on_chip else lm.sized(m, base, B, C).n_chips
    tl = lm.layer_latency_bottom_up(m, C, level, B=B)
    return replace(base, kv_caches=caches, kv_window=WINDOW if design == "full" else 0, kv_on_chip=on_chip, n_chips=n,
                   overhead_s=m.n_layers * tl), kv

print(f"H100 baselines at C={C}: int8 B=1 {gpu_i8[1]*1e3:.2f} ms, B=16 {gpu_i8[16]*1e3:.2f} ms; LoopSpec B=1 {gpu_ls*1e3:.2f} ms")
print(f"SRAM {base.sram_bytes_per_chip/1e6:.0f} MB/chip; weights+LM head int8 {lm.fit_check(m, replace(base, n_chips=1))[0]/1e6:.0f} MB\n")
for B in (1, 16):
    for G in (128, 256, 512, 1024, 1792, 2048):
        P = C - G
        for design in ("per-loop", "decode-only", "full"):
            for on_chip in (False, True):
                if design != "per-loop" and not on_chip:
                    continue
                row = []
                for level in ("install_int8_low", "install_int8_high"):
                    hw, kv = hw_for(design, B, P, G, level, on_chip)
                    t = lm.token_time(m, hw, B, C, depth)
                    if B == 1:
                        hw0 = replace(hw, overhead_s=0)
                        ls = [lm.chip_token_time_loopspec(m, hw0, C, st, ln, level) for ln in st["lanes"]]
                        row.append(f"b {gpu_ls/t:5.1f}x c {gpu_ls/ls[1]:5.1f}-{gpu_ls/ls[0]:5.1f}x")
                    else:
                        row.append(f"vs int8 {gpu_i8[16]/t:5.1f}x")
                print(f"B={B:<2} P={P:<5} G={G:<5} {design:11s} {'on-chip' if on_chip else 'HBM    '} KV {kv/1e6:6.0f} MB chips {hw.n_chips:>3} | low: {row[0]} | high: {row[1]}")
        print()


def sweep():
    """Per-layer latency across the sourced install range (20..40 cycles/install = 3.8..6.7 µs at B=1), eff 1.0/1.3,
    for per-loop caches on-chip vs as trained (HBM) and full sharing on-chip; speedup and speedup per chip."""
    designs = {"per-loop on-chip": ("per-loop", True), "per-loop HBM": ("per-loop", False), "full on-chip": ("full", True)}
    print("\nC=2048, q=0.5. (b) chip vs GPU+LoopSpec, (c) both LoopSpec (lanes lo-hi), B=16 vs H100 int8. /chip = speedup / chips")
    print(f"{'design':17s} {'cyc':>3} {'t_L B=1':>7} eff | chips B1 B16 | (b) B=1  /chip | (c) B=1       /chip     | B=16   /chip")
    for name, (design, on_chip) in designs.items():
        for cyc in (20, 25, 30, 35, 40):
            level = f"install_int8_c{cyc}"
            lm.LAYER_CYCLE_PARAMS[level] = dict(lm.LAYER_CYCLE_PARAMS["install_int8_low"], install_cycles=cyc)
            for eff in (1.0, 1.3):
                hw1, _ = hw_for(design, 1, C, 0, level, on_chip)
                hw16, _ = hw_for(design, 16, C, 0, level, on_chip)
                b = gpu_ls / (eff * lm.token_time(m, hw1, 1, C, depth))
                ls = [eff * lm.chip_token_time_loopspec(m, replace(hw1, overhead_s=0), C, st, ln, level) for ln in st["lanes"]]
                c_lo, c_hi = gpu_ls / ls[1], gpu_ls / ls[0]
                s16 = gpu_i8[16] / (eff * lm.token_time(m, hw16, 16, C, depth))
                n1, n16 = hw1.n_chips, hw16.n_chips
                print(f"{name:17s} {cyc:>3} {lm.layer_latency_bottom_up(m, C, level) * 1e6:5.2f}µs {eff:3.1f} | "
                      f"{n1:>6} {n16:>3} | {b:5.1f}x {b / n1:5.2f} | {c_lo:5.1f}-{c_hi:5.1f}x {c_lo / n1:5.2f}-{c_hi / n1:5.2f} | "
                      f"{s16:5.1f}x {s16 / n16:5.2f}")
        print()


LINKS = {"optimistic (0.30 µs, 1.344 TB/s)": (0.30e-6, 1.344e12), "pessimistic (0.722 µs, 235 GB/s)": (0.722e-6, 0.235e12),
         "1 µs hops (1.344 TB/s)": (1e-6, 1.344e12)}   # results/LATENCY_SOURCES.md, NOTEBOOK_KV_SHARING.md P1


def colocated_chips(design, B):
    """Chips for the co-located layout: whole layers, each with its KV, packed per chip, plus the LM head on a chip
    with room (else one more). None if one layer + its KV exceeds a chip."""
    _, kv = hw_for(design, B, C, 0, "install_int8_low", True)
    layer, sram = (m.core_params * base.weight_bytes + kv) / m.n_layers, base.sram_bytes_per_chip
    per_chip = int(sram // layer)
    if per_chip == 0:
        return None
    n = math.ceil(m.n_layers / per_chip)
    fewest = m.n_layers - (n - 1) * per_chip          # layers on the least-full chip
    return n if sram - fewest * layer >= m.lmhead_params * base.weight_bytes else n + 1


def placement_table():
    """P1: speedups with the KV placement term, sourced link parameters, t_L 3.83-6.68 µs, eff 1.0 / 1.3."""
    designs = [("per-loop, HBM", "per-loop", False, "aggregate"),
               ("per-loop, on-chip, near", "per-loop", True, "near"),
               ("per-loop, on-chip, stream", "per-loop", True, "stream"),
               ("per-loop, on-chip, co-located", "per-loop", True, "aggregate"),
               ("shared, on-chip, near", "full", True, "near"),
               ("shared, on-chip, stream", "full", True, "stream"),
               ("shared, on-chip, co-located", "full", True, "aggregate")]
    print("\nP1. C=2048, q=0.5. b = chip vs GPU+LoopSpec (B=1), c = both LoopSpec (lanes lo-hi), B=16 vs H100 int8. "
          "/chip = speedup / chips. 'n/a' = co-located layout does not fit (one layer's weights + KV > 500 MB).")
    for link, (hop, bw) in LINKS.items():
        print(f"\n== links: {link}")
        print(f"{'design':30s} {'t_L':>6} eff | chips B1 B16 | r B1  r B16 | (b) B=1 /chip | (c) B=1        /chip      | B=16  /chip")
        for name, design, on_chip, place in designs:
            for cyc in (20, 25, 30, 35, 40):
                level = f"install_int8_c{cyc}"
                lm.LAYER_CYCLE_PARAMS[level] = dict(lm.LAYER_CYCLE_PARAMS["install_int8_low"], install_cycles=cyc)
                for eff in (1.0, 1.3):
                    hws = {}
                    for B in (1, 16):
                        hw, _ = hw_for(design, B, C, 0, level, on_chip)
                        hws[B] = replace(hw, kv_placement=place, hop_latency_s=hop, link_bw=bw)
                    if place == "aggregate" and on_chip:      # co-located: chips from whole-layer packing
                        n_co = {B: colocated_chips(design, B) for B in (1, 16)}
                        hws = {B: replace(hws[B], n_chips=n_co[B] or hws[B].n_chips) for B in (1, 16)}
                        ok = {B: n_co[B] is not None for B in (1, 16)}
                    else:
                        ok = {1: True, 16: True}
                    hw1, hw16 = hws[1], hws[16]
                    r1, r16 = (lm.kv_remote_fraction(m, hws[B], B, C) if place != "aggregate" else 0.0 for B in (1, 16))
                    if ok[1]:
                        b = gpu_ls / (eff * lm.token_time(m, hw1, 1, C, depth))
                        ls = [eff * lm.chip_token_time_loopspec(m, replace(hw1, overhead_s=0), C, st, ln, level) for ln in st["lanes"]]
                        c_lo, c_hi = gpu_ls / ls[1], gpu_ls / ls[0]
                        n1 = hw1.n_chips
                        s1 = f"{b:5.1f}x {b / n1:4.2f} | {c_lo:5.1f}-{c_hi:5.1f}x {c_lo / n1:4.2f}-{c_hi / n1:4.2f}"
                    else:
                        s1 = f"{'n/a':>11} | {'n/a':>25}"
                    if ok[16]:
                        s16 = gpu_i8[16] / (eff * lm.token_time(m, hw16, 16, C, depth))
                        s2 = f"{s16:5.1f}x {s16 / hw16.n_chips:4.2f}"
                    else:
                        s2 = "  n/a"
                    print(f"{name:30s} {lm.layer_latency_bottom_up(m, C, level) * 1e6:4.2f}µ {eff:3.1f} | {hw1.n_chips:>6} {hw16.n_chips:>3} | "
                          f"{r1:4.2f} {r16:5.2f} | {s1} | {s2}")
            print()


def window_chips():
    """Chip counts of the shared designs with and without the own-loop window, at 1/16 sequences x 2048/8192."""
    global WINDOW
    print("\nShared KV: chips without / with a 64-token own-loop window (sized = near/stream; co-located = whole layers)")
    for B, Cx in ((1, 2048), (16, 2048), (1, 8192), (16, 8192)):
        row = []
        for WINDOW in (0, 64):
            kv = footprint("full", B, Cx, 0)
            layer = (m.core_params * base.weight_bytes + kv) / m.n_layers
            per = int(base.sram_bytes_per_chip // layer)
            n_co = None
            if per:
                n = math.ceil(m.n_layers / per)
                n_co = n if base.sram_bytes_per_chip - (m.n_layers - (n - 1) * per) * layer >= m.lmhead_params else n + 1
            row.append((kv, chips_for(B, kv), n_co))
        (k0, s0, c0), (k1, s1, c1) = row
        print(f"{B:>2}x{Cx:<5} KV {k0 / 1e6:7.0f} -> {k1 / 1e6:7.0f} MB (+{100 * (k1 / k0 - 1):.1f} %) | "
              f"sized {s0} -> {s1} | co-located {c0} -> {c1}")
    WINDOW = 0


if __name__ == "__main__":
    import sys
    if "--window" in sys.argv:
        WINDOW = int(sys.argv[sys.argv.index("--window") + 1])
        placement_table()
    elif "--window-chips" in sys.argv:
        window_chips()
    else:
        sweep()
        placement_table()
