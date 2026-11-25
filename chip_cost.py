"""results/NOTEBOOK_CHIP_COST.md: the per-chip table re-expressed per mm² of silicon, per watt and per cost proxy vs H100.

Reads the committed per-chip table (results/kv_placement_p1.txt, sourced links: optimistic and pessimistic blocks) and
divides each design's speedup by n_chips x (LoopCore chip / H100) for area, power and cost. The chip ratios are ranges
(RATIOS below; each bound and its source in the notebook). A metric's range = lowest speedup / highest ratio .. highest
speedup / lowest ratio. Proposed numbers only; loopcore_model.py is not used or changed.
    python3 chip_cost.py [table]
"""
import itertools
import math
import re
import sys
from collections import defaultdict

TABLE = "results/kv_placement_p1.txt"
SOURCED = ("optimistic", "pessimistic")

# ---------------------------------------------------------------- inputs (each source in NOTEBOOK_CHIP_COST.md, C1)
# Labels: Q = quoted from an opened source, D = derived, M = from memory, unchecked, U = unverified search snippet.
H100 = dict(area=814, power=700, fp8_tflops=1979, hbm_gb=80)            # Q: NVIDIA Hopper blog Table 3; product page
LC = dict(sram_mb=500, fp8_tflops=1200, links=96, link_gbps=112, sram_tbps=150)   # Q: NVIDIA LPX blog (Groq 3 LPU)
SRAM_MB_PER_MM2 = (1.56, 3.8)     # usable SRAM: Q/D AMD V-Cache N7 64 MB / 41 mm² (no 4 nm credit) .. D dense N5 macro
                                  # (N3 ~4.3 MB/mm² scaled by the N3B/N5 cell); Zen 5 L3 (N4P) is 2.1 (Q/D)
TENSOR_FRACTION_H100 = (0.25, 0.5)  # M: share of the H100 die that is tensor-core datapath (no source found)
SERDES_MM2_PER_LANE = (0.28, 0.63)  # U: 112G lanes, 5 nm .. 7 nm (snippets of ISSCC 2023 Park et al., 7 nm DSP designs)
OTHER_FRACTION = (0.10, 0.20)     # M: vector units, control, NoC, PCIe, pads, as a share on top
HBM_PHY_MM2 = (15, 30)            # M: one HBM3 PHY + controller (per-loop HBM design only)
SRAM_PJ_PER_BIT = (0.12, 0.22)    # Q: Jouppi ISCA 2021 Table 2, 7 nm, 8 KB .. 1 MB arrays (per-bit D)
MAC_PJ = 0.077                    # Q/D: Jouppi Table 2, 7 nm int8 mult 0.07 + add 0.007 (no FP8 figure exists)
SERDES_PJ_PER_BIT = (4.5, 4.63)   # Q (abstract/title): ISSCC 2022 Guo; ISSCC 2023 Park, 112G, 5 nm
POWER_OVERHEAD = (1.3, 1.6)       # M: clocking, control, leakage, voltage regulation on top of the datapath sum
HBM_STACK_W = (21, 32)            # D: 3.9 pJ/bit HBM2 (O'Connor MICRO-50) at H100 per-stack rate .. U: 32 W HBM3E
WAFER_USD = (16_000, 18_500)      # Q: Epoch AI (N5 P5 .. N4 median), CSET $16,988; Samsung SF4 price not published
D0_PER_CM2 = (0.07, 0.10)         # U: TSMC N5 0.10-0.11 (AnandTech relay of TSMC slide) .. M: mature 0.07
HBM_USD_PER_GB = (12, 17)         # Q: Epoch AI B200 $14-17 (HBM3E); U: ~$12 HBM3
COWOS_USD = (600, 900)            # D: CoWoS-S $12k per interposer wafer (Epoch) / ~15 H100-size interposers
PACKAGE_USD = (100, 300)          # M: organic flip-chip package + test, no interposer (LoopCore on-chip designs)
ASSEMBLY_LOSS = (0.10, 0.25)      # D: Epoch B200: packaging yield loss + auxiliary = 23 % of the total


def dies_per_wafer(a):
    return math.pi * 150 ** 2 / a - math.pi * 300 / math.sqrt(2 * a)


def die_usd(a, wafer, d0):
    """Murphy yield; harvesting and SRAM redundancy make real yield higher for both chips (same model for both)."""
    ad = a / 100 * d0
    return wafer / (dies_per_wafer(a) * ((1 - math.exp(-ad)) / ad) ** 2)


def loopcore_area():
    sram = (LC["sram_mb"] / SRAM_MB_PER_MM2[1], LC["sram_mb"] / SRAM_MB_PER_MM2[0])
    tf_per_mm2 = [H100["fp8_tflops"] / (f * H100["area"]) for f in TENSOR_FRACTION_H100]   # ~9.7 .. 4.9
    compute = (LC["fp8_tflops"] / max(tf_per_mm2), LC["fp8_tflops"] / min(tf_per_mm2))
    serdes = tuple(LC["links"] * s for s in SERDES_MM2_PER_LANE)
    return tuple((s + c + io) * (1 + o) for s, c, io, o in zip(sram, compute, serdes, OTHER_FRACTION)), \
        dict(sram=sram, compute=compute, serdes=serdes)


def loopcore_power():
    sram = tuple(LC["sram_tbps"] * 8e12 * e * 1e-12 for e in SRAM_PJ_PER_BIT)
    mac = LC["fp8_tflops"] * 1e12 / 2 * MAC_PJ * 1e-12
    # a transceiver's pJ/b covers its TX + RX per bit of line rate; each link is one 112G lane pair
    serdes = tuple(LC["links"] * LC["link_gbps"] * 1e9 * e * 1e-12 for e in SERDES_PJ_PER_BIT)
    return tuple((s + mac + io) * o for s, io, o in zip(sram, serdes, POWER_OVERHEAD)), dict(sram=sram, mac=mac, serdes=serdes)


def cost_corners(area):
    """LoopCore / H100 cost ratios over every corner of the cost inputs, the shared ones (wafer price, defect density,
    HBM price, CoWoS, assembly loss) set the same for both chips. Returns {design: [(ratio, lc_usd, h100_usd), ...]}."""
    out = {"cost": [], "cost_hbm": []}
    for a, phy, pkg in zip(area, HBM_PHY_MM2, PACKAGE_USD):          # LoopCore's own inputs: low together, high together
        for w, d0, g, cw, loss in itertools.product(WAFER_USD, D0_PER_CM2, HBM_USD_PER_GB, COWOS_USD, ASSEMBLY_LOSS):
            h100 = (die_usd(H100["area"], w, d0) + H100["hbm_gb"] * g + cw) * (1 + loss)
            lc = (die_usd(a, w, d0) + pkg) * (1 + loss)
            lc_hbm = (die_usd(a + phy, w, d0) + 16 * g + cw) * (1 + loss)
            out["cost"].append((lc / h100, lc, h100))
            out["cost_hbm"].append((lc_hbm / h100, lc_hbm, h100))
    return out


def ratios():
    """(low, high) of LoopCore / H100 per metric; "_hbm" = the per-loop HBM design."""
    area, _ = loopcore_area()
    power, _ = loopcore_power()
    corners = cost_corners(area)
    rng = lambda xs: (min(xs), max(xs))
    return {"area": (area[0] / H100["area"], area[1] / H100["area"]),
            "area_hbm": ((area[0] + HBM_PHY_MM2[0]) / H100["area"], (area[1] + HBM_PHY_MM2[1]) / H100["area"]),
            "power": (power[0] / H100["power"], power[1] / H100["power"]),
            "power_hbm": ((power[0] + HBM_STACK_W[0]) / H100["power"], (power[1] + HBM_STACK_W[1]) / H100["power"]),
            **{k: rng([r for r, _, _ in v]) for k, v in corners.items()}}


def calibration():
    """Same area method on Graphcore GC200 (TSMC N7, 823 mm², 900 MB, ~250 TFLOPS FP16: Q, The Register; datasheet D).
    N7 SRAM 1.56-2.5 MB/mm² (V-Cache .. agent's dense-macro estimate); FP16 counted as 2x FP8 area (M); N7 logic 1.6x
    less dense than 4N (M); IPU-Links ~20 mm² (M)."""
    sram = (900 / 2.5, 900 / 1.56)
    tf_per_mm2 = [H100["fp8_tflops"] / (f * H100["area"]) / 1.6 for f in TENSOR_FRACTION_H100]
    compute = (500 / max(tf_per_mm2), 500 / min(tf_per_mm2))
    return tuple((s + c + 20) * (1 + o) for s, c, o in zip(sram, compute, OTHER_FRACTION))


def parse(path=TABLE):
    """{design: {"chips": (n1, n16), "b": [..], "c": [..], "b16": [..]}} over the sourced link blocks."""
    rows, block = defaultdict(lambda: {"b": [], "c": [], "b16": []}), None
    num = r"\s*([\d.]+)x"
    for line in open(path, encoding="utf-8"):
        if line.startswith("== links:"):
            block = line.split(":", 1)[1].split()[0]
            continue
        if block not in SOURCED or "|" not in line or line.startswith("design"):
            continue
        cells = [c.strip() for c in line.split("|")]
        name = re.match(r"(.+?)\s+[\d.]+µ", cells[0]).group(1).strip()
        n1, n16 = map(int, cells[1].split())
        d = rows[name]
        d["chips"] = (n1, n16)
        if "n/a" not in cells[3]:
            d["b"].append(float(re.match(num, cells[3]).group(1)))
            lo, hi = re.match(r"\s*([\d.]+)-\s*([\d.]+)x", cells[4]).groups()
            d["c"] += [float(lo), float(hi)]
        if "n/a" not in cells[5]:
            d["b16"].append(float(re.match(num, cells[5]).group(1)))
    return dict(rows)


def normalised(speeds, n, ratio):
    """(low, high) of speedup / (n x ratio)."""
    return min(speeds) / (n * ratio[1]), max(speeds) / (n * ratio[0])


def verdict(lo, hi):
    return "above 1" if lo > 1 else "below 1" if hi < 1 else "straddles"


def report(rows, ratios):
    cols = [("b", 0, "(b) B=1"), ("c", 0, "(c) B=1"), ("b16", 1, "B=16")]
    for metric in ("chip", "area", "power", "cost"):
        print(f"\n## speedup per {metric} vs H100" + ("" if metric == "chip" else
              f" (LoopCore/H100 ratio {ratios[metric][0]:.2f}-{ratios[metric][1]:.2f}; HBM design "
              f"{ratios[metric + '_hbm'][0]:.2f}-{ratios[metric + '_hbm'][1]:.2f})"))
        print("| design | chips 1x / 16x | " + " | ".join(c[2] for c in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for name, d in rows.items():
            r = (1.0, 1.0) if metric == "chip" else ratios[metric + ("_hbm" if "HBM" in name else "")]
            cells = []
            for key, i, _ in cols:
                if not d[key]:
                    cells.append("does not fit")
                    continue
                lo, hi = normalised(d[key], d["chips"][i], r)
                cells.append(f"{lo:.2f}-{hi:.2f} ({verdict(lo, hi)})")
            print(f"| {name} | {d['chips'][0]} / {d['chips'][1]} | " + " | ".join(cells) + " |")


def estimate():
    area, a = loopcore_area()
    power, p = loopcore_power()
    corners = cost_corners(area)
    usd = lambda k, i: (min(c[i] for c in corners[k]), max(c[i] for c in corners[k]))
    f = lambda t, u="": f"{t[0]:.0f}-{t[1]:.0f}{u}"
    print("# LoopCore chip estimate (inputs and labels at the top of chip_cost.py)")
    print(f"area {f(area, ' mm²')} = (SRAM {f(a['sram'])} + compute {f(a['compute'])} + SerDes {f(a['serdes'])}) x "
          f"(1 + other {OTHER_FRACTION[0]:.2f}-{OTHER_FRACTION[1]:.2f}); MB/mm² {LC['sram_mb'] / area[1]:.2f}-{LC['sram_mb'] / area[0]:.2f}")
    print(f"power {f(power, ' W')} = (SRAM {f(p['sram'])} + MACs {p['mac']:.0f} + SerDes {f(p['serdes'])}) x "
          f"{POWER_OVERHEAD[0]}-{POWER_OVERHEAD[1]} (full load)")
    print(f"cost: LoopCore {f(usd('cost', 1), ' USD')}; LoopCore + 1 HBM stack {f(usd('cost_hbm', 1), ' USD')}; "
          f"H100 {f(usd('cost', 2), ' USD')} (die + 80 GB HBM + CoWoS + assembly)")
    print(f"calibration, GC200: predicted {f(calibration(), ' mm²')} vs actual 823 mm²")
    for k, v in ratios().items():
        print(f"ratio {k:9s} {v[0]:.2f}-{v[1]:.2f}")


if __name__ == "__main__":
    estimate()
    report(parse(sys.argv[1] if len(sys.argv) > 1 else TABLE), ratios())
