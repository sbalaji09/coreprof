"""Plot a bench_cdb.py run: throughput vs slots (baseline vs CDB, one panel per q) and
CDB's achieved speedup as % of ideal vs slots.

Usage: python plot_bench.py results/bench_cdb_<ts>   (prefix; PNGs are written next to it)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot import GRID, INK, MUTED, style

# Fixed categorical order: color follows the batcher / definition, never its rank.
COLORS = {"baseline": "#2a78d6", "cdb": "#eb6834"}
LABELS = {"baseline": "baseline (depth-synchronous)", "cdb": "CDB"}
PCT = {"pct_ideal_full": ("#2a78d6", "vs full-batch call time (q=1 control)"),
       "pct_ideal_own": ("#eb6834", "vs CDB's own call time (= utilization)"),
       "pct_paper_bound": ("#1baf7a", "paper-style: vs no-exit baseline, bound 4/mean depth")}


def throughput(points, prefix):
    qs = sorted(points.q.unique())
    rows = [("steady_row_loops_per_s", "useful row-loops/s"), ("steady_tokens_per_s", "tokens/s")]
    fig, axes = plt.subplots(len(rows), len(qs), figsize=(4.2 * len(qs), 3.6 * len(rows)), dpi=150, squeeze=False)
    for r, (col, ylabel) in enumerate(rows):
        for c, q in enumerate(qs):
            ax = axes[r][c]
            for name in ("baseline", "cdb"):
                g = points[(points.q == q) & (points.batcher == name)].sort_values("S")
                ax.plot(g.S, g[f"{col}_median"], color=COLORS[name], linewidth=2, marker="o", markersize=5,
                        label=LABELS[name])
                ax.fill_between(g.S, g[f"{col}_min"], g[f"{col}_max"], color=COLORS[name], alpha=0.18, linewidth=0)
                last = g.iloc[-1]
                ax.annotate(f"{last[f'{col}_median']:,.0f}", (last.S, last[f"{col}_median"]),
                            textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=INK)
            title = f"q = {q:g}" + (" (control: no early exit)" if q >= 1 else "")
            style(ax, ylabel if c == 0 else "", title if r == 0 else "", sorted(points.S.unique()))
            ax.set_xlabel("max_slots" if r == len(rows) - 1 else "", color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Steady-state decode throughput (bf16, prefill excluded; band = min–max of repeats)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = f"{prefix}_throughput.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def pct_ideal(speedup, prefix):
    qs = [q for q in sorted(speedup.q.unique()) if q < 1]
    fig, axes = plt.subplots(1, len(qs), figsize=(5 * len(qs), 3.8), dpi=150, squeeze=False)
    for c, q in enumerate(qs):
        ax = axes[0][c]
        g = speedup[speedup.q == q].sort_values("S")
        for col, (color, label) in PCT.items():
            ax.plot(g.S, g[f"{col}_median"], color=color, linewidth=2, marker="o", markersize=5, label=label)
            ax.fill_between(g.S, g[f"{col}_min"], g[f"{col}_max"], color=color, alpha=0.18, linewidth=0)
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=":")
        style(ax, "% of ideal speedup" if c == 0 else "", f"q = {q:g}", sorted(speedup.S.unique()))
        ax.set_xlabel("max_slots", color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=7, labelcolor=INK, loc="lower left")
    fig.suptitle("CDB's achieved speedup as % of ideal (no slot ever idle)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = f"{prefix}_pct_ideal.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main(prefix):
    prefix = prefix.removesuffix("_points.csv").removesuffix("_speedup.csv").removesuffix("_runs.csv")
    throughput(pd.read_csv(f"{prefix}_points.csv"), prefix)
    pct_ideal(pd.read_csv(f"{prefix}_speedup.csv"), prefix)


if __name__ == "__main__":
    main(sys.argv[1])
