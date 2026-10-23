"""Plot a harness CSV: latency vs. batch size and tokens/s vs. batch size.

Usage: python plot.py results/<timestamp>.csv   (PNGs are written next to the CSV)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (series color follows the target, never its rank).
COLORS = {"ut_step": "#2a78d6", "decode_step": "#eb6834"}
LABELS = {"ut_step": "one recurrent step", "decode_step": "full decode token"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, ylabel, title, batch_sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes, [str(b) for b in batch_sizes])
    ax.set_xlabel("batch size (sequences)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_metric(df, column, ylabel, title, path, band=False):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for (target, C), g in df.groupby(["target", "C"]):
        g = g.sort_values("B")
        color = COLORS.get(target, "#6250d6")
        label = f"{LABELS.get(target, target)} (C={C})"
        ax.plot(g["B"], g[column], color=color, linewidth=2, marker="o", markersize=5, label=label)
        if band:
            ax.fill_between(g["B"], g["p10_ms"], g["p90_ms"], color=color, alpha=0.18, linewidth=0)
        last = g.iloc[-1]
        ax.annotate(f"{last[column]:,.1f}", (last["B"], last[column]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8, color=INK)
    style(ax, ylabel, title, sorted(df["B"].unique()))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main(csv_path):
    df = pd.read_csv(csv_path)
    skipped = df[df["status"] != "ok"]
    if len(skipped):
        print("not plotted (status != ok):\n" + skipped[["target", "B", "C", "status"]].to_string(index=False))
    df = df[df["status"] == "ok"]
    stem = os.path.splitext(csv_path)[0]
    plot_metric(df, "median_ms", "latency (ms)",
                "Decode latency: median, band = p10–p90", f"{stem}_latency.png", band=True)
    plot_metric(df, "tokens_per_s", "tokens / s",
                "Throughput: batch size ÷ median latency", f"{stem}_tokens_per_s.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
