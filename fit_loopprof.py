"""Fit LoopProf step times to t = δ + γB + αC (the proposal's model) and to physically
motivated alternatives, with error bars, and draw the Week 4 plots.

Why more than one model: in one recurrent step each of the B sequences reads its own C tokens
of KV cache, so KV traffic scales with B·C, not C. We fit
    M1  t = δ + γ·B + α·C            (proposal)
    M2  t = δ + γ·B + α'·(B·C)       (KV traffic ∝ B·C)
    M3  t = δ + γ·B + α·C + α'·(B·C) (both)
and compare them by R², adjusted R², AIC and residuals.

Error bars:
  - OLS standard errors and 95 % CIs (t-distribution),
  - bootstrap over configs (resample (B, C) points; captures model misfit),
  - bootstrap over repeats (resample each config's raw per-rep times, refit on the new
    medians; captures measurement noise). Needs loopprof_raw.jsonl next to the CSV.

Usage: python fit_loopprof.py results/<dir>/loopprof.csv [--target ut] [--boot 2000]
Writes <dir>/fit_<target>.json, <dir>/fit_<target>.md and PNG plots next to the CSV.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

MODELS = {
    "M1: δ + γB + αC": ["1", "B", "C"],
    "M2: δ + γB + α'·BC": ["1", "B", "BC"],
    "M3: δ + γB + αC + α'·BC": ["1", "B", "C", "BC"],
}
PARAM = {"1": "δ (ms)", "B": "γ (ms per sequence)", "C": "α (ms per context token)",
         "BC": "α' (ms per sequence·context token)"}
T_975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
         10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00}


def t975(df):
    if df <= 0:
        return float("nan")
    keys = sorted(T_975)
    for k in keys:
        if df <= k:
            return T_975[k]
    return 1.96


def design(B, C, cols):
    feats = {"1": np.ones_like(B), "B": B, "C": C, "BC": B * C}
    return np.column_stack([feats[c] for c in cols])


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    rss = float(resid @ resid)
    df = n - k
    sigma2 = rss / df if df > 0 else float("nan")
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / df if df > 0 else float("nan")
    aic = n * math.log(rss / n) + 2 * k if rss > 0 else float("-inf")
    return beta, se, resid, dict(n=n, k=k, rss=rss, r2=r2, adj_r2=adj, aic=aic, df=df)


def load(csv_path, target):
    col = f"{target}_median_ms"
    rows = [r for r in csv.DictReader(open(csv_path)) if r["status"] == "ok" and r.get(col)]
    B = np.array([float(r["B"]) for r in rows])
    C = np.array([float(r["C"]) for r in rows])
    y = np.array([float(r[col]) for r in rows])
    return rows, B, C, y


def load_raw(csv_path, target):
    path = os.path.join(os.path.dirname(csv_path), "loopprof_raw.jsonl")
    key = {"ut": "ut_step", "decode": "decode_step"}[target]
    raw = {}
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            if key in d:
                raw[(float(d["B"]), float(d["C"]))] = np.array(d[key])
    return raw


def bootstrap(B, C, y, cols, n_boot, rng, raw=None):
    """Return (configs CI, repeats CI), each a [k, 2] array of 2.5/97.5 percentiles."""
    k = len(cols)
    by_cfg = np.empty((n_boot, k))
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        by_cfg[i] = np.linalg.lstsq(design(B[idx], C[idx], cols), y[idx], rcond=None)[0]
    ci_cfg = np.percentile(by_cfg, [2.5, 97.5], axis=0).T
    ci_rep = None
    if raw and all((b, c) in raw for b, c in zip(B, C)):
        X = design(B, C, cols)
        by_rep = np.empty((n_boot, k))
        for i in range(n_boot):
            yb = np.array([np.median(rng.choice(raw[(b, c)], len(raw[(b, c)]))) for b, c in zip(B, C)])
            by_rep[i] = np.linalg.lstsq(X, yb, rcond=None)[0]
        ci_rep = np.percentile(by_rep, [2.5, 97.5], axis=0).T
    return ci_cfg, ci_rep


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 0.1:
        return f"{v:.3f}"
    if a >= 1e-3:
        return f"{v:.5f}"
    return f"{v:.3e}"


def plots(rows, B, C, y, fits, out_prefix, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return []
    Bs, Cs = sorted(set(B)), sorted(set(C))
    files = []

    def grid(vals):
        g = np.full((len(Cs), len(Bs)), np.nan)
        for b, c, v in zip(B, C, vals):
            g[Cs.index(c), Bs.index(b)] = v
        return g

    def heat(vals, title, fname, cmap="viridis", center=False, fmt_s="{:.1f}"):
        g = grid(vals)
        fig, ax = plt.subplots(figsize=(7, 5))
        kw = {}
        if center:
            lim = np.nanmax(np.abs(g))
            kw = dict(vmin=-lim, vmax=lim)
        im = ax.imshow(g, origin="lower", aspect="auto", cmap=cmap, **kw)
        ax.set_xticks(range(len(Bs)), [int(b) for b in Bs])
        ax.set_yticks(range(len(Cs)), [int(c) for c in Cs])
        ax.set_xlabel("batch size B")
        ax.set_ylabel("context length C")
        ax.set_title(title)
        for i in range(len(Cs)):
            for j in range(len(Bs)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, fmt_s.format(g[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if not center else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = f"{out_prefix}_{fname}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(path)

    heat(y, f"measured {target}_step time (ms)", "heat_time")
    util = [float(r["util_measured"]) * 100 for r in rows]
    heat(util, "bandwidth utilization (% of measured copy peak)", "heat_util", cmap="magma")
    for name, f in fits.items():
        short = name.split(":")[0]
        heat(f["resid"], f"residual, {name} (ms)", f"resid_{short}", cmap="coolwarm", center=True,
             fmt_s="{:+.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in Cs:
        m = C == c
        o = np.argsort(B[m])
        axs[0].plot(B[m][o], y[m][o], "o-", label=f"C={int(c)}")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xlabel("batch size B")
    axs[0].set_ylabel(f"{target}_step time (ms)")
    axs[0].set_title("time vs batch size, per context length")
    axs[0].legend(fontsize=7)
    for b in Bs:
        m = B == b
        o = np.argsort(C[m])
        axs[1].plot(C[m][o], y[m][o], "o-", label=f"B={int(b)}")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel("context length C")
    axs[1].set_title("time vs context length, per batch size")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    path = f"{out_prefix}_lines.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--target", choices=["ut", "decode"], default="ut")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, B, C, y = load(args.csv, args.target)
    if len(y) < 5:
        raise SystemExit(f"only {len(y)} usable configs; need at least 5")
    raw = load_raw(args.csv, args.target)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    prefix = os.path.join(out_dir, f"fit_{args.target}")

    fits, report = {}, {"csv": args.csv, "target": f"{args.target}_step", "n_configs": len(y),
                        "B_values": sorted({int(b) for b in B}), "C_values": sorted({int(c) for c in C}),
                        "models": {}}
    for name, cols in MODELS.items():
        X = design(B, C, cols)
        beta, se, resid, st = ols(X, y)
        ci_cfg, ci_rep = bootstrap(B, C, y, cols, args.boot, rng, raw)
        tq = t975(st["df"])
        params = {}
        for j, c in enumerate(cols):
            params[PARAM[c]] = {
                "estimate": float(beta[j]), "se": float(se[j]),
                "ci95_ols": [float(beta[j] - tq * se[j]), float(beta[j] + tq * se[j])],
                "ci95_boot_configs": [float(x) for x in ci_cfg[j]],
                "ci95_boot_repeats": [float(x) for x in ci_rep[j]] if ci_rep is not None else None,
            }
        rel = np.abs(resid) / y
        fits[name] = {"resid": resid}
        report["models"][name] = {
            "params": params, **{k: v for k, v in st.items()},
            "max_abs_resid_ms": float(np.max(np.abs(resid))),
            "mean_abs_rel_resid_pct": float(100 * rel.mean()),
            "worst_config": {"B": int(B[np.argmax(np.abs(resid))]), "C": int(C[np.argmax(np.abs(resid))])},
        }
    # Lowest AIC wins, but a model within 2 AIC units of the minimum is statistically tied; among
    # tied models prefer the one with fewer parameters (MODELS is ordered simplest-first per k).
    aics = {k: v["aic"] for k, v in report["models"].items()}
    a_min = min(aics.values())
    tied = [k for k in report["models"] if aics[k] - a_min < 2]
    best = min(tied, key=lambda k: (report["models"][k]["k"], aics[k]))
    report["best_by_aic"] = best
    report["aic_delta"] = {k: round(v - a_min, 2) for k, v in aics.items()}

    # Flat-region check: how much does time grow from the smallest B to each B, at fixed C?
    flat = {}
    for c in sorted(set(C)):
        m = C == c
        if m.sum() < 2:
            continue
        b0 = B[m].min()
        t0 = y[m][B[m] == b0][0]
        flat[int(c)] = {int(b): round(float(t / t0), 3) for b, t in sorted(zip(B[m], y[m]))}
    report["growth_vs_smallest_B"] = flat
    util = np.array([float(r["util_measured"]) for r in rows])
    report["bandwidth_util_measured_peak"] = {"min": float(util.min()), "median": float(np.median(util)),
                                              "max": float(util.max())}

    with open(f"{prefix}.json", "w") as f:
        json.dump(report, f, indent=1)

    lines = [f"# LoopProf fit: {args.target}_step time", "",
             f"{len(y)} configs; B in {report['B_values']}; C in {report['C_values']}.", ""]
    for name, mr in report["models"].items():
        tag = "  ← preferred (lowest AIC, or simplest within ΔAIC < 2)" if name == best else ""
        lines += [f"## {name}{tag}", "",
                  f"R² = {mr['r2']:.4f}, adjusted R² = {mr['adj_r2']:.4f}, AIC = {mr['aic']:.1f} "
                  f"(Δ = {report['aic_delta'][name]:.1f}), "
                  f"max |residual| = {mr['max_abs_resid_ms']:.2f} ms (at B={mr['worst_config']['B']}, "
                  f"C={mr['worst_config']['C']}), mean |residual| = {mr['mean_abs_rel_resid_pct']:.1f} %", "",
                  "| parameter | estimate | ± SE | 95 % CI (OLS) | 95 % CI (bootstrap, configs) | 95 % CI (bootstrap, repeats) |",
                  "|---|---:|---:|---|---|---|"]
        for pname, p in mr["params"].items():
            rep = p["ci95_boot_repeats"]
            lines.append(f"| {pname} | {fmt(p['estimate'])} | {fmt(p['se'])} | "
                         f"[{fmt(p['ci95_ols'][0])}, {fmt(p['ci95_ols'][1])}] | "
                         f"[{fmt(p['ci95_boot_configs'][0])}, {fmt(p['ci95_boot_configs'][1])}] | "
                         + (f"[{fmt(rep[0])}, {fmt(rep[1])}]" if rep else "n/a") + " |")
        lines.append("")
    u = report["bandwidth_util_measured_peak"]
    lines += ["## Bandwidth utilization (bytes model ÷ measured copy peak)", "",
              f"min {100 * u['min']:.1f} %, median {100 * u['median']:.1f} %, max {100 * u['max']:.1f} %", "",
              "## Time relative to the smallest batch at the same context length", "",
              "| C | " + " | ".join(f"B={b}" for b in report["B_values"]) + " |",
              "|---:|" + "---:|" * len(report["B_values"])]
    for c, d in flat.items():
        lines.append(f"| {c} | " + " | ".join(f"{d[b]:.2f}" if b in d else "" for b in report["B_values"]) + " |")
    files = plots(rows, B, C, y, fits, prefix, args.target)
    lines += ["", "Plots: " + ", ".join(os.path.basename(p) for p in files)]
    with open(f"{prefix}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {prefix}.json, {prefix}.md")


if __name__ == "__main__":
    main()
