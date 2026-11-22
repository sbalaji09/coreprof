"""Check the analytical model's predictions against the best measured GPU step (stdlib only, runs on a laptop).

Target = the GPU "envelope": at each (B, C) the fastest bf16 step measured: fused (torch.compile + CUDA graph),
fused + coordinate_descent_tuning, or uncompiled CUDA graph (see CALIBRATION.md, decision 4).

Usage:
  python calibrate.py [--gpu l40s|h100]            # measured targets, predictions pending
  python calibrate.py --gpu h100 --pred calibration_predictions_h100.json [--tol 10] [--write]
  python calibrate.py --gpu l40s --template        # (re)write the predictions template, keeping filled values
  python calibrate.py --selftest

Two tests, both must pass:
  1. six summary quantities, each within --tol percent (error is 0 inside a measured range);
  2. step time at EVERY grid point ("grid": {"BxC": ms}): median |error| <= tol and max |error| <= 2 x tol.
Measured values are recomputed from the raw run files every time, never copied by hand.
--write replaces this GPU's block between the calibration markers in CALIBRATION.md.
Exit code: 0 only if both tests pass.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# gpu -> device name, template path, and the runs whose per-point minimum is the target
RUNS = {
    "l40s": ("NVIDIA L40S", "calibration_predictions.json",
             ["results/modal/delta_compile-l40s-20261116-214809", "results/modal/delta_grid-l40s-20261116-184352",
              "results/modal/delta_bf16_cd-l40s-20261120-010859"]),
    "h100": ("NVIDIA H100 80GB HBM3", "calibration_predictions_h100.json",
             ["results/modal/delta_compile-h100-20261116-214809", "results/modal/delta_grid-h100-20261116-184347",
              "results/modal/delta_bf16_cd-h100-20261120-010911"]),
}
markers = lambda gpu: (f"<!-- calibration:{gpu}:begin -->", f"<!-- calibration:{gpu}:end -->")

# key -> (label, unit, source); order = table order
ROWS = {
    "weight_ms": ("weight streaming time per loop step", "ms",
                  "fastest weight GEMV/GEMM kernels of either implementation, 1x128 and 16x2048"),
    "kv_us_per_token": ("KV cost per cached token", "µs", "M2 fit over the envelope"),
    "crossover_tokens": ("weight/KV crossover", "cached tokens", "weight_ms / kv_us_per_token"),
    "step_ms_b1_c128": ("best GPU step time, B=1, C=128", "ms", "envelope"),
    "delta_ms": ("fixed cost δ (M2 fit)", "ms", "M2 fit over the envelope"),
    "overhead_ms": ("non-weight GPU time (attention + small kernels + gaps), B=1, C=128", "ms",
                    "step − weight kernels of the fastest implementation"),
}


def fit_m2(rows):
    """OLS of t = δ + γ·B + α′·(B·C). Returns (δ, γ, α′ in ms per token, R²)."""
    X = [(1.0, r["B"], r["B"] * r["C"]) for r in rows]
    y = [r["t"] for r in rows]
    A = [[sum(x[i] * x[j] for x in X) for j in range(3)] + [sum(x[i] * t for x, t in zip(X, y))] for i in range(3)]
    for i in range(3):  # Gauss-Jordan on the 3x3 normal equations
        for j in range(3):
            if j != i:
                f = A[j][i] / A[i][i]
                A[j] = [a - f * b for a, b in zip(A[j], A[i])]
    b = [A[i][3] / A[i][i] for i in range(3)]
    pred = [sum(bi * xi for bi, xi in zip(b, x)) for x in X]
    ybar = sum(y) / len(y)
    r2 = 1 - sum((t - p) ** 2 for t, p in zip(y, pred)) / sum((t - ybar) ** 2 for t in y)
    return b[0], b[1], b[2], r2


def weight_kernel_ms(bd):
    """Weight GEMV/GEMM time of one breakdown row. Runs after fc7ed97 have an attention category, so
    gemm_ms is weights only; older runs counted flash attention as gemm, so use the top weight kernel.
    Breakdowns before kernel_cats_version 2 (2026-11-22) filed o_proj as "other" when inductor lowered it as bmm
    (triton_red_fused_bmm_N, coordinate-descent runs): add those kernels back from top_kernels_ms."""
    if "attention_ms" in bd:
        if bd.get("kernel_cats_version", 1) >= 2:
            return bd["gemm_ms"]
        return bd["gemm_ms"] + sum(v for k, v in bd["top_kernels_ms"].items() if "fused_bmm_" in k.lower())
    return next(v for k, v in bd["top_kernels_ms"].items()
                if ("gemvx" in k or "gemm" in k) and "flash" not in k)


def load_run(run):
    d = ROOT / run
    rows = list(csv.DictReader(open(d / "graph_grid.csv")))
    js = json.load(open(d / "delta_split.json"))
    impl = js.get("grid_impl", "graph_static")
    kb = js["kernel_breakdown"]
    w = {bc: weight_kernel_ms(kb[f"{impl} {bc}"]) for bc in ("1x128", "16x2048")}
    label = impl + ("+cd" if (js.get("args") or {}).get("coord_descent") else "")   # same impl name, different runs
    return {(int(r["B"]), int(r["C"])): float(r["median_ms"]) for r in rows}, label, w


def envelope(gpu):
    """{(B, C): (ms, impl)} = fastest implementation per point, and per-impl weight times."""
    env, weights = {}, {}
    for run in RUNS[gpu][2]:
        grid, impl, w = load_run(run)
        weights[impl] = w
        for bc, t in grid.items():
            if bc not in env or t < env[bc][0]:
                env[bc] = (t, impl)
    return env, weights


def measured(gpu="l40s"):
    """Summary quantities, each (lo, hi); lo == hi for point values. Plus the envelope grid."""
    env, weights = envelope(gpu)
    rows = [{"B": B, "C": C, "t": t} for (B, C), (t, _) in env.items()]
    delta, _, alpha, r2 = fit_m2(rows)
    assert r2 > 0.99, f"{gpu} envelope fit changed: R²={r2}"
    w1 = min(w["1x128"] for w in weights.values())
    w16 = min(w["16x2048"] for w in weights.values())
    step, win = env[(1, 128)]
    return {
        "weight_ms": (min(w1, w16), max(w1, w16)),
        "kv_us_per_token": (alpha * 1e3,) * 2,
        "crossover_tokens": (min(w1, w16) / alpha, max(w1, w16) / alpha),
        "step_ms_b1_c128": (step,) * 2,
        "delta_ms": (delta,) * 2,
        "overhead_ms": (step - weights[win]["1x128"],) * 2,
    }, env


def error_pct(pred, lo, hi):
    """0 inside the measured range, else % distance to the nearest end."""
    if lo <= pred <= hi:
        return 0.0
    ref = lo if pred < lo else hi
    return 100 * (pred - ref) / ref


def fmt(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def grid_errors(env, pred):
    """{(B, C): error %} for every predicted grid point."""
    g = pred.get("grid") or {}
    out = {}
    for (B, C), (t, _) in env.items():
        p = g.get(f"{B}x{C}")
        if p is not None:
            out[(B, C)] = 100 * (p - t) / t
    return out


def report(meas, env, pred, tol, gpu="l40s"):
    """Returns (markdown block, all_ok)."""
    name, tmpl, runs = RUNS[gpu]
    lines = [f"| quantity | measured ({name}, best GPU) | source | model prediction | error % |", "|---|---|---|---|---|"]
    ok = True
    for key, (label, unit, src) in ROWS.items():
        lo, hi = meas[key]
        m = f"{fmt(lo)} {unit}" if lo == hi else f"{fmt(lo)}–{fmt(hi)} {unit}"
        p = (pred.get("predictions") or {}).get(key)
        if p is None:
            ok = False
            lines.append(f"| {label} | {m} | {src} | pending | pending |")
            continue
        e = error_pct(p, lo, hi)
        ok &= abs(e) <= tol
        lines.append(f"| {label} | {m} | {src} | {fmt(p)} {unit} | {e:+.1f}{'' if abs(e) <= tol else ' ✗'} |")
    n = sum((pred.get("predictions") or {}).get(k) is not None for k in ROWS)

    ge = grid_errors(env, pred)
    wins = {}
    for _, impl in env.values():
        wins[impl] = wins.get(impl, 0) + 1
    if len(ge) < len(env):
        ok = False
        grid_line = f"pending ({len(ge)}/{len(env)} grid points predicted)"
    else:
        med = statistics.median(abs(e) for e in ge.values())
        (wB, wC), we = max(ge.items(), key=lambda kv: abs(kv[1]))
        g_ok = med <= tol and abs(we) <= 2 * tol
        ok &= g_ok
        grid_line = (f"median |error| {med:.1f} %, max {we:+.1f} % at B={wB}, C={wC} "
                     f"({'pass' if g_ok else '**fail**'}: needs median ≤ {tol:g} %, max ≤ {2 * tol:g} %)")
    if ok:
        status = f"Calibrated on {name}: {pred.get('date', '?')} (summary within ±{tol:g} %, grid passes)."
    else:
        status = f"Calibrated on {name}: **no** ({n}/{len(ROWS)} summary predictions, {len(ge)}/{len(env)} grid points)."
    inputs = pred.get("inputs") or {}
    inp = "\n".join(f"- {k}: {'pending' if v is None else v}" for k, v in inputs.items()) or "- pending"
    body = (f"**Status:** {status}\n\n"
            f"Measured: the fastest of {', '.join(f'`{r}/`' for r in runs)} at each point "
            f"(wins: {', '.join(f'{k} {v}' for k, v in sorted(wins.items()))} of {len(env)}). "
            f"Predictions: `{pred.get('_path', tmpl)}`"
            f"{', from ' + pred['source'] if pred.get('source') else ''}. "
            f"Error % is 0 inside a measured range, else the distance to its nearest end.\n\n"
            + "\n".join(lines) + f"\n\n**Step time over the whole grid ({len(env)} points):** {grid_line}\n\n"
            f"**Model inputs:**\n\n{inp}")
    begin, end = markers(gpu)
    return f"{begin}\n{body}\n{end}", ok


def write_template(gpu, env):
    """Predictions template with every grid point; keeps values already filled in."""
    path = ROOT / RUNS[gpu][1]
    old = json.load(open(path)) if path.exists() else {}
    grid = old.get("grid") or {}
    tmpl = {
        "date": old.get("date"), "source": old.get("source"),
        "inputs": old.get("inputs") or {"weight_bytes": None, "weight_bandwidth_GBps": None,
                                         "kv_bytes_per_token": None, "kv_bandwidth_GBps": None,
                                         "overhead_term": None},
        "predictions": {k: (old.get("predictions") or {}).get(k) for k in ROWS},
        "grid": {f"{B}x{C}": grid.get(f"{B}x{C}") for (B, C) in sorted(env, key=lambda bc: (bc[1], bc[0]))},
    }
    path.write_text(json.dumps(tmpl, indent=2) + "\n")
    print(f"wrote {path.name}: {len(tmpl['grid'])} grid points")


def selftest():
    for gpu in RUNS:
        meas, env = measured(gpu)
        exact = {"predictions": {k: v[0] for k, v in meas.items()}, "date": "x",
                 "grid": {f"{B}x{C}": t for (B, C), (t, _) in env.items()}}
        assert report(meas, env, exact, 10, gpu)[1]
        assert not report(meas, env, {}, 10, gpu)[1]
        assert not report(meas, env, {**exact, "grid": {}}, 10, gpu)[1]           # grid required
        off = {**exact, "predictions": {**exact["predictions"], "delta_ms": meas["delta_ms"][0] * 1.2}}
        assert not report(meas, env, off, 10, gpu)[1] and report(meas, env, off, 25, gpu)[1]
        one_bad = {**exact, "grid": {**exact["grid"], "1x128": env[(1, 128)][0] * 1.25}}
        assert not report(meas, env, one_bad, 10, gpu)[1]                         # max > 2 x tol fails
        # the M2 formula fitted to the envelope is itself a (data-fitted) model: how would it score?
        _, d, g, a = (None, *fit_m2([{"B": B, "C": C, "t": t} for (B, C), (t, _) in env.items()])[:3])
        m2 = {**exact, "grid": {f"{B}x{C}": d + g * B + a * B * C for (B, C) in env}}
        print(f"{gpu}: envelope {len(env)} points, wins {sorted({i: sum(j == i for _, j in env.values()) for _, i in env.values()}.items())}; "
              f"weight {meas['weight_ms'][0]:.2f}-{meas['weight_ms'][1]:.2f} ms, step {meas['step_ms_b1_c128'][0]:.2f} ms, "
              f"δ {meas['delta_ms'][0]:.2f} ms, α′ {meas['kv_us_per_token'][0]:.4f} µs, overhead {meas['overhead_ms'][0]:.2f} ms; "
              f"M2-fit-as-model grid: {report(meas, env, m2, 10, gpu)[0].split(' points):** ')[1].splitlines()[0]}")
    bmm = {"gemm_ms": 3.0, "attention_ms": 0.2, "top_kernels_ms": {"triton_red_fused_bmm_3": 0.3,
                                                                  "triton_red_fused__to_copy_add_bmm_mean_4": 0.05}}
    assert abs(weight_kernel_ms(bmm) - 3.3) < 1e-9                                     # o_proj re-sorted, norm not
    assert weight_kernel_ms({**bmm, "kernel_cats_version": 2}) == 3.0                   # already sorted at run time
    assert error_pct(3.9, 3.77, 3.98) == 0 and round(error_pct(2.85, 3.77, 3.98)) == -24  # datasheet case
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=list(RUNS), default="l40s")
    ap.add_argument("--pred", help="JSON: {date, source, inputs, predictions: {key: number|null}, grid: {BxC: ms|null}}")
    ap.add_argument("--tol", type=float, default=10.0, help="summary tolerance and grid median tolerance, %% (default 10)")
    ap.add_argument("--write", action="store_true", help="update this GPU's calibration block in CALIBRATION.md")
    ap.add_argument("--template", action="store_true", help="(re)write this GPU's predictions template")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    meas, env = measured(a.gpu)
    if a.template:
        return write_template(a.gpu, env)
    pred = {}
    if a.pred:
        pred = json.load(open(a.pred))
        pred["_path"] = a.pred
        unknown = set(pred.get("predictions") or {}) - set(ROWS)
        bad_pts = set(pred.get("grid") or {}) - {f"{B}x{C}" for B, C in env}
        if unknown or bad_pts:
            sys.exit(f"unknown keys: predictions {sorted(unknown)}, grid {sorted(bad_pts)}")
    block, ok = report(meas, env, pred, a.tol, a.gpu)
    print(block)
    if a.write:
        doc = (ROOT / "CALIBRATION.md").read_text()
        begin, end = markers(a.gpu)
        i, j = doc.index(begin), doc.index(end) + len(end)
        (ROOT / "CALIBRATION.md").write_text(doc[:i] + block + doc[j:])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
