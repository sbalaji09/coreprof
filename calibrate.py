"""Check the analytical model's predictions against measured GPU data (stdlib only, runs on a laptop).

Usage:
  python calibrate.py [--gpu l40s|h100]            # measured values + table, predictions pending
  python calibrate.py --gpu h100 --pred calibration_predictions_h100.json [--tol 10] [--write]
  python calibrate.py --selftest

Measured values are recomputed from the raw run files every time, never copied by hand.
--write replaces the block between the calibration markers in CALIBRATION.md.
Exit code: 0 only if every quantity has a prediction within --tol percent.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# gpu -> (delta_grid run, expected grid points, device name). L40S stays on the run WEEK5.md is built on.
RUNS = {
    "l40s": ("results/modal/delta_grid-l40s-20261112-172748", 46, "NVIDIA L40S"),
    "h100": ("results/modal/delta_grid-h100-20261116-184347", 48, "NVIDIA H100 80GB HBM3"),
}
markers = lambda gpu: (f"<!-- calibration:{gpu}:begin -->", f"<!-- calibration:{gpu}:end -->")

# key -> (label, unit, source); order = table order
ROWS = {
    "weight_ms": ("weight streaming time per loop step", "ms", "delta_split.json kernel_breakdown (weight GEMV/GEMM kernels, 1x128 and 16x2048)"),
    "kv_us_per_token": ("KV cost per cached token", "µs", "graph_grid.csv, M2 fit"),
    "crossover_tokens": ("weight/KV crossover", "cached tokens", "weight_ms / kv_us_per_token"),
    "step_ms_b1_c128": ("best-case step time, B=1, C=128 (CUDA graph, static cache)", "ms", "graph_grid.csv"),
    "delta_ms": ("fixed cost δ (M2 fit)", "ms", "graph_grid.csv, M2 fit"),
    "overhead_ms": ("non-weight GPU overhead (small kernels + gaps)", "ms", "step_ms_b1_c128 − weight kernels at 1x128"),
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
    gemm_ms is weights only; older runs counted flash attention as gemm, so use the top weight kernel."""
    if "attention_ms" in bd:
        return bd["gemm_ms"]
    return next(v for k, v in bd["top_kernels_ms"].items()
                if ("gemvx" in k or "gemm" in k) and "flash" not in k)


def measured(gpu="l40s"):
    """Each value is (lo, hi); lo == hi for point values."""
    run, n_expected, _ = RUNS[gpu]
    grid = ROOT / run
    rows = [{"B": int(r["B"]), "C": int(r["C"]), "t": float(r["median_ms"])}
            for r in csv.DictReader(open(grid / "graph_grid.csv"))]
    delta, _, alpha, r2 = fit_m2(rows)
    assert len(rows) == n_expected and r2 > 0.99, f"{gpu} grid fit changed: n={len(rows)} R²={r2}"
    kb = json.load(open(grid / "delta_split.json"))["kernel_breakdown"]
    w1, w16 = weight_kernel_ms(kb["graph_static 1x128"]), weight_kernel_ms(kb["graph_static 16x2048"])
    gemv, gemm = min(w1, w16), max(w1, w16)
    step = next(r["t"] for r in rows if (r["B"], r["C"]) == (1, 128))
    return {
        "weight_ms": (gemv, gemm),
        "kv_us_per_token": (alpha * 1e3,) * 2,
        "crossover_tokens": (gemv / alpha, gemm / alpha),
        "step_ms_b1_c128": (step,) * 2,
        "delta_ms": (delta,) * 2,
        "overhead_ms": (step - w1,) * 2,
    }


def error_pct(pred, lo, hi):
    """0 inside the measured range, else % distance to the nearest end."""
    if lo <= pred <= hi:
        return 0.0
    ref = lo if pred < lo else hi
    return 100 * (pred - ref) / ref


def fmt(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def report(meas, pred, tol, gpu="l40s"):
    """Returns (markdown block, all_ok)."""
    lines = [f"| quantity | measured ({RUNS[gpu][2]}) | source | model prediction | error % |", "|---|---|---|---|---|"]
    ok = True
    for key, (label, unit, src) in ROWS.items():
        lo, hi = meas[key]
        m = f"{fmt(lo)} {unit}" if lo == hi else f"{fmt(lo)}–{fmt(hi)} {unit}"
        p = pred.get("predictions", {}).get(key)
        if p is None:
            ok = False
            lines.append(f"| {label} | {m} | {src} | pending | pending |")
            continue
        e = error_pct(p, lo, hi)
        ok &= abs(e) <= tol
        lines.append(f"| {label} | {m} | {src} | {fmt(p)} {unit} | {e:+.1f}{'' if abs(e) <= tol else ' ✗'} |")
    n = sum(pred.get("predictions", {}).get(k) is not None for k in ROWS)
    run, _, name = RUNS[gpu]
    if ok:
        status = f"Calibrated on {name}: {pred.get('date', '?')} (all {len(ROWS)} within ±{tol:g} %)."
    else:
        status = f"Calibrated on {name}: **no** ({n}/{len(ROWS)} predictions recorded, tolerance ±{tol:g} %)."
    inputs = pred.get("inputs") or {}
    inp = "\n".join(f"- {k}: {'pending' if v is None else v}" for k, v in inputs.items()) or "- pending"
    body = (f"**Status:** {status}\n\nMeasured: `{run}/`. Predictions: `{pred.get('_path', 'none')}`"
            f"{', from ' + pred['source'] if pred.get('source') else ''}. "
            f"Error % is 0 inside the measured range, else the distance to its nearest end.\n\n"
            + "\n".join(lines) + f"\n\n**Model inputs:**\n\n{inp}")
    begin, end = markers(gpu)
    return f"{begin}\n{body}\n{end}", ok


def selftest():
    for gpu in RUNS:
        meas = measured(gpu)
        exact = {"predictions": {k: v[0] for k, v in meas.items()}, "date": "x"}
        assert report(meas, exact, 10, gpu)[1]
        assert not report(meas, {}, 10, gpu)[1]
        off = {"predictions": {**exact["predictions"], "delta_ms": meas["delta_ms"][0] * 1.2}}
        assert not report(meas, off, 10, gpu)[1] and report(meas, off, 25, gpu)[1]
    l40s, h100 = measured("l40s"), measured("h100")
    assert round(l40s["weight_ms"][0], 2) == 3.77 and round(h100["weight_ms"][0], 2) == 1.49
    assert error_pct(3.9, 3.77, 3.98) == 0 and round(error_pct(2.85, 3.77, 3.98)) == -24  # datasheet case
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", choices=list(RUNS), default="l40s")
    ap.add_argument("--pred", help="JSON: {date, source, inputs: {...}, predictions: {key: number|null}}")
    ap.add_argument("--tol", type=float, default=10.0, help="max |error %%| to count as calibrated (default 10)")
    ap.add_argument("--write", action="store_true", help="update the calibration block in CALIBRATION.md")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    pred = {}
    if a.pred:
        pred = json.load(open(a.pred))
        pred["_path"] = a.pred
        unknown = set(pred.get("predictions", {})) - set(ROWS)
        if unknown:
            sys.exit(f"unknown prediction keys {sorted(unknown)}; expected {list(ROWS)}")
    block, ok = report(measured(a.gpu), pred, a.tol, a.gpu)
    print(block)
    if a.write:
        doc = (ROOT / "CALIBRATION.md").read_text()
        begin, end = markers(a.gpu)
        i, j = doc.index(begin), doc.index(end) + len(end)
        (ROOT / "CALIBRATION.md").write_text(doc[:i] + block + doc[j:])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
