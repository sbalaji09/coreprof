"""Live progress + results of the torch.compile / int8 baseline runs (reads the Modal volume; no GPU needed).

Usage:
  .venv/bin/python watch_runs.py                  # the current int8 runs, once
  .venv/bin/python watch_runs.py -w 60            # refresh every 60 s (Ctrl-C to stop)
  .venv/bin/python watch_runs.py <run_id> ...     # other delta_compile runs

"uncompiled" comes from the earlier delta_grid run on the same GPU type (a DIFFERENT container: ±0.25 ms
noise on small steps). The run's own same-container comparison appears in its report when it finishes.
"""
import argparse
import csv
import io
import json
import statistics
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ["delta_int8_triton-h100-20261117-233820", "delta_int8_triton-l40s-20261117-233833",
                "delta_int8_cd_small-h100-20261117-233847", "delta_int8_cd_small-l40s-20261117-233859"]
# (task, gpu) -> the run each task is compared against: compile vs uncompiled, int8 vs bf16 compiled
BASELINE = {("delta_compile", "h100"): "delta_grid-h100-20261116-184347",
            ("delta_compile", "l40s"): "delta_grid-l40s-20261116-184352",
            ("delta_int8", "h100"): "delta_compile-h100-20261116-214809",
            ("delta_int8", "l40s"): "delta_compile-l40s-20261116-214809"}
for _t in ("delta_int8_triton", "delta_int8_cd", "delta_int8_cd_small"):
    for _g in ("h100", "l40s"):
        BASELINE[(_t, _g)] = BASELINE[("delta_int8", _g)]


def read(vol, path):
    try:
        return b"".join(vol.read_file(path)).decode()
    except Exception:          # not written yet
        return None


def grid(text):
    return {(int(r["B"]), int(r["C"])): float(r["median_ms"]) for r in csv.DictReader(io.StringIO(text or ""))}


def bar(done, total, width=30):
    n = round(width * done / total) if total else 0
    return "[" + "#" * n + "." * (width - n) + f"] {done}/{total}"


def show(vol, rid):
    task, gpu = rid.split("-")[:2]
    base_dir = ROOT / "results/modal" / BASELINE[(task, gpu)]
    base_js = json.loads((base_dir / "delta_split.json").read_text())
    base = grid((base_dir / "graph_grid.csv").read_text())
    comp = grid(read(vol, f"/results/{rid}/graph_grid.csv"))
    js = json.loads(read(vol, f"/results/{rid}/delta_split.json") or "{}")
    finished = read(vol, f"/results/{rid}/delta_split_grid.md") is not None

    if finished:
        stage = f"FINISHED (pull: ./run_modal.sh --task {task} --run-id {rid} --pull)"
    elif "compile_variants" in js or "uncompiled_reference" in js or "int8_reference" in js:
        stage = "same-container reference + max-autotune (last step)"
    elif comp:
        stage = "timing the grid"
    elif "lm_head" in js or "kernel_breakdown" in js:
        stage = "kernel breakdowns / LM head"
    else:
        stage = "starting (image, model load, correctness check)"
    print(f"\n=== {rid}  ({js.get('gpu', gpu.upper())})")
    print(f"stage:    {stage}")
    print(f"grid:     {bar(len(comp), len(base))}")
    chk = js.get("compiled_vs_eager_check")
    q = js.get("int8_vs_bf16_check")
    if q:
        print(f"int8:     {q['linears_quantized']} linears, {q['bf16_weight_bytes'] / 1e9:.2f} -> "
              f"{js['weight_bytes_looped_block'] / 1e9:.2f} GB; vs bf16 step: hidden {100 * q['rel_err_hidden']:.1f} %, "
              f"top-1 token agrees {100 * q['top1_agreement']:.0f} %")
    if chk:
        print(f"numerics: compiled vs eager {100 * chk['rel_err_hidden']:.1f} % (bf16 rounding level, see "
              f"compile_check runs); compile {chk.get('compile_s')} s per shape, {chk.get('graph_breaks')} graph breaks")
    if not comp:
        return

    old, new = ("bf16 ms", "int8 ms") if task.startswith("delta_int8") else ("uncompiled ms", "compiled ms")
    print(f"\n{'B':>4} {'C':>5}  {old:>13} {new:>11} {'saved ms':>8}  speedup")
    sp = []
    for (B, C), t in sorted(comp.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        u = base.get((B, C))
        if u:
            sp.append(u / t)
        print(f"{B:>4} {C:>5}  {u or float('nan'):>13.2f} {t:>11.2f} {(u - t) if u else float('nan'):>8.2f}  "
              + (f"{u / t:.2f}x" if u else "n/a"))
    if sp:
        med = statistics.median(sp)
        verdict = ("YES, clearly faster" if med >= 1.10 else "slightly faster" if med >= 1.03
                   else "no real difference (within container noise)" if med > 0.97 else "SLOWER")
        print(f"\nso far: {new.split()[0]} is {verdict}: median {med:.2f}x, range {min(sp):.2f}-{max(sp):.2f}x "
              f"over {len(sp)} points")

    # what it means for LoopCore: the "if weights were free" bound at B=1, C=128
    if task.startswith("delta_int8"):
        return      # int8 weight time is estimated from same-container step times at the end of the run
    w_comp = ((js.get("kernel_breakdown") or {}).get(f"{js.get('grid_impl')} 1x128") or {}).get("gemm_ms")
    w_base = (base_js["kernel_breakdown"].get(f"{base_js.get('grid_impl', 'graph_static')} 1x128") or {}).get("gemm_ms")
    t_c, t_u = comp.get((1, 128)), base.get((1, 128))
    if w_comp and w_base and t_c and t_u:
        print(f"LoopCore bound at B=1, C=128 (step ÷ step-without-weights): uncompiled {t_u / (t_u - w_base):.2f}x "
              f"-> compiled {t_c / (t_c - w_comp):.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("-w", "--watch", type=int, metavar="SECONDS", help="refresh every SECONDS")
    a = ap.parse_args()
    vol = modal.Volume.from_name("loopcore-vol")
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")          # clear screen
        print(time.strftime("%H:%M:%S"), "torch.compile baseline runs")
        for rid in a.runs:
            show(vol, rid)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
