"""Week 6b training progress as bars, from the Modal container logs (no GPU, nothing downloaded).

Usage:
  .venv/bin/python watch_w6b.py          # once
  .venv/bin/python watch_w6b.py -w 60    # refresh every 60 s (Ctrl-C to stop)

The bar moves at each eval: every 100 steps in the lr probe, every 250 in the main runs. The volume's log.jsonl
can't be used live: the trainer holds it open, and Modal's background commits skip open files.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from watch_runs import bar

MODAL = str(Path(__file__).resolve().parent / ".venv/bin/modal")


def modal(*args):
    return subprocess.run([MODAL, *args], capture_output=True, text=True).stdout


def runs(app_id):
    """Parse train_looped.py's stdout: a meta line per run, then eval records, then a final line."""
    out = []
    for ln in modal("app", "logs", app_id).splitlines():
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "variant" in r and "final_val_loss" not in r:
            a = r["args"]
            name = r["variant"] + (f" s{a['seed']}" if a["seed"] else "")
            if "w6b_lr" in a["out_dir"]:
                name = f"probe lr {a['lr']:g} {name}"
            out.append(dict(name=name, steps=r["steps"], smoke="smoke" in a["out_dir"], last=None, final=None))
        elif out and "step" in r:
            out[-1]["last"] = r
        elif out and "final_val_loss" in r:
            out[-1]["final"] = r
    return out


def show():
    apps = json.loads(modal("app", "list", "--json") or "[]")
    print(time.strftime("%H:%M:%S"), "Week 6b runs (Modal apps listed:", len(apps), ")")
    shown = 0
    for app in apps:
        live = app["state"] != "stopped"
        for r in runs(app["app_id"]):
            if r["smoke"]:
                continue
            shown += 1
            last, n = r["last"], r["steps"]
            if r["final"]:
                tail = f"done  final val {r['final']['final_val_loss']:.4f}  ({r['final']['hours']:.2f} h)"
                done = n
            elif last:
                tail = (f"val {last['val_loss']:.3f}  {last['tok_per_s'] / 1e3:.0f}k tok/s  "
                        f"~{(n - last['step']) * last['sec'] / 3600:.1f} h left" if live else "STOPPED before finishing")
                done = last["step"]
            else:
                tail, done = ("compiling / first steps" if live else "STOPPED before first eval"), 0
            print(f"  {r['name']:<28} {bar(done, n)}  {tail}")
    if not shown:
        print("  no Week 6b runs in the recent Modal apps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--watch", type=int, metavar="SECONDS", help="refresh every SECONDS")
    a = ap.parse_args()
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")
        show()
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
