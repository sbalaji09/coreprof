"""Week 6b training progress as bars, and time left per launched task, from the Modal container logs (no GPU,
nothing downloaded).

Usage:
  .venv/bin/python watch_w6b.py          # once
  .venv/bin/python watch_w6b.py -w 60    # refresh every 60 s (Ctrl-C to stop)
  .venv/bin/python watch_w6b.py -w 120 --guard   # also stop a main run that breaks a pre-registered spending check

Runs report at each eval (every 100 steps in the lr probe, every 250 in the main runs); between evals the bar shows
"~" steps estimated from the last eval's log timestamp and step time. The volume's log.jsonl can't be used live:
the trainer holds it open, and Modal's background commits skip open files.
"""
import argparse
import json
from datetime import datetime, timedelta
import subprocess
import time
from pathlib import Path

from modal_app import TASKS
from watch_runs import bar

MODAL = str(Path(__file__).resolve().parent / ".venv/bin/modal")


def modal(*args):
    return subprocess.run([MODAL, *args], capture_output=True, text=True).stdout


def runs(app_id):
    """Parse the task id (env line) and train_looped.py's stdout: a meta line per run, eval records, a final line."""
    out, rid, resumed = [], None, 0
    for ln in modal("app", "logs", "--timestamps", app_id).splitlines():
        try:
            ts, ln = datetime.fromisoformat(ln[:25]), ln[26:]
        except ValueError:
            ts = None
        if ln.startswith("task="):
            rid = ln[5:].strip()
        if ln.startswith("resumed at step "):
            resumed = int(ln.split()[-1])
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
            out.append(dict(name=name, steps=r["steps"], smoke="smoke" in a["out_dir"], probe="w6b_lr" in a["out_dir"],
                            every=a["eval_every"], t0=ts, resumed=resumed, evals=[], last=None, final=None))
            resumed = 0
        elif out and "step" in r:
            r["t"] = ts
            out[-1]["last"] = r
            out[-1]["evals"].append(r)
        elif out and "final_val_loss" in r:
            out[-1]["final"] = r
    return rid, out


def broken(r):
    """CALIBRATION.md Week 6b spending checks (a698ff0): val > 6.5 at step 500, or a rise of > 0.5 nats between evals."""
    ev = r["evals"]
    for prev, cur in zip([None] + ev, ev):
        if cur["step"] == 500 and cur["val_loss"] > 6.5:
            return f"val {cur['val_loss']:.3f} > 6.5 at step 500"
        if prev and cur["val_loss"] - prev["val_loss"] > 0.5:
            return f"val rose {prev['val_loss']:.3f} -> {cur['val_loss']:.3f} (steps {prev['step']}->{cur['step']})"
    return None


def now_step(r):
    """Steps done now: the last eval's step plus the time since that log line over the measured step time."""
    last = r["last"]
    if not last.get("t"):
        return last["step"]
    since = (datetime.now(last["t"].tzinfo) - last["t"]).total_seconds()
    return min(r["steps"], last["step"] + int(since / last["sec"]))


def silent(r):
    """A live run with no eval for twice its usual interval (or 45 min with none yet) has probably died: on 2026-11-24 a
    Modal spend limit killed six runs and nothing else flagged it for an hour."""
    last = r["last"]
    t, limit = (last["t"], 2 * r["every"] * last["sec"] / 60) if last and last.get("t") else (r["t0"], 45)
    if not t:
        return None
    mins = (datetime.now(t.tzinfo) - t).total_seconds() / 60
    return f"no new eval for {mins:.0f} min (limit {limit:.0f})" if mins > limit else None


def time_left(app, rid, rs):
    """Elapsed, run k of n, and the time left for the whole task: the current run from its step time, each queued
    run at the length of a finished one (or the current run's projected length). Eval and compile time add ~5 %."""
    start = datetime.fromisoformat(app["created_at"])
    elapsed = (datetime.now(start.tzinfo) - start).total_seconds() / 3600
    task = rid.rsplit("-", 3)[0]
    n = sum("train_looped.py" in c for c in TASKS.get(task, []))
    done = sum(1 for r in rs if r["final"])
    cur = next((r for r in rs if not r["final"]), None)
    per_run = [r["final"]["hours"] for r in rs if r["final"]]
    left = None
    if cur and cur["last"]:
        sec = cur["last"]["sec"]
        left = (cur["steps"] - now_step(cur)) * sec / 3600
        per_run.append(cur["steps"] * sec / 3600)
    if per_run and (left is not None or not cur):
        left = (left or 0) + max(0, n - done - (1 if cur else 0)) * sum(per_run) / len(per_run)
    k = min(done + 1, n) if n else done + 1
    est = f"~{left:.1f} h left (finish ~{(datetime.now() + timedelta(hours=left)).strftime('%H:%M')})" \
        if left is not None else "time left: known after the current run's first eval"
    return f"{rid}: {elapsed:.1f} h elapsed, run {k} of {n or '?'}, {est}"


def show(guard=False):
    apps = json.loads(modal("app", "list", "--json") or "[]")
    print(time.strftime("%H:%M:%S"), "Week 6b runs (Modal apps listed:", len(apps), ")")
    shown, parsed = 0, [(app, *runs(app["app_id"])) for app in apps]
    running = {r["name"] for app, _, rs in parsed if app["state"] != "stopped" for r in rs}
    for app, rid, rs in parsed:
        live = app["state"] != "stopped"
        if live and rid:
            print(time_left(app, rid, rs))
        for r in rs:
            if r["smoke"] or (not live and not r["final"] and r["name"] in running):   # superseded by a resumed run
                continue
            shown += 1
            last, n = r["last"], r["steps"]
            if r["final"]:
                tail = f"done  final val {r['final']['final_val_loss']:.4f}  ({r['final']['hours']:.2f} h)"
                done = n
            elif last:
                done = now_step(r) if live else last["step"]
                tail = (f"~step {done}, val {last['val_loss']:.3f} @ {last['step']}  {last['tok_per_s'] / 1e3:.0f}k tok/s  "
                        f"~{(n - done) * last['sec'] / 3600:.1f} h left" if live else "STOPPED before finishing")
            else:
                done = r["resumed"]
                tail = ((f"resumed at step {done}, compiling" if done else "compiling / first steps") if live
                        else "STOPPED before first eval")
            print(f"  {r['name']:<28} {bar(done, n)}  {tail}")
            quiet = live and not r["final"] and silent(r)
            if quiet:
                print(f"  ALERT {r['name']} ({app['app_id']}): {quiet}; check `modal container list` and the app logs")
            why = broken(r)
            if why and not r["probe"] and not r["final"]:
                print(f"  ALERT {r['name']} ({app['app_id']}): {why}")
                if guard and live:
                    p = subprocess.run([MODAL, "app", "stop", "-y", app["app_id"]], capture_output=True, text=True)
                    print(f"  STOPPED {app['app_id']}; its last checkpoint stays on the volume (resumable)" if
                          p.returncode == 0 else f"  STOP FAILED for {app['app_id']}: {p.stderr.strip()[-300:]}")
    if not shown:
        print("  no Week 6b runs in the recent Modal apps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--watch", type=int, metavar="SECONDS", help="refresh every SECONDS")
    ap.add_argument("--guard", action="store_true", help="stop a main run that breaks a spending check")
    a = ap.parse_args()
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")
        show(a.guard)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
