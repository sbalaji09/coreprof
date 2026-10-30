"""Week 3 benchmark: baseline batcher vs Continuous Depth Batching on the same workload.

For every max_slots S and exit threshold q, both batchers decode the same GSM8K requests
(bf16, no logit recording, no per-call timing syncs). Each point gets a warmup run, then
`--repeats` measured runs with baseline and CDB interleaved so clock drift hits both alike.

Throughput is normalized by work, because in bf16 the two batchers generate slightly
different tokens and exit loops (README, "Equivalence results"):
  - tokens/s        decode tokens emitted (exits) per second
  - row-loops/s     useful (row, loop) block evaluations per second; both batchers compact
                    exited rows away, so every row-loop is useful work
Both are reported for a steady-state window (block calls that start while requests are still
queued, so every slot can be refilled; excludes the end-of-run tail) and for the full run.
Prefill (B=1, serial) is excluded from both and reported separately.

Ideal speedup (proposal: no slot ever idle): ideal block calls = useful row-loops / S,
ideal time = ideal calls x measured time per block call. CDB's achieved speedup over baseline
is reported as a % of the ideal one, with two choices for "time per block call" (see WEEK3.md):
  pct_ideal_own   CDB's own steady time per call (this makes % of ideal = CDB's utilization)
  pct_ideal_full  time per call of a *full* S-row batch, measured from the q=1.0 baseline
And the paper-style figure (Schwethelm et al. 2026, Eq. 1): CDB(q) over the no-exit
baseline (q=1.0) in tokens/s, against the bound r_max / mean exit depth.

Each finished (q, S, batcher, repeat) is appended to results/bench_cdb_<run-id>_runs.jsonl at
once; rerunning with the same --run-id skips those and continues. Repeats of a point must give
identical outputs (at q=1.0 baseline and CDB too), and the GPU is re-checked before each S; on
a mismatch the sweep exits with status 3 without recording the bad run.

Writes results/bench_cdb_<run-id>_runs.jsonl and .csv (one row per run), _points.csv
(median/min/max per point), _speedup.csv, and .json metadata (one entry per session).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

import selfcheck
from batching import baseline, cdb
from batching.common import REVISION, get_model
from batching.workload import GSM8K_REVISION, gsm8k_prompts

BATCHERS = {"baseline": baseline, "cdb": cdb}
N_LOOPS = 4


def window(res, steady):
    """Work and decode time over the steady-state window or the full run (prefill excluded)."""
    steps, st = res.steps, res.stats
    if not steady:
        return dict(calls=st.block_calls, row_loops=st.row_loops, tokens=sum(st.decode_hist),
                    time_s=st.decode_s)
    # Calls that start with requests still queued: every free slot is refilled at once (CDB) or
    # at the next step boundary (baseline). The queue only shrinks, so this is a prefix of the
    # run; call 0 is skipped so the span can start at a recorded call end.
    idx = [i for i, s in enumerate(steps) if i >= 1 and s.queue_len > 0]
    if not idx:
        return None
    i0, i1 = idx[0], idx[-1]
    span = steps[i1].t_end - steps[i0 - 1].t_end
    prefill = steps[i1].prefill_s_before - steps[i0 - 1].prefill_s_before
    win = steps[i0:i1 + 1]
    return dict(calls=len(win), row_loops=sum(s.n_rows for s in win),
                tokens=sum(s.n_exits for s in win), time_s=span - prefill)


def output_hash(res):
    return hashlib.sha256(json.dumps([res.outputs, res.exit_loops]).encode()).hexdigest()[:16]


def measure(name, prompts, max_new, S, q):
    res = BATCHERS[name].run(prompts, max_new, S, q, "bf16")
    st = res.stats
    row = dict(batcher=name, S=S, q=q, n_requests=len(prompts), max_new_tokens=max_new,
               output_hash=output_hash(res),
               prefill_s=st.prefill_s, prefill_ms_per_request=1e3 * st.prefill_s / len(prompts),
               wall_s=st.wall_s, sequences_finished=st.sequences_finished,
               mean_exit_depth=sum((u + 1) * c for u, c in enumerate(st.decode_hist)) / max(1, sum(st.decode_hist)),
               **{f"exit_pct_loop{u + 1}": 100 * c / max(1, sum(st.decode_hist)) for u, c in enumerate(st.decode_hist)})
    for tag, steady in (("full", False), ("steady", True)):
        w = window(res, steady)
        if w is None:
            continue
        row.update({
            f"{tag}_calls": w["calls"], f"{tag}_row_loops": w["row_loops"], f"{tag}_tokens": w["tokens"],
            f"{tag}_time_s": w["time_s"],
            f"{tag}_tokens_per_s": w["tokens"] / w["time_s"],
            f"{tag}_row_loops_per_s": w["row_loops"] / w["time_s"],
            f"{tag}_ms_per_call": 1e3 * w["time_s"] / w["calls"],
            f"{tag}_utilization": w["row_loops"] / (w["calls"] * S),
        })
    return row


def gpu_clocks():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        sm, temp = out.split(",")
        return int(sm), int(temp)
    except Exception:
        return None, None


def summarize(runs: pd.DataFrame):
    metrics = [c for c in runs.columns if c.startswith(("full_", "steady_"))] + \
              ["prefill_s", "prefill_ms_per_request", "mean_exit_depth"] + \
              [f"exit_pct_loop{u}" for u in range(1, N_LOOPS + 1)]
    g = runs.groupby(["q", "S", "batcher"])[metrics]
    med, lo, hi = g.median(), g.min(), g.max()
    out = med.add_suffix("_median").join(lo.add_suffix("_min")).join(hi.add_suffix("_max")).reset_index()
    return out


def speedups(runs: pd.DataFrame):
    """Per (q, S): CDB over baseline, paired by repeat (they ran back to back), median and range."""
    rows = []
    full_call = (runs[(runs.q == 1.0) & (runs.batcher == "baseline")]
                 .groupby("S")["steady_ms_per_call"].median())
    nonexit = (runs[(runs.q == 1.0) & (runs.batcher == "baseline")]
               .groupby("S")["steady_tokens_per_s"].median())
    for (q, S), grp in runs.groupby(["q", "S"]):
        b = grp[grp.batcher == "baseline"].set_index("repeat")
        c = grp[grp.batcher == "cdb"].set_index("repeat")
        reps = b.index.intersection(c.index).sort_values()
        n = len(reps)
        if not n:
            continue
        b, c = b.loc[reps], c.loc[reps]
        d = pd.DataFrame({
            "speedup_row_loops": c.steady_row_loops_per_s / b.steady_row_loops_per_s,
            "speedup_tokens": c.steady_tokens_per_s / b.steady_tokens_per_s,
            "speedup_row_loops_full": c.full_row_loops_per_s / b.full_row_loops_per_s,
            "call_ratio": b.steady_calls / c.steady_calls,
            # proposal's ideal, t_call = CDB's own steady time per block call
            "ideal_speedup_own": (S / (c.steady_ms_per_call / 1e3)) / b.steady_row_loops_per_s,
            # same, t_call = a full S-row block call (q=1 baseline, steady)
            "ideal_speedup_full": (S / (full_call.get(S, float("nan")) / 1e3)) / b.steady_row_loops_per_s,
            # paper-style: vs the no-exit baseline, bound r_max / mean depth
            "speedup_vs_noexit_tokens": c.steady_tokens_per_s / nonexit.get(S, float("nan")),
            "paper_bound": N_LOOPS / c.mean_exit_depth,
        })
        d["pct_ideal_own"] = 100 * d.speedup_row_loops / d.ideal_speedup_own
        d["pct_ideal_full"] = 100 * d.speedup_row_loops / d.ideal_speedup_full
        d["pct_paper_bound"] = 100 * d.speedup_vs_noexit_tokens / d.paper_bound
        row = dict(q=q, S=S, repeats=n)
        for col in d.columns:
            row[f"{col}_median"], row[f"{col}_min"], row[f"{col}_max"] = d[col].median(), d[col].min(), d[col].max()
        rows.append(row)
    return pd.DataFrame(rows)


def hash_key(row):
    """Runs that must produce identical outputs. At q=1 both batchers have the same schedule."""
    return (row["q"], row["S"]) if row["q"] >= 1.0 else (row["q"], row["S"], row["batcher"])


def load_done(path):
    """Finished runs from an earlier session. A line cut off by a crash is dropped and rerun."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"dropping truncated checkpoint line: {line[:60]}...")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_summaries(rows, prefix):
    runs = pd.DataFrame(rows)
    runs.to_csv(f"{prefix}_runs.csv", index=False)
    summarize(runs).to_csv(f"{prefix}_points.csv", index=False)
    sp = speedups(runs)
    sp.to_csv(f"{prefix}_speedup.csv", index=False)
    cols = ["q", "S", "speedup_row_loops_median", "call_ratio_median", "ideal_speedup_own_median",
            "pct_ideal_own_median", "ideal_speedup_full_median", "pct_ideal_full_median",
            "speedup_vs_noexit_tokens_median", "paper_bound_median", "pct_paper_bound_median"]
    print(sp[cols].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--q", type=float, nargs="+", default=[1.0, 0.5, 0.3],
                    help="run in this order; q=1.0 first because the others use it as their control")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=8, help="requests per slot (each slot is refilled ~this often)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None, help="reuse an earlier id to skip its finished points")
    args = ap.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    prefix = out / f"bench_cdb_{run_id}"
    ckpt, meta_path = Path(f"{prefix}_runs.jsonl"), Path(f"{prefix}.json")

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    sessions = json.loads(meta_path.read_text())["sessions"] if meta_path.exists() else []
    for old in sessions:
        for k in ("cycles", "max_new_tokens"):
            if old["args"][k] != vars(args)[k]:
                raise SystemExit(f"--{k} differs from the earlier session of run {run_id}")
    get_model("bf16")
    gpu_uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    sessions.append(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), gpu=torch.cuda.get_device_name(),
                         gpu_uuid=gpu_uuid, torch=torch.__version__, transformers=transformers.__version__,
                         python=platform.python_version(), model_revision=REVISION,
                         gsm8k_revision=GSM8K_REVISION, dtype="bf16", git_commit=git("rev-parse", "HEAD"),
                         git_dirty=bool(git("status", "--porcelain")), args=vars(args)))
    meta_path.write_text(json.dumps(dict(run_id=run_id, sessions=sessions), indent=2))
    print(json.dumps(sessions[-1]))

    all_prompts = gsm8k_prompts(max(args.slots) * args.cycles)
    rows = load_done(ckpt)
    done = {(r["q"], r["S"], r["batcher"], r["repeat"]) for r in rows}
    hashes = {hash_key(r): r["output_hash"] for r in rows}
    print(f"resuming with {len(done)} finished runs" if done else "fresh run", flush=True)

    t_start = time.time()
    for q in args.q:
        for S in args.slots:
            todo = [(rep, name) for rep in range(args.repeats) for name in BATCHERS   # interleaved
                    if (q, S, name, rep) not in done]
            if not todo:
                continue
            if not selfcheck.gpu_repeatability():
                raise SystemExit(3)
            prompts = all_prompts[: S * args.cycles]
            for name in BATCHERS:                          # warmup: new shapes for this S
                BATCHERS[name].run(prompts[:S], 32, S, 0.5, "bf16")
            for rep, name in todo:
                sm0, _ = gpu_clocks()
                row = measure(name, prompts, args.max_new_tokens, S, q)
                sm1, temp = gpu_clocks()
                row.update(repeat=rep, gpu_uuid=gpu_uuid, sm_clock_start=sm0, sm_clock_end=sm1, temp_c=temp)
                if hashes.setdefault(hash_key(row), row["output_hash"]) != row["output_hash"]:
                    print(f"OUTPUT MISMATCH at q={q} S={S} {name} rep={rep}: GPU results not repeatable",
                          flush=True)
                    raise SystemExit(3)
                append_row(ckpt, row)
                rows.append(row)
                print(f"[{time.time() - t_start:6.0f}s] S={S:2d} q={q} rep={rep} {name:8s} "
                      f"steady {row.get('steady_tokens_per_s', float('nan')):7.1f} tok/s "
                      f"{row.get('steady_row_loops_per_s', float('nan')):7.1f} row-loops/s "
                      f"{row.get('steady_ms_per_call', float('nan')):5.1f} ms/call "
                      f"util {row.get('steady_utilization', float('nan')):.2f} | full "
                      f"{row['full_tokens_per_s']:7.1f} tok/s | prefill {row['prefill_s']:.1f}s", flush=True)
            torch.cuda.empty_cache()

    write_summaries(rows, prefix)
    print(f"wrote {prefix}_runs.csv and _points/_speedup CSVs")


if __name__ == "__main__":
    main()
