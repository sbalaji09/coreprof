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

Writes results/bench_cdb_<ts>_runs.csv (one row per run), _points.csv (median/min/max per
point), _speedup.csv, and .json metadata.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

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


def measure(name, prompts, max_new, S, q):
    res = BATCHERS[name].run(prompts, max_new, S, q, "bf16")
    st = res.stats
    row = dict(batcher=name, S=S, q=q, n_requests=len(prompts), max_new_tokens=max_new,
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
        b = grp[grp.batcher == "baseline"].sort_values("repeat").reset_index(drop=True)
        c = grp[grp.batcher == "cdb"].sort_values("repeat").reset_index(drop=True)
        n = min(len(b), len(c))
        b, c = b.iloc[:n], c.iloc[:n]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--q", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=8, help="requests per slot (each slot is refilled ~this often)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    get_model("bf16")
    all_prompts = gsm8k_prompts(max(args.slots) * args.cycles)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    meta = dict(timestamp=ts, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                transformers=transformers.__version__, python=platform.python_version(),
                model_revision=REVISION, gsm8k_revision=GSM8K_REVISION, dtype="bf16",
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                args=vars(args))
    (out / f"bench_cdb_{ts}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))

    runs = []
    runs_path = out / f"bench_cdb_{ts}_runs.csv"
    t_start = time.time()
    for S in args.slots:
        prompts = all_prompts[: S * args.cycles]
        for name in BATCHERS:                              # warmup: new shapes for this S
            BATCHERS[name].run(prompts[:S], 32, S, 0.5, "bf16")
        for q in args.q:
            for rep in range(args.repeats):
                for name in BATCHERS:                      # interleaved: b, c, b, c, ...
                    sm0, _ = gpu_clocks()
                    row = measure(name, prompts, args.max_new_tokens, S, q)
                    sm1, temp = gpu_clocks()
                    row.update(repeat=rep, sm_clock_start=sm0, sm_clock_end=sm1, temp_c=temp)
                    runs.append(row)
                    pd.DataFrame(runs).to_csv(runs_path, index=False)   # keep partial results
                    print(f"[{time.time() - t_start:6.0f}s] S={S:2d} q={q} rep={rep} {name:8s} "
                          f"steady {row.get('steady_tokens_per_s', float('nan')):7.1f} tok/s "
                          f"{row.get('steady_row_loops_per_s', float('nan')):7.1f} row-loops/s "
                          f"{row.get('steady_ms_per_call', float('nan')):5.1f} ms/call "
                          f"util {row.get('steady_utilization', float('nan')):.2f} | full "
                          f"{row['full_tokens_per_s']:7.1f} tok/s | prefill {row['prefill_s']:.1f}s", flush=True)
            torch.cuda.empty_cache()

    runs = pd.DataFrame(runs)
    summarize(runs).to_csv(out / f"bench_cdb_{ts}_points.csv", index=False)
    sp = speedups(runs)
    sp.to_csv(out / f"bench_cdb_{ts}_speedup.csv", index=False)
    cols = ["q", "S", "speedup_row_loops_median", "call_ratio_median", "ideal_speedup_own_median",
            "pct_ideal_own_median", "ideal_speedup_full_median", "pct_ideal_full_median",
            "speedup_vs_noexit_tokens_median", "paper_bound_median", "pct_paper_bound_median"]
    print(sp[cols].round(3).to_string(index=False))
    print(f"wrote {runs_path} and _points/_speedup CSVs")


if __name__ == "__main__":
    main()
