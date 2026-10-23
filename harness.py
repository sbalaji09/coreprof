"""Decode-latency harness for Ouro-1.4B: one recurrent (UT) step vs. one full token.

See NOTES.md for what exactly is being timed and why.

Layout (meant to grow into LoopProf):
  - TARGETS maps a name to a builder: builder(model, state, args) -> zero-arg callable.
    Add a new thing to time by writing a builder and registering it.
  - sweep() iterates over the sweep axes (currently C x B). Add an axis there.
  - time_cuda() / summarize() produce the per-config stats that become CSV columns.
"""
import argparse
import csv
import datetime as dt
import gc
import itertools
import json
import os
import platform
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

MODEL = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"  # pinned; NOTES.md line refs match this
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CSV_FIELDS = [
    "target", "B", "C", "status", "median_ms", "p10_ms", "p90_ms", "iqr_ms",
    "mean_ms", "tokens_per_s", "peak_mem_gb", "sm_clock_mhz", "temp_c",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--context-lens", type=int, nargs="+", default=[512])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--targets", nargs="+", default=["ut_step", "decode_step"])
    p.add_argument("--ut-step", type=int, default=0, help="which recurrent step ut_step times (0-based)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


# ---------------------------------------------------------------- metadata

def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_clock_and_temp():
    out = _cmd(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"])
    if not out:
        return None, None
    clock, temp = (x.strip() for x in out.splitlines()[0].split(","))
    return int(clock), int(temp)


def run_metadata(args, model):
    smi = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                "--format=csv,noheader"])
    commit = _cmd(["git", "rev-parse", "HEAD"])
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "model": MODEL,
        "model_revision": REVISION,
        "attn_implementation": model.config._attn_implementation,
        "total_ut_steps": model.model.total_ut_steps,
        "dtype": args.dtype,
        "git_commit": commit,
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])) if commit else None,
        "hostname": platform.node(),
        "args": vars(args),
    }


# ---------------------------------------------------------------- model state

@dataclass
class State:
    """A batch of B sequences whose KV cache holds exactly C tokens per slot."""
    B: int
    C: int
    cache: object          # UniversalTransformerCache, 96 slots of [B, heads, C, head_dim]
    next_ids: torch.Tensor  # [B, 1] the decode token

    def reset(self):
        """Slice every cache slot back to C tokens (a view, no copy) so each rep sees C past tokens."""
        for kv in (self.cache.key_cache, self.cache.value_cache):
            for i, t in enumerate(kv):
                if t is not None and t.shape[2] > self.C:
                    kv[i] = t[:, :, : self.C]


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=DTYPES[dtype], trust_remote_code=True
    )
    return model.cuda().eval()


def prefill(model, B, C, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (B, C + 1), device="cuda", generator=g)
    out = model.model(input_ids=ids[:, :C], use_cache=True)  # base model: skips full-sequence lm_head
    return State(B=B, C=C, cache=out[0].past_key_values, next_ids=ids[:, C:])


# ---------------------------------------------------------------- targets

def build_ut_step(model, state, args):
    """One iteration of the recurrent loop body (modeling_ouro.py:593-608): 24 layers + norm + exit gate."""
    m = model.model
    ut = args.ut_step
    assert 0 <= ut < m.total_ut_steps, f"--ut-step must be in [0, {m.total_ut_steps})"
    cache_position = torch.arange(state.C, state.C + 1, device="cuda")
    position_ids = cache_position.unsqueeze(0)
    hidden = m.embed_tokens(state.next_ids)
    # Same per-forward setup as OuroModel.forward (lines 565-588).
    position_embeddings = m.rotary_emb(hidden, position_ids)
    mask = create_causal_mask(config=m.config, input_embeds=hidden, attention_mask=None,
                              cache_position=cache_position, past_key_values=state.cache,
                              position_ids=position_ids)
    masks = {"full_attention": mask}
    layers = m.layers[: m.config.num_hidden_layers]

    def loop_body(h, current_ut):
        for layer in layers:
            h = layer(h, attention_mask=masks[layer.attention_type], position_ids=position_ids,
                      past_key_value=state.cache, use_cache=True, cache_position=cache_position,
                      position_embeddings=position_embeddings, current_ut=current_ut)
        h = m.norm(h)
        return h, m.early_exit_gate(h)

    # Realistic input for step `ut`: the decode token after the preceding steps (untimed).
    for prev in range(ut):
        hidden, _ = loop_body(hidden, prev)
    state.reset()

    return lambda: loop_body(hidden, ut)


def build_decode_step(model, state, args):
    """One full decode token: OuroForCausalLM.forward (all UT steps, exit selection, lm_head)."""
    cache_position = torch.arange(state.C, state.C + 1, device="cuda")
    return lambda: model(input_ids=state.next_ids, past_key_values=state.cache, use_cache=True,
                         cache_position=cache_position, logits_to_keep=1)


TARGETS = {"ut_step": build_ut_step, "decode_step": build_decode_step}


# ---------------------------------------------------------------- timing

def time_cuda(fn, reset, warmup, reps):
    """Per-rep GPU time in ms via CUDA events; the queue is drained before each start."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(warmup + reps):
        reset()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return times


def summarize(times):
    p10, p25, p50, p75, p90 = np.percentile(times, [10, 25, 50, 75, 90])
    return {"median_ms": p50, "p10_ms": p10, "p90_ms": p90, "iqr_ms": p75 - p25,
            "mean_ms": float(np.mean(times))}


# ---------------------------------------------------------------- sweep

def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def sweep(args):
    """Yield (C, B) configs. Context length is the outer axis so B varies fastest."""
    return itertools.product(args.context_lens, args.batch_sizes)


def run_config(model, args, C, B):
    """Return one CSV row per target for this config (status='oom' if it didn't fit)."""
    rows, raw = [], {}
    base = {"B": B, "C": C}
    try:
        state = prefill(model, B, C, args.seed)
    except torch.cuda.OutOfMemoryError:
        free_cuda()
        return [{**base, "target": t, "status": "oom_prefill"} for t in args.targets], raw

    for target in args.targets:
        row = {**base, "target": target}
        try:
            state.reset()
            fn = TARGETS[target](model, state, args)
            torch.cuda.reset_peak_memory_stats()
            times = time_cuda(fn, state.reset, args.warmup, args.reps)
            clock, temp = gpu_clock_and_temp()
            stats = summarize(times)
            row.update(stats, status="ok", tokens_per_s=B / (stats["median_ms"] / 1e3),
                       peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                       sm_clock_mhz=clock, temp_c=temp)
            raw[target] = times
        except torch.cuda.OutOfMemoryError:
            row["status"] = "oom"
            fn = None
            free_cuda()
        rows.append(row)
    del state, fn
    free_cuda()
    return rows, raw


def main():
    args = parse_args()
    unknown = set(args.targets) - TARGETS.keys()
    if unknown:
        raise SystemExit(f"unknown targets {unknown}; choose from {list(TARGETS)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.dtype)
    meta = run_metadata(args, model)
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.out_dir, f"{stamp}.csv")
    meta_path = os.path.join(args.out_dir, f"{stamp}.json")
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=1))

    meta["raw_times_ms"] = {}
    with torch.inference_mode(), open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for C, B in sweep(args):
            rows, raw = run_config(model, args, C, B)
            for row in rows:
                writer.writerow(row)
                f.flush()
                if row["status"] == "ok":
                    print(f"{row['target']:>11}  B={B:<3} C={C:<5} median={row['median_ms']:8.3f} ms  "
                          f"p10-p90=[{row['p10_ms']:.3f}, {row['p90_ms']:.3f}]  "
                          f"{row['tokens_per_s']:9.1f} tok/s  peak={row['peak_mem_gb']:.2f} GB  "
                          f"sm={row['sm_clock_mhz']} MHz")
                else:
                    print(f"{row['target']:>11}  B={B:<3} C={C:<5} {row['status']}")
            for target, times in raw.items():
                meta["raw_times_ms"][f"{target}/B{B}/C{C}"] = times

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {csv_path} and {meta_path}")


if __name__ == "__main__":
    main()
