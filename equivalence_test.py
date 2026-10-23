"""Equivalence checks for the batchers (DESIGN_CDB.md §5). Prints PASS/FAIL per check.

  check 0   fp32, q=1: reference == HF generate() (validates the reference itself)
  check 1-3 fp32, q in {0.3, 0.5, 1.0}: reference == baseline == CDB, tokens AND per-token exit loops
  bf16      q in {0.5, 0.3}: agreement numbers vs the reference (reported, no pass threshold)

Runs selfcheck's GPU repeatability test first and refuses to continue if it fails.
"""
import argparse
import datetime as dt
import gc
import json
import os

import torch

import selfcheck
from batching import baseline, cdb, reference
from batching.common import get_model, set_numerics
from batching.workload import gsm8k_prompts

BATCHED = {"baseline": baseline, "cdb": cdb}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["fp32", "bf16"])
    p.add_argument("--fp32-prompts", type=int, default=16)
    p.add_argument("--fp32-tokens", type=int, default=32)
    p.add_argument("--fp32-slots", type=int, default=4)
    p.add_argument("--fp32-qs", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    p.add_argument("--bf16-prompts", type=int, default=128)
    p.add_argument("--bf16-tokens", type=int, default=128)
    p.add_argument("--bf16-slots", type=int, default=32)
    p.add_argument("--bf16-qs", type=float, nargs="+", default=[0.5, 0.3])
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def free_gpu():
    get_model.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()


def first_divergence(a_tok, a_exit, b_tok, b_exit):
    """Index of the first position where token or exit loop differs (len if none)."""
    n = min(len(a_tok), len(b_tok))
    for i in range(n):
        if a_tok[i] != b_tok[i] or a_exit[i] != b_exit[i]:
            return i
    return n


def top2_gap(logits):
    top = torch.topk(logits.float(), 2).values
    return (top[0] - top[1]).item()


# ---------------------------------------------------------------- fp32

def hf_generate(model, prompts, max_new):
    """HF's own decode with every token forced to the last loop (= q of 1)."""
    model.early_exit_step = model.model.total_ut_steps - 1
    outs = []
    try:
        with torch.inference_mode():
            for p in prompts:
                ids = torch.tensor([p], device="cuda")
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                     do_sample=False, pad_token_id=0)
                outs.append(out[0, len(p):].tolist())
    finally:
        model.early_exit_step = None
    return outs


def report_mismatch(name, ref, res, max_show=3):
    shown = 0
    for r, (a, b) in enumerate(zip(res.outputs, ref.outputs)):
        d = first_divergence(a, res.exit_loops[r], b, ref.exit_loops[r])
        if d == len(b) and len(a) == len(b):
            continue
        kind = "token" if d < min(len(a), len(b)) and a[d] != b[d] else "exit loop"
        msg = f"    {name} req {r}: first {kind} mismatch at position {d}"
        if ref.logits is not None and d < len(b):
            msg += f" (reference top-2 logit gap there: {top2_gap(ref.logits[r][d]):.2e})"
        if d < min(len(a), len(b)):
            msg += f"; tokens {a[d]} vs {b[d]}, exit loops {res.exit_loops[r][d]} vs {ref.exit_loops[r][d]}"
        print(msg)
        shown += 1
        if shown == max_show:
            break


def fp32_checks(args, prompts):
    set_numerics()
    model = get_model("fp32")
    results, ok_all = {}, True

    ref1 = reference.run(prompts, args.fp32_tokens, 1, 1.0, "fp32", record_logits=True)
    hf = hf_generate(model, prompts, args.fp32_tokens)
    same = sum(a == b for a, b in zip(ref1.outputs, hf))
    ok = same == len(prompts)
    ok_all &= ok
    print(f"[check 0] fp32 q=1.0 reference vs HF generate(): {same}/{len(prompts)} sequences identical"
          f"  -> {'PASS' if ok else 'FAIL'}")
    results["check0_reference_vs_hf"] = {"identical": same, "total": len(prompts), "pass": ok}

    for i, q in enumerate(args.fp32_qs, start=1):
        ref = ref1 if q == 1.0 else reference.run(prompts, args.fp32_tokens, 1, q, "fp32", record_logits=True)
        entry = {"reference_hist": ref.stats.decode_hist}
        for name, mod in BATCHED.items():
            res = mod.run(prompts, args.fp32_tokens, args.fp32_slots, q, "fp32")
            tok_same = sum(a == b for a, b in zip(res.outputs, ref.outputs))
            exit_same = sum(a == b for a, b in zip(res.exit_loops, ref.exit_loops))
            ok = tok_same == exit_same == len(prompts)
            if q == 1.0:
                ok &= res.outputs == hf
            ok_all &= ok
            print(f"[check {i}] fp32 q={q} {name} vs reference: tokens identical {tok_same}/{len(prompts)}, "
                  f"exit loops identical {exit_same}/{len(prompts)}"
                  + (", == HF generate()" if q == 1.0 and res.outputs == hf else "")
                  + f"  -> {'PASS' if ok else 'FAIL'}")
            print(f"           {res.stats.summary()}")
            if not ok:
                report_mismatch(name, ref, res)
            entry[name] = {"tokens_identical": tok_same, "exits_identical": exit_same, "pass": ok,
                           "hist": res.stats.decode_hist, "block_calls": res.stats.block_calls,
                           "utilization": res.stats.slot_utilization}
        results[f"fp32_q{q}"] = entry
    free_gpu()
    return ok_all, results


# ---------------------------------------------------------------- bf16

def agreement(ref, res):
    n = len(ref.outputs)
    exact = prefix_frac = 0.0
    exit_match = exit_total = 0
    max_logit_diff = 0.0
    for r in range(n):
        a_tok, a_exit, b_tok, b_exit = res.outputs[r], res.exit_loops[r], ref.outputs[r], ref.exit_loops[r]
        d = first_divergence(a_tok, a_exit, b_tok, b_exit)
        exact += a_tok == b_tok and a_exit == b_exit
        prefix_frac += d / len(b_tok)
        # exit-loop agreement among positions whose tokens (so far) agree
        t = 0
        while t < min(len(a_tok), len(b_tok)) and a_tok[t] == b_tok[t]:
            exit_match += a_exit[t] == b_exit[t]
            exit_total += 1
            if a_exit[t] != b_exit[t]:
                break
            t += 1
        # logits are comparable where the inputs were identical: positions <= d with the same exit loop
        upto = d + 1 if d < min(len(a_tok), len(b_tok)) and a_exit[d] == b_exit[d] else d
        if upto:
            diff = (res.logits[r][:upto] - ref.logits[r][:upto]).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)
    return {"exact_sequences_pct": 100 * exact / n, "mean_prefix_agreement_pct": 100 * prefix_frac / n,
            "exit_agreement_on_shared_tokens_pct": 100 * exit_match / max(1, exit_total),
            "max_logit_diff_shared_prefix": max_logit_diff}


def hist_pct(h):
    n = max(1, sum(h))
    return " / ".join(f"{100 * c / n:.1f}" for c in h)


def bf16_checks(args, prompts):
    results = {}
    for q in args.bf16_qs:
        ref = reference.run(prompts, args.bf16_tokens, 1, q, "bf16", record_logits=True)
        entry = {"reference": {"hist": ref.stats.decode_hist, "summary": ref.stats.summary()}}
        print(f"[bf16 q={q}] reference exit loops 1/2/3/4 = {hist_pct(ref.stats.decode_hist)} %")
        for name, mod in BATCHED.items():
            res = mod.run(prompts, args.bf16_tokens, args.bf16_slots, q, "bf16", record_logits=True)
            m = agreement(ref, res)
            print(f"[bf16 q={q}] {name} vs reference: exact sequences {m['exact_sequences_pct']:.1f}%, "
                  f"mean prefix agreement {m['mean_prefix_agreement_pct']:.1f}%, exit agreement on shared "
                  f"tokens {m['exit_agreement_on_shared_tokens_pct']:.2f}%, max |logit diff| on shared prefix "
                  f"{m['max_logit_diff_shared_prefix']:.3f}")
            print(f"           {res.stats.summary()}")
            entry[name] = {**m, "hist": res.stats.decode_hist, "block_calls": res.stats.block_calls,
                           "utilization": res.stats.slot_utilization, "summary": res.stats.summary()}
            del res
            torch.cuda.empty_cache()
        results[f"bf16_q{q}"] = entry
        del ref
        torch.cuda.empty_cache()
    free_gpu()
    return results


def main():
    args = parse_args()
    if not selfcheck.gpu_repeatability():
        raise SystemExit("GPU failed the repeatability check; not running equivalence tests.")
    out = {"args": vars(args), "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "gpu": torch.cuda.get_device_name()}
    ok = True
    if args.only != "bf16":
        ok, out["fp32"] = fp32_checks(args, gsm8k_prompts(args.fp32_prompts))
        print("fp32 equivalence: " + ("ALL PASS" if ok else "FAIL"))
    if args.only != "fp32":
        if not ok:
            print("Skipping bf16: fp32 must pass first.")
        else:
            out["bf16"] = bf16_checks(args, gsm8k_prompts(args.bf16_prompts))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"equivalence_{dt.datetime.now():%Y%m%d-%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
