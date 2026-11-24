"""Week 3 quality check: does early exit wreck GSM8K answers?

First N GSM8K test questions, 4-shot chain-of-thought (the first 4 train examples, calculator
annotations stripped), greedy, decoded with CDB in bf16 at each exit threshold q. q = 1.0 is the
no-exit model. The answer is the number after "####" in the generated text, cut at the next
"Question:"; if there is no "####", the last number before the cut. Data is cached under $HF_HOME.

Writes results/quality_<ts>.json (accuracy, exit histogram, per-question predictions).

Week 6 (--variants): the same check for weight / KV formats LoopCore would use, each compared question
by question against "bf16" (run it first in the same list): accuracy with a 95 % Wilson interval, the
exact McNemar test on the questions where exactly one of the two is right, and mean exit depth with the
standard error of its paired per-question difference. The report is rewritten after every variant.
"""
import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from batching import cdb
from batching.common import get_model
from batching.workload import GSM8K_REVISION, tokenizer

N_SHOT = 4
# Week 6 variants. weights: bf16 | int8 (delta_split.Int8WeightOnlyLinear, the path the GPU baseline timed)
# | int4 (rounded, group 128). n_caches / kv_quant: batching/kv_store.py.
VARIANTS = {
    "bf16": dict(),
    "int8w": dict(weights="int8"),
    "int4w": dict(weights="int4"),
    "fp8kv": dict(kv_quant="fp8"),
    "int4kv": dict(kv_quant="int4"),
    "kv2": dict(n_caches=2),
    "kv1": dict(n_caches=1),
    "chip": dict(weights="int8", kv_quant="fp8", n_caches=1),   # LoopCore's assumed design point
    "kvdec": dict(share_decode=True),   # decode-only sharing, prompt keeps 4 caches (NOTEBOOK_KV_SHARING.md E2/E3)
}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load(split):
    path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=GSM8K_REVISION)
    return pd.read_parquet(path)


def clean(answer):
    return re.sub(r"<<.*?>>", "", answer)


def to_number(s):
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract(text):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(" + NUM.pattern + ")", text)
    if m:
        return to_number(m.group(1))
    nums = NUM.findall(text)
    return to_number(nums[-1]) if nums else None


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: b, c = questions only the reference / only the variant got right."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_model(weights):
    get_model.cache_clear()            # quantization is in place: never reuse a model across weight formats
    import torch
    torch.cuda.empty_cache()
    model = get_model("bf16")
    if weights != "bf16":
        import delta_split
        n = delta_split.quantize_int8(model) if weights == "int8" else delta_split.quantize_int4_fake(model)
        print(f"weights {weights}: {n} looped linears quantized", flush=True)
    return model


def compare_to(r, ref):
    a = [x["ok"] for x in ref["per_question"]]
    b = [x["ok"] for x in r["per_question"]]
    only_ref = sum(x and not y for x, y in zip(a, b))
    only_var = sum(y and not x for x, y in zip(a, b))
    dd = [x - y for x, y in zip(r["depth_per_question"], ref["depth_per_question"])]
    n, diff = len(a), (only_var - only_ref) / len(a)
    se = math.sqrt(max(0.0, only_ref + only_var - n * diff * diff)) / n        # paired difference of proportions
    return dict(accuracy_change_pts=100 * diff, accuracy_change_ci95=(100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)),
                only_reference_right=only_ref,
                only_variant_right=only_var, mcnemar_p=mcnemar_p(only_ref, only_var),
                same_answer=sum(x["pred"] == y["pred"] for x, y in zip(r["per_question"], ref["per_question"])),
                depth_change=statistics.mean(dd), depth_change_se=statistics.stdev(dd) / math.sqrt(len(dd)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--q", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["bf16"],
                    help="Week 6 weight/KV formats; put bf16 first to get paired comparisons")
    args = ap.parse_args()

    train, test = load("train"), load("test").head(args.n)
    shots = "".join(f"Question: {r.question}\nAnswer: {clean(r.answer)}\n\n" for r in train.head(N_SHOT).itertuples())
    tok = tokenizer()
    prompts = [tok(shots + f"Question: {q}\nAnswer:")["input_ids"] for q in test.question]
    gold = [to_number(a.split("####")[-1].strip()) for a in test.answer]

    report = dict(n=args.n, n_shot=N_SHOT, slots=args.slots, max_new_tokens=args.max_new_tokens,
                  gsm8k_revision=GSM8K_REVISION, variants={v: VARIANTS[v] for v in args.variants}, results={})
    out = Path(args.out_dir) / f"quality_{time.strftime('%Y%m%d-%H%M%S')}.json"
    weights_loaded = None
    for v, q in [(v, q) for v in args.variants for q in args.q]:
        cfg = VARIANTS[v]
        if cfg.get("weights", "bf16") != weights_loaded:
            weights_loaded = cfg.get("weights", "bf16")
            model = load_model(weights_loaded)
        key = str(q) if args.variants == ["bf16"] else f"{v} q={q}"    # old key format for plain runs
        t0 = time.time()
        res = cdb.run(prompts, args.max_new_tokens, args.slots, q, "bf16", model=model,
                      n_caches=cfg.get("n_caches"), kv_quant=cfg.get("kv_quant", "none"),
                      share_decode=cfg.get("share_decode", False))
        texts = [tok.decode(o, skip_special_tokens=True) for o in res.outputs]
        preds = [extract(t) for t in texts]
        correct = [p is not None and g is not None and abs(p - g) < 1e-6 for p, g in zip(preds, gold)]
        hist = res.stats.decode_hist
        n_tok = max(1, sum(hist))
        r = dict(variant=v, q=q, accuracy=sum(correct) / len(correct), correct=sum(correct),
                 accuracy_ci95=wilson(sum(correct), len(correct)),
                 mean_exit_depth=sum((u + 1) * c for u, c in enumerate(hist)) / n_tok,
                 depth_per_question=[statistics.mean(e[1:]) if len(e) > 1 else float(e[0]) for e in res.exit_loops],
                 exit_pct=[100 * c / n_tok for c in hist], seconds=time.time() - t0,
                 truncated=sum("####" not in t.split("Question:")[0] for t in texts),
                 kv_clipped=res.stats.kv_clipped,
                 per_question=[dict(gold=g, pred=p, ok=ok, text=t) for g, p, ok, t in zip(gold, preds, correct, texts)])
        ref = report["results"].get(f"bf16 q={q}")
        if ref and v != "bf16":
            r["vs_bf16"] = compare_to(r, ref)
        report["results"][key] = r
        lo, hi = r["accuracy_ci95"]
        print(f"{key}: accuracy {100 * r['accuracy']:.1f}% ({r['correct']}/{args.n}, 95% CI {100 * lo:.1f}-{100 * hi:.1f}), "
              f"mean exit depth {r['mean_exit_depth']:.2f}, exit loops 1/2/3/4 = "
              + " / ".join(f"{x:.1f}" for x in r["exit_pct"]) + f" %, no '####': {r['truncated']}, "
              f"fp8 clipped: {r['kv_clipped']}, {r['seconds']:.0f} s", flush=True)
        if "vs_bf16" in r:
            c = r["vs_bf16"]
            print(f"  vs bf16: {c['accuracy_change_pts']:+.1f} pts (95% CI {c['accuracy_change_ci95'][0]:+.1f} to "
                  f"{c['accuracy_change_ci95'][1]:+.1f}; only bf16 right {c['only_reference_right']}, only "
                  f"{v} right {c['only_variant_right']}, McNemar p={c['mcnemar_p']:.3f}), same answer {c['same_answer']}"
                  f"/{args.n}, depth {c['depth_change']:+.3f} ± {c['depth_change_se']:.3f}", flush=True)
        out.write_text(json.dumps(report, indent=1))

    ref = report["results"].get("1.0")
    if ref and args.variants == ["bf16"]:
        for q, r in report["results"].items():
            agree = sum(a["pred"] == b["pred"] for a, b in zip(r["per_question"], ref["per_question"]))
            r["same_answer_as_q1"] = agree
            print(f"q={q}: same final answer as q=1.0 on {agree}/{args.n}")
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
