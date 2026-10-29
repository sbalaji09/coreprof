"""Week 3 quality check: does early exit wreck GSM8K answers?

First N GSM8K test questions, 4-shot chain-of-thought (the first 4 train examples, calculator
annotations stripped), greedy, decoded with CDB in bf16 at each exit threshold q. q = 1.0 is the
no-exit model. The answer is the number after "####" in the generated text, cut at the next
"Question:"; if there is no "####", the last number before the cut. Data is cached under $HF_HOME.

Writes results/quality_<ts>.json (accuracy, exit histogram, per-question predictions).
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from batching import cdb
from batching.workload import GSM8K_REVISION, tokenizer

N_SHOT = 4
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--q", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    train, test = load("train"), load("test").head(args.n)
    shots = "".join(f"Question: {r.question}\nAnswer: {clean(r.answer)}\n\n" for r in train.head(N_SHOT).itertuples())
    tok = tokenizer()
    prompts = [tok(shots + f"Question: {q}\nAnswer:")["input_ids"] for q in test.question]
    gold = [to_number(a.split("####")[-1].strip()) for a in test.answer]

    report = dict(n=args.n, n_shot=N_SHOT, slots=args.slots, max_new_tokens=args.max_new_tokens,
                  gsm8k_revision=GSM8K_REVISION, results={})
    for q in args.q:
        t0 = time.time()
        res = cdb.run(prompts, args.max_new_tokens, args.slots, q, "bf16")
        texts = [tok.decode(o, skip_special_tokens=True) for o in res.outputs]
        preds = [extract(t) for t in texts]
        correct = [p is not None and g is not None and abs(p - g) < 1e-6 for p, g in zip(preds, gold)]
        hist = res.stats.decode_hist
        n_tok = max(1, sum(hist))
        r = dict(accuracy=sum(correct) / len(correct), correct=sum(correct),
                 mean_exit_depth=sum((u + 1) * c for u, c in enumerate(hist)) / n_tok,
                 exit_pct=[100 * c / n_tok for c in hist], seconds=time.time() - t0,
                 truncated=sum("####" not in t.split("Question:")[0] for t in texts),
                 per_question=[dict(gold=g, pred=p, ok=ok, text=t) for g, p, ok, t in zip(gold, preds, correct, texts)])
        report["results"][str(q)] = r
        print(f"q={q}: accuracy {100 * r['accuracy']:.1f}% ({r['correct']}/{args.n}), mean exit depth "
              f"{r['mean_exit_depth']:.2f}, exit loops 1/2/3/4 = "
              + " / ".join(f"{x:.1f}" for x in r["exit_pct"]) + f" %, no '####': {r['truncated']}, "
              f"{r['seconds']:.0f} s", flush=True)

    ref = report["results"].get("1.0")
    if ref:
        for q, r in report["results"].items():
            agree = sum(a["pred"] == b["pred"] for a, b in zip(r["per_question"], ref["per_question"]))
            r["same_answer_as_q1"] = agree
            print(f"q={q}: same final answer as q=1.0 on {agree}/{args.n}")
    out = Path(args.out_dir) / f"quality_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
