"""Week 6 check: is the shared-KV collapse real or a KVStore/CDB bug?

Reference independent of batching/kv_store.py: Hugging Face's own UniversalTransformerCache, B=1, all 4 loops
per token (q=1.0). "shared": every loop's cache slot is overwritten with the LAST loop's KV, for the prompt
and after each new token, the same rule as KVStore(n_caches=1) at q=1.0. Compares the greedy tokens with
cdb.run(..., n_caches=1) at q=1.0 on the same questions, plus the unshared reference as a control.

--e1 (results/NOTEBOOK_KV_SHARING.md, E1): HF reference only, three arms graded on GSM8K: unshared, full sharing
(above), and decode-only (Ouro §5.4.2 "last-step only": the prompt keeps its 4 per-loop caches; only generated
positions are overwritten with loop 4's KV).
"""
import argparse

import torch

from batching import cdb
from batching.common import get_model
from batching.workload import tokenizer
from quality_check import N_SHOT, clean, extract, load, to_number


def share_last_loop(cache, n_loops, n_layers, last_only):
    for l in range(n_layers):
        src = (n_loops - 1) * n_layers + l
        for u in range(n_loops - 1):
            i = u * n_layers + l
            for store in (cache.key_cache, cache.value_cache):
                if last_only:
                    store[i][:, :, -1:] = store[src][:, :, -1:]
                else:
                    store[i].copy_(store[src])


@torch.inference_mode()
def reference(model, prompt, n_new, shared, share_prompt=True):
    m = model.model
    n_loops, n_layers = m.total_ut_steps, m.config.num_hidden_layers
    out, hidden, _ = m(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True)
    cache, toks = out.past_key_values, []
    if shared and share_prompt:
        share_last_loop(cache, n_loops, n_layers, last_only=False)
    for _ in range(n_new):
        toks.append(int(model.lm_head(hidden[-1][:, -1]).argmax(-1)))
        out, hidden, _ = m(input_ids=torch.tensor([[toks[-1]]], device="cuda"), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if shared:
            share_last_loop(cache, n_loops, n_layers, last_only=True)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--e1", action="store_true", help="E1: graded unshared / full / decode-only arms, HF reference only")
    args = ap.parse_args()
    train, test = load("train"), load("test").head(args.n)
    shots = "".join(f"Question: {r.question}\nAnswer: {clean(r.answer)}\n\n" for r in train.head(N_SHOT).itertuples())
    tok = tokenizer()
    prompts = [tok(shots + f"Question: {q}\nAnswer:")["input_ids"] for q in test.question]
    model = get_model("bf16")
    if args.e1:
        return e1(model, tok, prompts, [to_number(a.split("####")[-1].strip()) for a in test.answer], args.max_new_tokens)
    eng = {c: cdb.run(prompts, args.max_new_tokens, 8, 1.0, "bf16", model=model, n_caches=c).outputs for c in (4, 1)}
    for i, p in enumerate(prompts):
        ref = {s: reference(model, p, args.max_new_tokens, s) for s in (False, True)}
        for name, a, b in (("unshared: engine vs HF", eng[4][i], ref[False]), ("shared: engine vs HF", eng[1][i], ref[True])):
            same = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"q{i} {name}: first {same} of {min(len(a), len(b))} tokens identical")
        for name, t in (("HF unshared", ref[False]), ("HF shared  ", ref[True]), ("engine kv1 ", eng[1][i])):
            print(f"   {name}: {tok.decode(t, skip_special_tokens=True)[:160]!r}")


def e1(model, tok, prompts, gold, n_new):
    arms = {"unshared": dict(shared=False), "full": dict(shared=True), "decode-only": dict(shared=True, share_prompt=False)}
    ok = {a: [] for a in arms}
    for i, (p, g) in enumerate(zip(prompts, gold)):
        toks = {a: reference(model, p, n_new, **kw) for a, kw in arms.items()}
        print(f"q{i} gold {g}, prompt {len(p)} tokens")
        for a, t in toks.items():
            text = tok.decode(t, skip_special_tokens=True)
            pred = extract(text)
            ok[a].append(pred is not None and g is not None and abs(pred - g) < 1e-6)
            same = next((j for j, (x, y) in enumerate(zip(t, toks["unshared"])) if x != y), len(t))
            print(f"   {a:11s} pred {pred} ok {ok[a][-1]}, first {same}/{len(t)} tokens = unshared: "
                  f"{text.split('Question:')[0][:300]!r}", flush=True)
    for a in arms:
        print(f"E1 {a}: {sum(ok[a])}/{len(ok[a])} correct")
    print("E1 decode-only right where unshared right:",
          f"{sum(u and d for u, d in zip(ok['unshared'], ok['decode-only']))}/{sum(ok['unshared'])}")


if __name__ == "__main__":
    main()
