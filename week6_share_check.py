"""Week 6 check: is the shared-KV collapse real or a KVStore/CDB bug?

Reference independent of batching/kv_store.py: Hugging Face's own UniversalTransformerCache, B=1, all 4 loops
per token (q=1.0). "shared": every loop's cache slot is overwritten with the LAST loop's KV, for the prompt
and after each new token, the same rule as KVStore(n_caches=1) at q=1.0. Compares the greedy tokens with
cdb.run(..., n_caches=1) at q=1.0 on the same questions, plus the unshared reference as a control.
"""
import argparse

import torch

from batching import cdb
from batching.common import get_model
from batching.workload import tokenizer
from quality_check import N_SHOT, clean, load


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
def reference(model, prompt, n_new, shared):
    m = model.model
    n_loops, n_layers = m.total_ut_steps, m.config.num_hidden_layers
    out, hidden, _ = m(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True)
    cache, toks = out.past_key_values, []
    if shared:
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
    args = ap.parse_args()
    train, test = load("train"), load("test").head(args.n)
    shots = "".join(f"Question: {r.question}\nAnswer: {clean(r.answer)}\n\n" for r in train.head(N_SHOT).itertuples())
    tok = tokenizer()
    prompts = [tok(shots + f"Question: {q}\nAnswer:")["input_ids"] for q in test.question]
    model = get_model("bf16")
    eng = {c: cdb.run(prompts, args.max_new_tokens, 8, 1.0, "bf16", model=model, n_caches=c).outputs for c in (4, 1)}
    for i, p in enumerate(prompts):
        ref = {s: reference(model, p, args.max_new_tokens, s) for s in (False, True)}
        for name, a, b in (("unshared: engine vs HF", eng[4][i], ref[False]), ("shared: engine vs HF", eng[1][i], ref[True])):
            same = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"q{i} {name}: first {same} of {min(len(a), len(b))} tokens identical")
        for name, t in (("HF unshared", ref[False]), ("HF shared  ", ref[True]), ("engine kv1 ", eng[1][i])):
            print(f"   {name}: {tok.decode(t, skip_special_tokens=True)[:160]!r}")


if __name__ == "__main__":
    main()
