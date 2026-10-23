"""Reference decoder (DESIGN_CDB.md §5a): each sequence alone (B=1), one loop at a time, with HF's
own UniversalTransformerCache (no padding, no mask). Same Q-exit rule and copy-forward KV rule as
the batchers, but independent of kv_store.py and engine.py, so it catches their bugs instead of
sharing them. max_slots is accepted for interface compatibility and ignored; `steps` is empty
(each token is its own sequence of single-row block calls).
"""
import torch

from .common import (EOS_TOKEN_ID, QExit, RunResult, RunStats, Timer, get_model,
                     model_dims, per_request, prefill, run_block, set_numerics)


def _decode_token(model, cache, token, pos, exit_rule, req, tok_idx, n_loops, n_layers):
    """Run one token through loops until it exits; copy its KV forward into skipped loops."""
    hidden = model.model.embed_tokens(torch.tensor([[token]], device="cuda"))
    position_ids = torch.tensor([[pos]], device="cuda")
    survival = torch.ones(1, device="cuda")
    for u in range(n_loops):
        hidden, gate = run_block(model, hidden, position_ids, cache, None, current_ut=u)
        loop = torch.full((1,), u, device="cuda")
        exit_now, survival = exit_rule(gate, survival, loop, [req], [tok_idx])
        if exit_now.item():
            break
    for j in range(u + 1, n_loops):           # copy-forward: loop u's k/v at this position
        for l in range(n_layers):
            src = u * n_layers + l
            cache.update(cache.key_cache[src][:, :, -1:], cache.value_cache[src][:, :, -1:],
                         j * n_layers + l)
    logits = model.lm_head(hidden[:, 0]).float()[0]
    return int(logits.argmax().item()), u, logits


def run(prompts, max_new_tokens, max_slots, exit_threshold, dtype, *, model=None, exit_rule=None,
        record_logits=False, time_calls=False):
    set_numerics()
    model = model or get_model(dtype)
    dims = model_dims(model)
    n_loops, n_layers = dims["n_loops"], dims["n_layers"]
    exit_rule = exit_rule or QExit(exit_threshold, n_loops)
    max_new = per_request(max_new_tokens, len(prompts))
    stats = RunStats(batcher="reference", max_slots=1, exit_threshold=exit_threshold, dtype=dtype)
    outputs, exit_loops, all_logits = [], [], []
    wall, prefill_timer = Timer(), Timer()

    with torch.inference_mode(), wall:
        for req, prompt in enumerate(prompts):
            with prefill_timer:
                pf = prefill(model, prompt, exit_rule, req, record_logits)
            out, loops, logits = [pf.first_token], [pf.exit_loop], [pf.logits]
            stats.prefill_hist[pf.exit_loop - 1] += 1
            pos = len(prompt)
            while len(out) < max_new[req] and out[-1] != EOS_TOKEN_ID:
                tok, u, lg = _decode_token(model, pf.cache, out[-1], pos, exit_rule, req, len(out),
                                           n_loops, n_layers)
                out.append(tok)
                loops.append(u + 1)
                logits.append(lg)
                stats.decode_hist[u] += 1
                stats.block_calls += u + 1
                stats.row_loops += u + 1
                pos += 1
            outputs.append(out)
            exit_loops.append(loops)
            all_logits.append(torch.stack(logits) if record_logits else None)

    stats.wall_s, stats.prefill_s = wall.total, prefill_timer.total
    stats.tokens_generated = sum(len(o) for o in outputs)
    stats.sequences_finished = len(outputs)
    return RunResult(outputs=outputs, exit_loops=exit_loops, stats=stats, steps=[],
                     logits=all_logits if record_logits else None)
