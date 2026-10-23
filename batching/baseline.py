"""Baseline: depth-synchronous decode steps (DESIGN_CDB.md §2, decisions D2/D3).

Each decode step takes one token from every occupied slot and runs loop 1 on all of them, then
loop 2 on those that have not exited, and so on, until the slowest token exits. Exited rows wait
(they are left out of later block calls in the step). Only then does the next step start; finished
sequences are replaced from the queue at step boundaries.
"""
import torch

from .common import Timer, get_model, set_numerics, warn_if_no_spread
from .engine import Engine


def run(prompts, max_new_tokens, max_slots, exit_threshold, dtype, *, model=None, exit_rule=None,
        record_logits=False, time_calls=False):
    set_numerics()
    model = model or get_model(dtype)
    eng = Engine(model, prompts, max_new_tokens, max_slots, exit_threshold, dtype, "baseline",
                 exit_rule, record_logits, time_calls)
    wall = Timer()
    step = 0
    with torch.inference_mode(), wall:
        while True:
            eng.refill_free_slots()                    # only at step boundaries
            pending = eng.active()
            if not pending:
                break
            while pending:                             # loops 1, 2, ... until every row has exited
                exits = eng.call(pending, decode_step=step)
                pending = [s for s, e in zip(pending, exits) if not e]
            step += 1
    eng.stats.decode_steps = step
    result = eng.result(wall)
    warn_if_no_spread(result.stats)
    return result
