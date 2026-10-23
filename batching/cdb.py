"""Continuous Depth Batching (DESIGN_CDB.md §2; "Relaxed Recursive Transformers" §2.4).

Every block call runs all occupied slots, each at its own loop depth. A row that exits emits its
token and its sequence's next token enters the very next call at loop 1; a finished sequence's slot
is refilled from the queue right away. No slot waits for slower rows.
"""
import torch

from .common import Timer, get_model, set_numerics, warn_if_no_spread
from .engine import Engine


def run(prompts, max_new_tokens, max_slots, exit_threshold, dtype, *, model=None, exit_rule=None,
        record_logits=False, time_calls=False):
    set_numerics()
    model = model or get_model(dtype)
    eng = Engine(model, prompts, max_new_tokens, max_slots, exit_threshold, dtype, "cdb",
                 exit_rule, record_logits, time_calls)
    wall = Timer()
    with torch.inference_mode(), wall:
        while True:
            eng.refill_free_slots()                    # after every block call
            rows = eng.active()
            if not rows:
                break
            eng.call(rows)
    result = eng.result(wall)
    warn_if_no_spread(result.stats)
    return result
