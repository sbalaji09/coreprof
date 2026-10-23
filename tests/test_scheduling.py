"""Slot refill / scheduling edge cases for baseline and CDB, on a tiny fp32 Ouro.

Exits are scripted per (request, token) so each edge case happens deterministically. Every run is
also checked against the B=1 reference: same tokens and same per-token exit loops.
"""
import random

import pytest
import torch

import batching.engine
import batching.reference
from batching import baseline, cdb, reference
from batching.common import QExit

BATCHERS = {"baseline": baseline, "cdb": cdb}


class ScriptedExit:
    """Exit token `tok_idx` of request `req` after loop exit_at(req, tok_idx) (0-based)."""

    def __init__(self, exit_at):
        self.exit_at = exit_at

    def __call__(self, gate, survival, loop, req_ids, tok_idx):
        target = torch.tensor([self.exit_at(r, t) for r, t in zip(req_ids, tok_idx)], device=loop.device)
        return loop >= target, survival


@pytest.fixture
def no_eos(monkeypatch):
    """Random weights can emit token 0 (EOS); disable EOS so sequence lengths are exactly as set."""
    monkeypatch.setattr(batching.engine, "EOS_TOKEN_ID", -1)
    monkeypatch.setattr(batching.reference, "EOS_TOKEN_ID", -1)


def make_prompts(n, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(1, 256) for _ in range(rng.randrange(3, 12))] for _ in range(n)]


def run_all(model, prompts, max_new, slots, rule, q=1.0):
    kw = dict(model=model, exit_rule=rule)
    ref = reference.run(prompts, max_new, 1, q, "fp32", **kw)
    out = {name: mod.run(prompts, max_new, slots, q, "fp32", **kw) for name, mod in BATCHERS.items()}
    for name, res in out.items():
        assert res.outputs == ref.outputs, f"{name} tokens differ from reference"
        assert res.exit_loops == ref.exit_loops, f"{name} exit loops differ from reference"
        check_invariants(res)
    return ref, out


def check_invariants(res):
    st = res.stats
    assert sum(sum(s.rows_per_loop) for s in res.steps) == st.row_loops
    assert st.block_calls == len(res.steps)
    assert all(s.n_rows <= s.max_slots for s in res.steps)
    assert sum(st.decode_hist) + sum(st.prefill_hist) == st.tokens_generated
    if st.batcher == "baseline":   # within a step: every row at the same loop, rows only drop out
        by_step = {}
        for s in res.steps:
            by_step.setdefault(s.decode_step, []).append(s)
        for calls in by_step.values():
            for u, s in enumerate(calls):
                assert s.rows_per_loop[u] == s.n_rows
            assert all(a.n_rows >= b.n_rows for a, b in zip(calls, calls[1:]))


def test_all_slots_exit_in_same_step(tiny_model, no_eos):
    prompts = make_prompts(4)
    ref, out = run_all(tiny_model, prompts, 5, 4, ScriptedExit(lambda r, t: 1))
    for s in out["cdb"].steps:          # everyone moves in lockstep: one depth per call
        assert sum(1 for n in s.rows_per_loop if n) == 1
    assert out["cdb"].stats.block_calls == out["baseline"].stats.block_calls   # nothing for CDB to gain


def test_sequence_ends_while_others_mid_loop(tiny_model, no_eos):
    # req 0 exits at loop 1 and has 2 tokens, so it finishes on the first block call while the
    # others (exit at loop 4) are still mid-token. Request 3 waits in the queue.
    prompts = make_prompts(4, seed=1)
    rule = ScriptedExit(lambda r, t: 0 if r == 0 else 3)
    ref, out = run_all(tiny_model, prompts, [2, 6, 6, 4], 3, rule)
    c = out["cdb"].steps
    assert c[0].n_finished == 1
    assert c[1].n_refills == 1 and c[1].rows_per_loop == [1, 2, 0, 0]   # new request at loop 1, others at 2
    b = out["baseline"].steps
    assert b[1].n_refills == 0 and b[1].n_rows == 2      # baseline refills only at the step boundary
    first_refill = next(s for s in b[1:] if s.n_refills)       # call 0's refills are the initial fill
    assert first_refill.decode_step == 1 and first_refill.rows_per_loop[0] == first_refill.n_rows
    # (No block-call comparison here: requests 1-2 need 5 tokens x 4 loops = 20 sequential calls
    # in either batcher, so the critical path, not slot reuse, sets the count.)


def test_queue_runs_empty(tiny_model, no_eos):
    prompts = make_prompts(5, seed=2)
    rng = random.Random(3)
    table = {(r, t): rng.randrange(4) for r in range(5) for t in range(8)}
    max_new = [3, 1, 4, 2, 5]           # request 1 finishes at prefill and never occupies a slot
    ref, out = run_all(tiny_model, prompts, max_new, 2, ScriptedExit(lambda r, t: table[r, t]))
    for res in out.values():
        assert [len(o) for o in res.outputs] == max_new
        assert res.stats.sequences_finished == 5
    rows = [s.n_rows for s in out["cdb"].steps]
    first_drop = next(i for i, n in enumerate(rows) if n < 2)
    assert all(n < 2 for n in rows[first_drop:])     # once the queue is empty, slots never refill


def test_more_slots_than_requests(tiny_model, no_eos):
    prompts = make_prompts(2, seed=4)
    run_all(tiny_model, prompts, [4, 7], 6, ScriptedExit(lambda r, t: (r + t) % 4))


def test_loop1_exits_copy_kv_forward(tiny_model, no_eos):
    """Every token exits at loop 1: loops 2-4 run on copied KV only (never occurs naturally)."""
    run_all(tiny_model, make_prompts(3, seed=5), 6, 2, ScriptedExit(lambda r, t: 0))


@pytest.mark.parametrize("q", [0.3, 0.5, 0.9, 1.0])
def test_natural_qexit_matches_reference(tiny_model, no_eos, q):
    run_all(tiny_model, make_prompts(5, seed=6), 6, 3, QExit(q), q=q)


def test_eos_ends_sequence(tiny_model, monkeypatch):
    prompts = make_prompts(3, seed=7)
    rule = ScriptedExit(lambda r, t: t % 4)
    monkeypatch.setattr(batching.engine, "EOS_TOKEN_ID", -1)
    monkeypatch.setattr(batching.reference, "EOS_TOKEN_ID", -1)
    free = reference.run(prompts, 8, 1, 1.0, "fp32", model=tiny_model, exit_rule=rule)
    eos = free.outputs[0][2]                     # pretend the 3rd token of request 0 is EOS
    monkeypatch.setattr(batching.engine, "EOS_TOKEN_ID", eos)
    monkeypatch.setattr(batching.reference, "EOS_TOKEN_ID", eos)
    ref, out = run_all(tiny_model, prompts, 8, 2, rule)
    assert ref.outputs[0] == free.outputs[0][:3]
    for res in out.values():
        assert res.outputs[0][-1] == eos and len(res.outputs[0]) == 3
