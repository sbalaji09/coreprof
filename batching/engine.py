"""Slot bookkeeping and the mixed-depth block call shared by baseline.py and cdb.py.

The two batchers differ only in *which* active rows they put into each block call and *when*
free slots are refilled; everything numerical happens here, so it is identical for both.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

from .common import (EOS_TOKEN_ID, QExit, RunResult, RunStats, StepRecord, Timer, model_dims,
                     per_request, prefill, run_block)
from .kv_store import KVStore, MixedDepthCache


@dataclass
class Slot:
    req: int        # request id (index into prompts)
    pos: int        # position of the token in flight
    loop: int       # 0-based loop the token in flight runs next
    tok_idx: int    # index (in the output) of the token being produced


class Engine:
    def __init__(self, model, prompts, max_new_tokens, max_slots, exit_threshold, dtype, batcher,
                 exit_rule=None, record_logits=False, time_calls=False):
        dims = model_dims(model)
        self.model, self.prompts = model, [list(p) for p in prompts]
        self.max_new = per_request(max_new_tokens, len(prompts))
        self.n_loops = dims["n_loops"]
        self.exit_rule = exit_rule or QExit(exit_threshold, self.n_loops)
        self.record_logits, self.time_calls = record_logits, time_calls
        max_len = max(len(p) for p in self.prompts) + max(self.max_new)
        self.store = KVStore(dims["n_loops"], dims["n_layers"], max_slots, dims["kv_heads"],
                             max_len, dims["head_dim"], model.dtype)
        self.hidden = torch.zeros(max_slots, 1, model.config.hidden_size, dtype=model.dtype, device="cuda")
        self.survival = torch.ones(max_slots, device="cuda")
        self.slots: list[Optional[Slot]] = [None] * max_slots
        self.queue = deque(range(len(prompts)))
        self.outputs = [[] for _ in prompts]
        self.exit_loops = [[] for _ in prompts]
        self.logits = [[] for _ in prompts] if record_logits else None
        self.stats = RunStats(batcher=batcher, max_slots=max_slots, exit_threshold=exit_threshold, dtype=dtype)
        self.steps: list[StepRecord] = []
        self.prefill_timer = Timer()
        self._refills = 0      # admissions since the last block call (goes into the next StepRecord)
        self._finished = 0     # sequences finished at prefill since the last block call

    # ------------------------------------------------------------ requests and slots

    def done(self, req):
        out = self.outputs[req]
        return len(out) >= self.max_new[req] or out[-1] == EOS_TOKEN_ID

    def emit(self, req, token, exit_loop, logits=None):
        self.outputs[req].append(token)
        self.exit_loops[req].append(exit_loop)
        if self.record_logits:
            self.logits[req].append(logits)

    def admit(self, slot) -> bool:
        """Prefill queued requests into a free slot until one needs decoding. False if queue empty."""
        while self.queue:
            req = self.queue.popleft()
            with self.prefill_timer:
                pf = prefill(self.model, self.prompts[req], self.exit_rule, req, self.record_logits)
            self.emit(req, pf.first_token, pf.exit_loop, pf.logits)
            self.stats.prefill_hist[pf.exit_loop - 1] += 1
            if self.done(req):            # max_new_tokens == 1, or EOS straight away
                self._finished += 1
                continue
            self.store.load_prefill(slot, pf.cache, pf.prompt_len)
            self.slots[slot] = Slot(req=req, pos=pf.prompt_len, loop=0, tok_idx=1)
            self._start_token(slot, pf.first_token)
            self._refills += 1
            return True
        return False

    def refill_free_slots(self):
        for s, state in enumerate(self.slots):
            if state is None and not self.admit(s):
                break

    def active(self):
        return [s for s, state in enumerate(self.slots) if state is not None]

    def _start_token(self, slot, token):
        """The sequence's next token enters at loop 0 with its embedding as hidden state."""
        emb = self.model.model.embed_tokens(torch.tensor([[token]], device="cuda"))
        self.hidden[slot] = emb[0]
        self.survival[slot] = 1.0
        self.slots[slot].loop = 0

    # ------------------------------------------------------------ one block call

    def call(self, rows, decode_step=None):
        """Run one loop for each row (slot) at its own depth. Returns the host list of exit flags."""
        S = self.slots
        loops = [S[s].loop for s in rows]
        timer = Timer() if self.time_calls else None
        if timer:
            timer.__enter__()

        cache = MixedDepthCache(self.store, loops, rows, [S[s].pos for s in rows])
        idx = torch.tensor(rows, device="cuda")
        h, gate = run_block(self.model, self.hidden[idx], cache.position_ids, cache,
                            cache.attention_mask, current_ut=0)
        exit_now, survival = self.exit_rule(gate, self.survival[idx], cache.loop_idx,
                                            [S[s].req for s in rows], [S[s].tok_idx for s in rows])
        self.hidden[idx] = h
        self.survival[idx] = survival
        exits = exit_now.tolist()                      # forced sync #1: who exits

        ex = [i for i, e in enumerate(exits) if e]
        if ex:
            logits = self.model.lm_head(h[torch.tensor(ex, device="cuda"), 0]).float()
            tokens = logits.argmax(-1).tolist()        # sync #2: the tokens (EOS check, next input)
            self.store.fill_skipped([rows[i] for i in ex], [S[rows[i]].pos for i in ex],
                                    [loops[i] for i in ex])

        if timer:
            timer.__exit__()

        finished = self._finished
        for j, i in enumerate(ex):
            s = rows[i]
            st = S[s]
            self.emit(st.req, tokens[j], loops[i] + 1, logits[j] if self.record_logits else None)
            self.stats.decode_hist[loops[i]] += 1
            if self.done(st.req):
                S[s] = None
                finished += 1
            else:
                st.pos += 1
                st.tok_idx += 1
                self._start_token(s, tokens[j])
        for i, e in enumerate(exits):
            if not e:
                S[rows[i]].loop += 1

        per_loop = [0] * self.n_loops
        for u in loops:
            per_loop[u] += 1
        self.steps.append(StepRecord(
            call_idx=len(self.steps), decode_step=decode_step, n_rows=len(rows),
            max_slots=len(S), rows_per_loop=per_loop, n_exits=len(ex), n_refills=self._refills,
            n_finished=finished, call_ms=timer.total * 1e3 if timer else None))
        self._refills = self._finished = 0
        self.stats.block_calls += 1
        self.stats.row_loops += len(rows)
        return exits

    # ------------------------------------------------------------ result

    def result(self, wall: Timer) -> RunResult:
        st = self.stats
        st.wall_s, st.prefill_s = wall.total, self.prefill_timer.total
        st.tokens_generated = sum(len(o) for o in self.outputs)
        st.sequences_finished = sum(1 for r in range(len(self.outputs)) if self.outputs[r] and self.done(r))
        logits = [torch.stack(l) for l in self.logits] if self.record_logits else None
        return RunResult(outputs=self.outputs, exit_loops=self.exit_loops, stats=st, steps=self.steps,
                         logits=logits)
