"""Pieces shared by every batcher: model loading, the Q-exit rule, prefill, the
mixed-depth block call, and the result/stats types. See DESIGN_CDB.md."""
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM

MODEL = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
EOS_TOKEN_ID = 0


# ---------------------------------------------------------------- model

def set_numerics():
    """fp32 must be real fp32: no TF32 shortcuts. (TF32 only affects fp32 matmuls, so this is a
    no-op for the bf16 model's main GEMMs.)"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@lru_cache(maxsize=1)
def get_model(dtype: str):
    """Load Ouro once per process (one dtype at a time, to bound GPU memory)."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=DTYPES[dtype], trust_remote_code=True
    )
    return model.cuda().eval()


def model_dims(model):
    cfg = model.config
    return dict(n_loops=model.model.total_ut_steps, n_layers=cfg.num_hidden_layers,
                kv_heads=cfg.num_key_value_heads,
                head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))


# ---------------------------------------------------------------- exit rule

class QExit:
    """Ouro's Q-exit criterion, decided per token (paper §3, "Inference with Early Exit").

    After loop u (0-based) with gate logit g: lambda = sigmoid(g), survival *= (1 - lambda),
    CDF = 1 - survival; exit if CDF >= q, and always at the last loop. Computed in fp32.
    q >= 1 never exits early (the HF/config default).

    Called as rule(gate, survival, loop, req_ids, tok_idx) -> (exit[R] bool, survival[R]).
    req_ids / tok_idx are host lists; the default rule ignores them (scripted test rules use them).
    """

    def __init__(self, q: float, n_loops: int = 4):
        self.q, self.n_loops = q, n_loops

    def __call__(self, gate, survival, loop, req_ids, tok_idx):
        lam = torch.sigmoid(gate.reshape(-1).float())
        survival = survival * (1.0 - lam)
        last = loop == self.n_loops - 1
        if self.q >= 1.0:
            return last, survival
        return last | ((1.0 - survival) >= self.q), survival


ExitRule = Callable[..., tuple]


# ---------------------------------------------------------------- results

@dataclass
class StepRecord:
    """One block call. Built from host data that already exists after the exit sync."""
    call_idx: int
    decode_step: Optional[int]          # baseline only: which depth-synchronous step
    n_rows: int
    max_slots: int
    rows_per_loop: list                 # [n at loop 1, ..., n at loop 4]
    n_exits: int
    n_refills: int
    n_finished: int
    call_ms: Optional[float] = None     # only with time_calls=True
    # Host-side bookkeeping for Week 3 windowing; perf_counter reads only, no extra syncs.
    queue_len: int = 0                  # requests still queued when the call started
    prefill_s_before: float = 0.0       # cumulative prefill time when the call started
    t_end: float = 0.0                  # perf_counter after the call's last host sync


@dataclass
class RunStats:
    batcher: str
    max_slots: int
    exit_threshold: float
    dtype: str
    tokens_generated: int = 0
    sequences_finished: int = 0
    block_calls: int = 0
    row_loops: int = 0
    decode_steps: Optional[int] = None
    wall_s: float = 0.0
    prefill_s: float = 0.0
    decode_hist: list = field(default_factory=lambda: [0, 0, 0, 0])   # exit loop 1..4, decode tokens
    prefill_hist: list = field(default_factory=lambda: [0, 0, 0, 0])  # first token of each request

    @property
    def decode_s(self):
        return self.wall_s - self.prefill_s

    @property
    def slot_utilization(self):
        return self.row_loops / max(1, self.block_calls * self.max_slots)

    def summary(self):
        n = max(1, sum(self.decode_hist))
        hist = " / ".join(f"{100 * c / n:.1f}" for c in self.decode_hist)
        return (f"{self.batcher}: {self.tokens_generated} tokens, {self.sequences_finished} seqs, "
                f"{self.block_calls} block calls, utilization {100 * self.slot_utilization:.1f}%, "
                f"decode {self.decode_s:.2f} s + prefill {self.prefill_s:.2f} s, "
                f"exit loops 1/2/3/4 = {hist} %")


@dataclass
class RunResult:
    outputs: list        # per request: generated token ids (first one comes from prefill)
    exit_loops: list     # per request: 1-based exit loop of each generated token
    stats: RunStats
    steps: list          # list[StepRecord]
    logits: Optional[list] = None   # per request: [n_tokens, vocab] fp32, only if record_logits


def per_request(max_new_tokens: Union[int, Sequence[int]], n: int) -> list:
    if isinstance(max_new_tokens, int):
        return [max_new_tokens] * n
    assert len(max_new_tokens) == n, "one max_new_tokens per prompt"
    return list(max_new_tokens)


def warn_if_no_spread(stats: RunStats):
    n = sum(stats.decode_hist)
    if n and stats.decode_hist[-1] / n > 0.95:
        print(f"WARNING: {100 * stats.decode_hist[-1] / n:.1f}% of decode tokens ran all loops at "
              f"q={stats.exit_threshold}; CDB has almost nothing to gain at this threshold.")


# ---------------------------------------------------------------- prefill

@dataclass
class Prefilled:
    cache: object         # HF UniversalTransformerCache, n_loops * n_layers slots of [1, H, P, D]
    prompt_len: int
    first_token: int
    exit_loop: int        # 1-based
    logits: Optional[torch.Tensor]


def prefill(model, prompt: Sequence[int], exit_rule: ExitRule, req_id: int, record_logits=False):
    """Run the prompt alone (B=1) through all loops with HF's own cache (paper: prefill needs every
    loop's KV). The first generated token uses the Q-exit rule on the last prompt position."""
    ids = torch.tensor([list(prompt)], device="cuda")
    out, hidden_list, gate_list = model.model(input_ids=ids, use_cache=True)
    survival = torch.ones(1, device="cuda")
    for u, gate in enumerate(gate_list):
        loop = torch.full((1,), u, device="cuda")
        exit_now, survival = exit_rule(gate[:, -1], survival, loop, [req_id], [0])
        if exit_now.item():
            break
    logits = model.lm_head(hidden_list[u][:, -1]).float()
    return Prefilled(cache=out.past_key_values, prompt_len=len(prompt),
                     first_token=int(logits.argmax(-1).item()), exit_loop=u + 1,
                     logits=logits[0] if record_logits else None)


# ---------------------------------------------------------------- the looped block

def run_block(model, hidden, position_ids, cache, attention_mask, current_ut):
    """One loop of Ouro's recurrent body (modeling_ouro.py:593-608): all layers, norm, exit gate."""
    m = model.model
    position_embeddings = m.rotary_emb(hidden, position_ids)
    for layer in m.layers[: m.config.num_hidden_layers]:
        hidden = layer(hidden, attention_mask=attention_mask, position_ids=position_ids,
                       past_key_value=cache, use_cache=True, cache_position=None,
                       position_embeddings=position_embeddings, current_ut=current_ut)
    hidden = m.norm(hidden)
    return hidden, m.early_exit_gate(hidden)


class Timer:
    """Wall-clock section timer that synchronizes so GPU work is attributed correctly."""

    def __init__(self):
        self.total = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.total += time.perf_counter() - self._t0
