# How Ouro-1.4B loops (what the harness times)

All line numbers refer to `modeling_ouro.py` at the pinned model revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`
(https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py).
Locally it is unpacked to
`$HF_HOME/modules/transformers_modules/ByteDance/Ouro-1.4B/574fa66.../modeling_ouro.py`.

## Shape of the model (config.json)

| | |
|---|---|
| layers (`num_hidden_layers`) | 24 |
| hidden / intermediate | 2048 / 5632 (SwiGLU MLP) |
| heads / KV heads / head_dim | 16 / 16 / 128 (plain MHA, no GQA) |
| recurrent steps (`total_ut_steps`) | 4 |
| `early_exit_threshold` | 1.0 |
| attention | SDPA (the default `_attn_implementation`), all layers full attention, no sliding window |

## 1. The loop: `OuroModel.forward`, lines 592–608

```python
for current_ut in range(self.total_ut_steps):          # 4 recurrent ("UT") steps
    for decoder_layer in self.layers[:num_hidden_layers]:   # ALL 24 layers
        hidden_states = decoder_layer(..., current_ut=current_ut)
    hidden_states = self.norm(hidden_states)              # final RMSNorm, every step
    hidden_states_list.append(hidden_states)
    gate_list.append(self.early_exit_gate(hidden_states)) # exit gate, every step
```

- **The recurrent core is the entire 24-layer stack.** There is no
  non-looped prelude or coda: the embedding (line 536) runs once before the
  loop, and `lm_head` runs once after it in `OuroForCausalLM.forward`.
- The weights are shared across steps. Step *t* runs the same 24 layers on
  the output of step *t−1*.
- The final `norm` is applied **inside** the loop (line 606). Step *t+1*
  therefore receives a normalized hidden state. Each step's output is kept
  in `hidden_states_list`.
- Rotary embeddings and the causal mask are computed once per forward pass
  (lines 565–588) and reused by every step and layer.
- Each layer (lines 401–423) uses a "sandwich" norm: pre-norm, then
  attention, then another norm, then the residual add. The MLP block follows
  the same pattern, so there are 4 RMSNorms per layer.

## 2. The exit gate: computed in the loop, used after it

- `early_exit_gate` is `nn.Linear(2048, 1)` (line 513). It produces one
  logit per token per step (line 608).
- The decision is made **after all 4 steps have run**, in
  `OuroForCausalLM.forward` (lines 721–806):
  - λ_t = sigmoid(gate_t).
  - The exit pdf is p_t = λ_t · Π_{s<t}(1−λ_s). The last step takes the
    remaining probability mass.
  - With `exit_threshold` set (it is 1.0 from the config), the code picks
    the first step whose CDF reaches the threshold (lines 780–800). It then
    applies `lm_head` to that step's hidden state.
  - `exit_at_step` (line 775) forces a specific step.
  - `use_weighted_exit` averages the logits of all steps.
- **Consequence for timing:** in the HF implementation the gate never skips
  compute. Every decode token always runs all 4 × 24 layers. Early exit only
  chooses *which* step's hidden state is turned into logits. (The card says
  the same about vLLM.) The "full decode step" timed by the harness is
  therefore embed + 4 steps + 4 gates + exit selection + lm_head, which is
  the real code path.

## 3. The KV cache: one separate cache per (step, layer)

- `OuroAttention.forward` writes to cache slot
  `current_ut * num_hidden_layers + layer_idx` (lines 314–322). Layer 5 of
  step 2 uses slot 2·24+5 = 53.
- The cache is `UniversalTransformerCache` (lines 129–225), with
  `max_cache_size = 24 × 4 = 96` slots (lines 541–549). Each slot is
  `[B, 16, seq, 128]`.
- **Each step attends only to its own step's past keys and values.** The
  token at step *t* sees the step-*t* KV of all earlier tokens, not the KV
  of other steps. A probe confirmed this: slots 0 and 24 hold different
  values.
- Memory is therefore 4× a normal 24-layer model:
  `96 × 2 (K,V) × B × 16 × C × 128 × 2 bytes` = **B·C·0.79 MB** in bf16.
  At C=512 that is 0.4 GB/sequence: B=32 → 12.9 GB, **B=64 → 25.8 GB,
  which does not fit on the 24 GB A10.**
- **The cache grows with `torch.cat`** (lines 177–178). Every decode token,
  every slot is reallocated and the whole past KV is copied
  (read C rows, write C+1 rows). That copy costs real memory bandwidth on
  top of attention's own read of the cache. It is a property of this HF
  implementation, not of looped models in general: a preallocated or paged
  cache (e.g. vLLM) would not pay it. The harness times the real code, so
  this cost **is included** in the numbers.
- For decode (1 new token, no padding) `create_causal_mask` returns `None`,
  so SDPA runs unmasked (verified with a probe).

## 4. What the harness times

**`ut_step`, one recurrent step.** This is exactly one iteration of the loop
body above (lines 593–608): 24 decoder layers, then `norm`, then
`early_exit_gate`. It is applied to a single new token per sequence with
cache slots for that step already filled to context length C. It is called
directly on `model.model.layers` with the same `position_embeddings`, mask,
`cache_position` and `current_ut` arguments that `OuroModel.forward` would
pass.
- The input hidden state is the real one: the embedding of the next token,
  passed through the earlier steps (computed once, untimed).
- By default step 0 is timed (`--ut-step` changes it). All steps do the same
  work at equal C.

**`decode_step`, one full token.** This is `OuroForCausalLM.forward` on one
token with the prefilled cache and `logits_to_keep=1`. It includes
embedding, rotary/mask setup, all 4 steps, the exit-gate selection and
`lm_head`, the same call `generate()` makes each step, minus sampling.

**Resetting state between repetitions.** Every timed call appends one token
to the cache. Before each repetition, untimed, the slots are sliced back to
length C (a view, no copy). Every repetition therefore sees exactly C past
tokens and pays the same `torch.cat` cost that real decode at length C pays.
