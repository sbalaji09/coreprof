"""Static KV store for mixed-depth decoding (DESIGN_CDB.md §3.1, §4).

K and V are each [caches, layers, slots, kv_heads, max_len, head_dim]. A row in a block call is
(slot, loop, position): it writes its new k/v at that position of its loop's cache and attends to
positions 0..position of the same (cache, slot).

Week 6 KV variants (accuracy only; the values are rounded, the storage stays in the model dtype):
  n_caches  - loops share caches: loop u uses cache u * n_caches // n_loops (4 = one per loop as
              shipped; 2 = loops 1-2 and 3-4; 1 = all loops). A position keeps the KV of the LAST loop
              that wrote it (the token's exit loop, or the last loop of the group after copy-forward);
              the prompt loads the last loop of each group. Each loop still writes its own fresh KV
              before attending, so the token in flight always sees its own current-loop KV.
  share_decode - decode-only sharing (Ouro arXiv 2510.25741 §5.4.2, "last-step only"; results/NOTEBOOK_KV_SHARING.md):
              the prompt keeps one cache per loop; every decode write goes to all caches, so a generated position holds
              the KV of the last loop that wrote it (loop 4 at q=1.0, the exit loop under early exit), as one shared
              cache would. Needs one cache per loop; storage stays 4x (accuracy only).
  kv_quant  - "fp8": e4m3, no scale (clipped at +-448, counted in .clipped); "int4": symmetric, one
              scale per 32 values along head_dim. Applied to every value written (prompt and decode).
"""
import torch

FP8_MAX = 448.0
INT4_GROUP = 32


def fake_quant(x, kind):
    """Round x to the KV format and back to x.dtype. Returns (x, number of fp8 values clipped)."""
    if kind == "none":
        return x, 0
    if kind == "fp8":
        clipped = (x.abs() > FP8_MAX).sum()
        return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype), clipped
    if kind == "int4":
        g = x.float().reshape(*x.shape[:-1], -1, INT4_GROUP)
        scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
        return ((g / scale).round().clamp(-8, 7) * scale).reshape(x.shape).to(x.dtype), 0
    raise ValueError(f"kv_quant must be none, fp8 or int4, got {kind!r}")


class KVStore:
    def __init__(self, n_loops, n_layers, n_slots, kv_heads, max_len, head_dim, dtype, device="cuda",
                 n_caches=None, kv_quant="none", share_decode=False):
        self.n_caches = n_caches or n_loops
        assert 1 <= self.n_caches <= n_loops
        assert not share_decode or self.n_caches == n_loops, "share_decode needs one cache per loop"
        self.share_decode = share_decode
        shape = (self.n_caches, n_layers, n_slots, kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.n_loops, self.n_layers, self.max_len = n_loops, n_layers, max_len
        self.cache_of_loop = [u * self.n_caches // n_loops for u in range(n_loops)]
        self._cache_idx = torch.tensor(self.cache_of_loop, device=device)
        assert kv_quant != "int4" or head_dim % INT4_GROUP == 0, "int4 KV needs head_dim divisible by 32"
        self.kv_quant = kv_quant
        self.clipped = torch.zeros((), dtype=torch.long, device=device)

    def _q(self, x):
        x, n = fake_quant(x, self.kv_quant)
        self.clipped += n
        return x

    @property
    def nbytes(self):
        return 2 * self.k.numel() * self.k.element_size()

    def load_prefill(self, slot: int, hf_cache, prompt_len: int):
        """Copy a B=1 HF UniversalTransformerCache (slot index = loop * n_layers + layer) into `slot`.
        Cache c gets the last loop of its group."""
        assert prompt_len <= self.max_len
        for c in range(self.n_caches):
            u = max(i for i, g in enumerate(self.cache_of_loop) if g == c)
            for l in range(self.n_layers):
                i = u * self.n_layers + l
                self.k[c, l, slot, :, :prompt_len] = self._q(hf_cache.key_cache[i][0])
                self.v[c, l, slot, :, :prompt_len] = self._q(hf_cache.value_cache[i][0])

    def write(self, layer, loop_idx, slot_idx, pos_idx, k_new, v_new):
        """Scatter one new k/v per row. loop_idx/slot_idx/pos_idx: [R] long; k_new/v_new: [R, H, D]."""
        k_new, v_new = self._q(k_new), self._q(v_new)
        if self.share_decode:
            self.k[:, layer][:, slot_idx, :, pos_idx] = k_new[:, None]    # indexed view is [R, caches, H, D]
            self.v[:, layer][:, slot_idx, :, pos_idx] = v_new[:, None]
            return
        c = self._cache_idx[loop_idx]
        self.k[:, layer][c, slot_idx, :, pos_idx] = k_new
        self.v[:, layer][c, slot_idx, :, pos_idx] = v_new

    def read(self, layer, loop_idx, slot_idx, length):
        """Gather each row's (loop, slot) cache, positions [0, length): returns [R, H, length, D] twice."""
        c = self._cache_idx[loop_idx]
        k = self.k[:, layer, :, :, :length][c, slot_idx]
        v = self.v[:, layer, :, :, :length][c, slot_idx]
        return k, v

    def fill_skipped(self, slot_idx, pos_idx, exit_loop_idx):
        """Copy-forward rule (DESIGN_CDB.md §4, option A): a token that exited at loop k (0-based)
        gets its exit cache's k/v copied into every later cache at its position, for every layer.
        All arguments are host lists of equal length (one entry per exiting row)."""
        by_cache = {}
        for s, p, k in zip(slot_idx, pos_idx, exit_loop_idx):
            c = self.cache_of_loop[k]
            if c < self.n_caches - 1:
                by_cache.setdefault(c, ([], []))
                by_cache[c][0].append(s)
                by_cache[c][1].append(p)
        for k, (slots, poss) in by_cache.items():
            s = torch.tensor(slots, device=self.k.device)
            p = torch.tensor(poss, device=self.k.device)
            src_k = self.k[k][:, s, :, p]     # [R, layers, H, D]
            src_v = self.v[k][:, s, :, p]
            for j in range(k + 1, self.n_caches):
                self.k[j][:, s, :, p] = src_k
                self.v[j][:, s, :, p] = src_v


class MixedDepthCache:
    """Stands in for HF's cache inside OuroAttention for one block call.

    Ouro calls update(k, v, current_ut * n_layers + layer_idx). Rows in one call sit at different
    loops, so the layers are called with current_ut=0 and the real per-row (loop, slot, position)
    comes from this object. update() scatters the new k/v and returns each row's padded k/v;
    `attention_mask` hides each row's positions beyond its own length.
    """

    def __init__(self, store: KVStore, loops, slots, positions):
        """loops/slots/positions: host lists, one entry per row."""
        dev = store.k.device
        self.store = store
        self.loop_idx = torch.tensor(loops, device=dev)
        self.slot_idx = torch.tensor(slots, device=dev)
        self.pos_idx = torch.tensor(positions, device=dev)
        self.length = max(positions) + 1
        keep = torch.arange(self.length, device=dev)[None, :] <= self.pos_idx[:, None]
        self.attention_mask = keep[:, None, None, :]           # [R, 1, 1, length], True = attend
        self.position_ids = self.pos_idx[:, None]              # [R, 1]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = layer_idx % self.store.n_layers
        self.store.write(layer, self.loop_idx, self.slot_idx, self.pos_idx,
                         key_states[:, :, 0], value_states[:, :, 0])
        return self.store.read(layer, self.loop_idx, self.slot_idx, self.length)
