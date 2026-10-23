"""Static KV store for mixed-depth decoding (DESIGN_CDB.md §3.1, §4).

K and V are each [loops, layers, slots, kv_heads, max_len, head_dim]. A row in a block call is
(slot, loop, position): it writes its new k/v at that position of its own loop's cache and
attends to positions 0..position of the same (loop, slot).
"""
import torch


class KVStore:
    def __init__(self, n_loops, n_layers, n_slots, kv_heads, max_len, head_dim, dtype, device="cuda"):
        shape = (n_loops, n_layers, n_slots, kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.n_loops, self.n_layers, self.max_len = n_loops, n_layers, max_len

    @property
    def nbytes(self):
        return 2 * self.k.numel() * self.k.element_size()

    def load_prefill(self, slot: int, hf_cache, prompt_len: int):
        """Copy a B=1 HF UniversalTransformerCache (slot index = loop * n_layers + layer) into `slot`."""
        assert prompt_len <= self.max_len
        for u in range(self.n_loops):
            for l in range(self.n_layers):
                i = u * self.n_layers + l
                self.k[u, l, slot, :, :prompt_len] = hf_cache.key_cache[i][0]
                self.v[u, l, slot, :, :prompt_len] = hf_cache.value_cache[i][0]

    def write(self, layer, loop_idx, slot_idx, pos_idx, k_new, v_new):
        """Scatter one new k/v per row. loop_idx/slot_idx/pos_idx: [R] long; k_new/v_new: [R, H, D]."""
        self.k[:, layer][loop_idx, slot_idx, :, pos_idx] = k_new
        self.v[:, layer][loop_idx, slot_idx, :, pos_idx] = v_new

    def read(self, layer, loop_idx, slot_idx, length):
        """Gather each row's (loop, slot) cache, positions [0, length): returns [R, H, length, D] twice."""
        k = self.k[:, layer, :, :, :length][loop_idx, slot_idx]
        v = self.v[:, layer, :, :, :length][loop_idx, slot_idx]
        return k, v

    def fill_skipped(self, slot_idx, pos_idx, exit_loop_idx):
        """Copy-forward rule (DESIGN_CDB.md §4, option A): a token that exited at loop k (0-based)
        gets loop k's k/v copied into loops k+1..last at its position, for every layer.
        All arguments are host lists of equal length (one entry per exiting row)."""
        by_loop = {}
        for s, p, k in zip(slot_idx, pos_idx, exit_loop_idx):
            if k < self.n_loops - 1:
                by_loop.setdefault(k, ([], []))
                by_loop[k][0].append(s)
                by_loop[k][1].append(p)
        for k, (slots, poss) in by_loop.items():
            s = torch.tensor(slots, device=self.k.device)
            p = torch.tensor(poss, device=self.k.device)
            src_k = self.k[k][:, s, :, p]     # [R, layers, H, D]
            src_v = self.v[k][:, s, :, p]
            for j in range(k + 1, self.n_loops):
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
