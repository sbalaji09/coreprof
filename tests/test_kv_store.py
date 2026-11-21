"""KVStore scatter / gather / copy-forward / prefill load, checked against plain Python indexing."""
import pytest
import torch

from batching.kv_store import KVStore, MixedDepthCache

U, L, S, H, T, D = 4, 3, 5, 2, 10, 4


@pytest.fixture
def store():
    torch.manual_seed(0)
    st = KVStore(U, L, S, H, T, D, torch.float32)
    st.k.normal_()
    st.v.normal_()
    return st


def test_write_scatters_only_target_positions(store):
    before_k, before_v = store.k.clone(), store.v.clone()
    loops, slots, poss = [0, 3, 1], [4, 0, 2], [7, 0, 9]
    k_new, v_new = torch.randn(3, H, D, device="cuda"), torch.randn(3, H, D, device="cuda")
    store.write(1, torch.tensor(loops, device="cuda"), torch.tensor(slots, device="cuda"),
                torch.tensor(poss, device="cuda"), k_new, v_new)
    for r, (u, s, p) in enumerate(zip(loops, slots, poss)):
        assert torch.equal(store.k[u, 1, s, :, p], k_new[r])
        assert torch.equal(store.v[u, 1, s, :, p], v_new[r])
        before_k[u, 1, s, :, p] = k_new[r]
        before_v[u, 1, s, :, p] = v_new[r]
    assert torch.equal(store.k, before_k) and torch.equal(store.v, before_v)   # nothing else moved


def test_read_gathers_each_rows_loop_and_slot(store):
    loops, slots = [2, 0, 3], [1, 1, 4]
    k, v = store.read(2, torch.tensor(loops, device="cuda"), torch.tensor(slots, device="cuda"), 6)
    assert k.shape == (3, H, 6, D)
    for r, (u, s) in enumerate(zip(loops, slots)):
        assert torch.equal(k[r], store.k[u, 2, s, :, :6])
        assert torch.equal(v[r], store.v[u, 2, s, :, :6])


def test_mixed_depth_cache_update_and_mask(store):
    loops, slots, poss = [0, 2], [3, 1], [4, 1]
    cache = MixedDepthCache(store, loops, slots, poss)
    assert cache.length == 5
    expected_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(cache.attention_mask[:, 0, 0].cpu(), expected_mask)
    k_new = torch.randn(2, H, 1, D, device="cuda")      # Ouro passes [R, H, 1, D]
    v_new = torch.randn(2, H, 1, D, device="cuda")
    k, v = cache.update(k_new, v_new, layer_idx=2 * L + 1)  # current_ut * n_layers + layer -> layer 1
    for r, (u, s, p) in enumerate(zip(loops, slots, poss)):
        assert torch.equal(store.k[u, 1, s, :, p], k_new[r, :, 0])
        assert torch.equal(k[r, :, p], k_new[r, :, 0])        # the row sees its own new key
        assert torch.equal(v[r], store.v[u, 1, s, :, :5])


@pytest.mark.parametrize("exit_loop", [0, 1, 2, 3])
def test_fill_skipped_copies_exit_loop_forward(store, exit_loop):
    before = store.k.clone()
    store.fill_skipped([2, 4], [5, 0], [exit_loop, exit_loop])
    expected = before.clone()
    for s, p in [(2, 5), (4, 0)]:
        for j in range(exit_loop + 1, U):
            expected[j, :, s, :, p] = before[exit_loop, :, s, :, p]
    assert torch.equal(store.k, expected)          # deeper loops copied, everything else untouched


def test_fill_skipped_mixed_exit_loops(store):
    before_v = store.v.clone()
    store.fill_skipped([0, 1, 3], [2, 2, 9], [0, 2, 3])
    for j in range(1, U):
        assert torch.equal(store.v[j, :, 0, :, 2], before_v[0, :, 0, :, 2])
    assert torch.equal(store.v[3, :, 1, :, 2], before_v[2, :, 1, :, 2])
    assert torch.equal(store.v[:3, :, 1, :, 2], before_v[:3, :, 1, :, 2])
    assert torch.equal(store.v[:, :, 3], before_v[:, :, 3])          # exit at last loop: no-op


def test_load_prefill_matches_hf_layout(store):
    class FakeHFCache:      # UniversalTransformerCache: list index = loop * n_layers + layer
        key_cache = [torch.randn(1, H, 6, D, device="cuda") for _ in range(U * L)]
        value_cache = [torch.randn(1, H, 6, D, device="cuda") for _ in range(U * L)]

    rest = store.k[:, :, 3, :, 6:].clone()
    store.load_prefill(3, FakeHFCache, 6)
    for u in range(U):
        for l in range(L):
            assert torch.equal(store.k[u, l, 3, :, :6], FakeHFCache.key_cache[u * L + l][0])
            assert torch.equal(store.v[u, l, 3, :, :6], FakeHFCache.value_cache[u * L + l][0])
    assert torch.equal(store.k[:, :, 3, :, 6:], rest)


# ---------------------------------------------------------------- Week 6 variants (run on CPU too)

from batching.kv_store import fake_quant  # noqa: E402


def _cpu_store(n_caches, kv_quant="none"):
    torch.manual_seed(0)
    return KVStore(U, L, S, H, T, D, torch.float32, device="cpu", n_caches=n_caches, kv_quant=kv_quant)


@pytest.mark.parametrize("n_caches, groups", [(4, [0, 1, 2, 3]), (2, [0, 0, 1, 1]), (1, [0, 0, 0, 0])])
def test_loops_map_to_cache_groups(n_caches, groups):
    st = _cpu_store(n_caches)
    assert st.cache_of_loop == groups and st.k.shape[0] == n_caches


def test_shared_cache_keeps_last_write_and_each_loop_sees_it():
    st = _cpu_store(1)
    one = lambda x: torch.tensor([x])
    for u in range(U):                       # one token running loops 0..3 at position 2
        k_new = torch.full((1, H, D), float(u))
        st.write(0, one(u), one(1), one(2), k_new, k_new)
        k, _ = st.read(0, one(u), one(1), 3)
        assert torch.equal(k[0, :, 2], k_new[0])        # the loop in flight sees its own fresh KV
    assert torch.equal(st.k[0, 0, 1, :, 2], torch.full((H, D), 3.0))   # last loop's KV stays


def test_pairs_copy_forward_from_exit_group():
    st = _cpu_store(2)
    st.k.normal_()
    before = st.k.clone()
    st.fill_skipped([3], [4], [1])            # exit at loop 2 (group 0): group 1 gets group 0's KV
    assert torch.equal(st.k[1, :, 3, :, 4], before[0, :, 3, :, 4])
    st.fill_skipped([2], [5], [2])            # exit in the last group: nothing to fill
    assert torch.equal(st.k[:, :, 2], before[:, :, 2])


def test_prefill_loads_last_loop_of_each_group():
    class FakeHFCache:
        key_cache = [torch.full((1, H, 6, D), float(i)) for i in range(U * L)]
        value_cache = key_cache
    st = _cpu_store(2)
    st.load_prefill(0, FakeHFCache, 6)
    for c, u in ((0, 1), (1, 3)):
        for l in range(L):
            assert torch.equal(st.k[c, l, 0, :, :6], FakeHFCache.key_cache[u * L + l][0])


def test_fp8_rounds_and_counts_clipping():
    x = torch.tensor([0.1234, -3.3, 500.0, -1000.0])
    y, n = fake_quant(x, "fp8")
    assert int(n) == 2 and y[2] == 448 and y[3] == -448
    assert (y[:2] - x[:2]).abs().max() <= 0.07 * x[:2].abs().max()        # e4m3: 3 mantissa bits
    assert torch.equal(fake_quant(y, "fp8")[0], y)                       # idempotent


def test_int4_groups_of_32_and_error_bound():
    torch.manual_seed(0)
    x = torch.randn(3, H, 2 * 32)
    y, _ = fake_quant(x, "int4")
    g, yg = x.reshape(3, H, 2, 32), y.reshape(3, H, 2, 32)
    scale = g.abs().amax(-1, keepdim=True) / 7
    assert ((yg - g).abs() <= scale / 2 + 1e-6).all()                    # round-to-nearest
    assert len(torch.unique(yg[0, 0, 0])) <= 16


def test_quantization_applies_to_writes():
    st = KVStore(U, L, S, H, T, 64, torch.float32, device="cpu", kv_quant="int4")
    one = lambda x: torch.tensor([x])
    k_new = torch.randn(1, H, 64)
    st.write(0, one(0), one(0), one(0), k_new, k_new)
    assert torch.equal(st.k[0, 0, 0, :, 0], fake_quant(k_new, "int4")[0][0])
