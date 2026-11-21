"""train_looped.py model checks on CPU (tiny dims): causality and that each KV mode does what it says."""
import pytest
import torch

import train_looped as tl


def tiny(variant):
    torch.manual_seed(0)
    cfg = dict(tl.VARIANTS[variant])
    cfg["n_kv_heads"] = 1 if cfg["n_kv_heads"] == 3 else 4
    return tl.LoopedLM(**cfg, d=32, h=4, layers=2, ff=64, ctx=16).eval()


@pytest.mark.parametrize("variant", list(tl.VARIANTS))
def test_causal(variant):
    m = tiny(variant)
    x = torch.randint(0, tl.VOCAB, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, tl.VOCAB, (2, 5))
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)          # positions < 7 never see the change
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_sharefirst_later_loops_ignore_their_own_kv_projection():
    m, x = tiny("sharefirst"), torch.randint(0, tl.VOCAB, (1, 10))
    with torch.no_grad():
        ref = m(x)
        calls = []
        for layer in m.layers:
            layer.kv.register_forward_hook(lambda *a: calls.append(1))
        assert torch.equal(m(x), ref)
    assert len(calls) == len(m.layers)          # K/V computed once per layer (loop 1 only)


def test_pairs_and_window_kv_calls():
    for variant, per_layer in (("pairs", 2), ("sharefirst_w64", 4)):   # w64: later loops compute own K/V for the window
        m, x, calls = tiny(variant), torch.randint(0, tl.VOCAB, (1, 10)), []
        for layer in m.layers:
            layer.kv.register_forward_hook(lambda *a: calls.append(1))
        with torch.no_grad():
            m(x)
        assert len(calls) == len(m.layers) * per_layer, variant


def test_window_covers_whole_sequence_equals_reference_softmax():
    # with a window as long as the sequence, a later loop attends to [shared K/V ; own K/V], all causal
    m = tiny("sharefirst_w64")
    layer, T = m.layers[0], 6
    x = torch.randn(1, T, 32)
    cos, sin = m.cos[:T], m.sin[:T]
    shared = layer.own_kv(layer.n1(torch.randn(1, T, 32)), cos, sin)
    t = torch.arange(T)
    causal = t[:, None] >= t[None, :]
    with torch.no_grad():
        got, _ = layer(x, cos, sin, shared, torch.cat((causal, causal), 1))
        a = layer.n1(x)
        q = tl.rope(layer.q(a).view(1, T, 4, 8).transpose(1, 2), cos, sin)
        k, v = layer.own_kv(a, cos, sin)
        K, V = torch.cat((shared[0], k), 2), torch.cat((shared[1], v), 2)
        s = (q @ K.transpose(-1, -2)) / 8 ** 0.5
        s = s.masked_fill(~torch.cat((causal, causal), 1), float("-inf"))
        y = (s.softmax(-1) @ V).transpose(1, 2).reshape(1, T, 32)
        x2 = x + layer.o(y)
        g, u = layer.up(layer.n2(x2)).chunk(2, -1)
        ref = x2 + layer.down(torch.nn.functional.silu(g) * u)
    assert torch.allclose(got, ref, atol=1e-5)


def test_perloop_computes_kv_every_loop_and_differs_from_sharefirst():
    m, x = tiny("perloop"), torch.randint(0, tl.VOCAB, (1, 10))
    calls = []
    for layer in m.layers:
        layer.kv.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        out = m(x)
    assert len(calls) == len(m.layers) * m.loops
    s = tiny("sharefirst")
    s.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert not torch.allclose(s(x), out)


def test_full_size_param_count():
    m = tl.LoopedLM(**tl.VARIANTS["perloop"])
    n = sum(p.numel() for p in m.parameters())
    assert 80e6 < n < 83e6
