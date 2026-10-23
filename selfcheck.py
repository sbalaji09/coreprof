"""Run before trusting harness numbers.

1. GPU sanity: the same matmul (fp32 and bf16) repeated must give bit-identical
   results, and two identical prefills must give identical KV caches.
2. Harness correctness: chaining build_ut_step over all UT steps must reproduce
   OuroModel.forward on the same decode token (hidden states and exit-gate logits).
"""
import argparse

import torch

import harness as H


def gpu_repeatability(n=100):
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
        b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
        try:
            ref = a @ b
            bad = sum(not torch.equal(a @ b, ref) for _ in range(n))
            dtype_ok = bad == 0 and not torch.isnan(ref).any().item()
            msg = f"{bad}/{n} repeated matmuls differ"
        except Exception as e:  # a faulty GPU raises cuBLAS / illegal-access errors here
            dtype_ok, msg = False, f"error: {str(e).splitlines()[0]}"
        print(f"[gpu {dtype}] {msg}  -> {'OK' if dtype_ok else 'FAIL'}")
        if not dtype_ok:  # the CUDA context is usually unusable after an error
            return False
    return True


def forward_repeatability(model, B=4, C=64):
    """Two identical prefills must produce identical KV caches."""
    a, b = H.prefill(model, B, C, seed=0), H.prefill(model, B, C, seed=0)
    diff = max((x - y).abs().max().item() for x, y in zip(a.cache.key_cache, b.cache.key_cache))
    ok = diff == 0
    print(f"[model] identical prefills, max cache diff {diff:.2e}  -> {'OK' if ok else 'FAIL'}")
    return ok


def ut_step_matches_forward(model, B=4, C=64, tol=0.05):
    state = H.prefill(model, B, C, seed=0)
    ref = model.model(input_ids=state.next_ids, past_key_values=state.cache, use_cache=True,
                      cache_position=torch.arange(C, C + 1, device="cuda"))
    ref_hidden, ref_gates = ref[1], ref[2]
    ok = True
    for ut in range(model.model.total_ut_steps):
        state.reset()
        h, gate = H.build_ut_step(model, state, argparse.Namespace(ut_step=ut))()
        rel = ((h - ref_hidden[ut]).float().norm() / ref_hidden[ut].float().norm()).item()
        gate_err = (gate - ref_gates[ut]).abs().max().item()
        step_ok = rel < tol and gate_err < tol * 10
        ok &= step_ok
        print(f"[ut_step={ut}] rel err hidden {rel:.2e}, max abs err gate {gate_err:.2e}  -> {'OK' if step_ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    with torch.inference_mode():
        ok = gpu_repeatability()
        if ok:
            model = H.load_model("bf16")
            ok = forward_repeatability(model) and ut_step_matches_forward(model)
    print("ALL OK" if ok else "SELFCHECK FAILED: do not trust timing numbers")
    raise SystemExit(0 if ok else 1)
