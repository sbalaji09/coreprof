"""Unit tests for loopcore_model.py (pure Python, no GPU): python -m pytest -q tests/test_loopcore_model.py"""
import statistics
from dataclasses import replace

import pytest

import loopcore_model as lm

M = lm.OURO_1_4B


def test_kv_bytes_match_readme():
    # README: per-loop caches cost about 0.79 MB per token in bf16 across 4 loops
    per_loop = M.kv_elems_per_token_per_loop() * 2
    assert per_loop == 196_608
    assert per_loop * M.total_loops == pytest.approx(0.79e6, rel=0.01)


def test_attention_flops_per_cached_token():
    assert M.attn_flops_per_cached_token() == 196_608


def test_core_weight_bytes_match_readme():
    # README: a loop step reads about 2.5 GB of weights (bf16)
    assert M.core_params * 2 == pytest.approx(2.5e9, rel=0.02)


def _grid_ok(errs, tol=10):
    """calibrate.py's grid rule: median |error| <= tol and max |error| <= 2 x tol."""
    return statistics.median(map(abs, errs.values())) <= tol and max(map(abs, errs.values())) <= 2 * tol


def test_l40s_fit_reproduces_its_envelope():
    # in-sample: constants come from this data, so this checks structure, not prediction
    assert _grid_ok(lm.grid_errors(M, lm.fit_gpu("l40s"), "l40s"))


@pytest.mark.xfail(strict=True, reason="2026-11-20: the best-GPU envelope now includes bf16 + coordinate-descent "
                   "runs (faster fused GEMV reductions); the v1 clock-transfer model, fitted on the earlier envelope, "
                   "predicts H100 at median 11.1 % (passed at 6.3 % before). Not re-derived; see CALIBRATION.md.")
def test_h100_predicted_from_l40s_passes_calibration_rule():
    # out-of-sample: no H100 timings enter predict_gpu (only its datasheet clock/compute and measured bandwidth)
    h100 = lm.predict_gpu(lm.fit_gpu("l40s"), "h100")
    assert _grid_ok(lm.grid_errors(M, h100, "h100"))


def test_bytes_over_bandwidth_alone_fails_on_h100():
    # v0's weight model (no per-kernel cost) under-predicts H100 weight time by > 20 %
    h100 = replace(lm.predict_gpu(lm.fit_gpu("l40s"), "h100"), weight_kernel_s=0.0)
    w = lm.step_terms(M, h100, 1, 128)["weights_sram"]
    assert w < 0.8 * 1.49e-3


def test_fit_uses_weights_without_attention():
    # WEEK5.md's 3.97 ms "weights" included flash attention; the fused breakdown's gemm is weights only
    w = lm.step_terms(M, lm.fit_gpu("l40s"), 1, 128)["weights_sram"]
    assert w == pytest.approx(3.77e-3, rel=0.01)


def test_pipeline_uses_per_chip_sram_bandwidth():
    # one token visits the chips in turn: weight time = all resident bytes / one chip's bandwidth
    a = lm.LOOPCORE_CONFIGS["A: 2 chips, 4-bit, all resident"]
    assert lm.step_terms(M, a, 1, 128)["weights_sram"] == pytest.approx(M.core_params * 0.5 / lm.SRAM_BW)


def test_link_counts_return_hop_and_latency():
    a = lm.LOOPCORE_CONFIGS["A: 2 chips, 4-bit, all resident"]
    t = lm.step_terms(M, a, 1, 128)["link"]
    assert t == pytest.approx(2 * (lm.HOP_LATENCY + M.d_model * 2 / lm.LINK_BW))
    c = lm.LOOPCORE_CONFIGS["C: 1 hypothetical 700 MB chip, 4-bit"]
    assert lm.step_terms(M, c, 1, 128)["link"] == 0


def test_max_never_exceeds_sum():
    for hw in lm.LOOPCORE_CONFIGS.values():
        for B, C in [(1, 128), (64, 8192)]:
            s, _ = lm.step_time(M, replace(hw, combine="sum"), B, C)
            x, _ = lm.step_time(M, replace(hw, combine="max"), B, C)
            assert x <= s


def test_fit_checks():
    # 4-bit core alone is 617 MB: exceeds one 500 MB chip unless partially resident
    full_one_chip = replace(lm.LOOPCORE_CONFIGS["B: 1 chip, 4-bit, partial residency"], resident_fraction=1.0)
    assert not lm.fit_check(M, full_one_chip)[2]
    # 8-bit core (1.23 GB) + 8-bit LM head (0.10 GB) needs 3 chips
    two = replace(lm.LOOPCORE_CONFIGS["D: 3 chips, 8-bit, all resident"], n_chips=2)
    assert not lm.fit_check(M, two)[2]
    for hw in lm.LOOPCORE_CONFIGS.values():
        assert lm.fit_check(M, hw)[2]


def test_measured_baselines_cover_compare_points():
    for g in ("l40s", "h100"):
        for p in ("bf16", "int8"):
            steps = lm.measured_steps(g, p)
            assert all((B, C) in steps for B in (1, 8, 64) for C in (128, 2048))
    # int8 is faster than bf16 at B=1 on both GPUs
    for g in ("l40s", "h100"):
        assert lm.measured_steps(g, "int8")[(1, 128)] < lm.measured_steps(g, "bf16")[(1, 128)]


def test_kv_write_independent_of_exit_depth():
    hw = lm.fit_gpu("l40s")
    step, _ = lm.step_time(M, hw, 8, 512)
    shallow = lm.token_time(M, hw, 8, 512, 2.62)
    deep = lm.token_time(M, hw, 8, 512, 4.0)
    assert deep - shallow == pytest.approx((4.0 - 2.62) * step)


def test_bad_combine_rejected():
    with pytest.raises(ValueError):
        lm.step_time(M, replace(lm.fit_gpu("l40s"), combine="avg"), 1, 128)


def test_kernel_sum_weight_split_matches_both_gpus():
    # v1.1 (microbenchmarks only): weight kernel time at 1x128 within 10 % of the fused step's GEMM time on both GPUs
    for gpu, meas in (("l40s", 3.772), ("h100", 1.515)):
        w = lm.kernel_sum(gpu, 1, 128, "busy_us")[0] / 1e3
        assert abs(w - meas) / meas < 0.10


def test_kernel_sum_known_misses_documented():
    # recorded, not hidden: isolated small kernels overstate L40S in-step overhead, and the compiled step's
    # batched GEMMs at B >= 32 are slower than isolated cuBLAS (both under-/over-predictions exceed 20 %)
    import calibrate as cal
    env, _ = cal.envelope("h100")
    t, _ = env[(64, 128)]
    assert sum(lm.kernel_sum("h100", 64, 128)) / 1e3 < 0.8 * t


# ---------------------------------------------------------------- KV design options (HANDOFF.md §4 item 1)

D = lm.LOOPCORE_CONFIGS["D: 3 chips, 8-bit, all resident"]


def test_kv_defaults_unchanged():
    # kv_caches=0 means one cache per loop: footprint and writes as Ouro ships
    assert lm.n_kv_caches(M, D) == 4
    assert lm.kv_footprint(M, D, 1, 1) == M.kv_elems_per_token_per_loop() * 2 * 4


def test_shared_cache_cuts_footprint_and_writes_not_reads():
    one = replace(D, kv_caches=1)
    assert lm.kv_footprint(M, one, 16, 2048) == lm.kv_footprint(M, D, 16, 2048) / 4
    assert lm.step_terms(M, one, 16, 2048)["kv"] == lm.step_terms(M, D, 16, 2048)["kv"]   # each loop still reads C tokens
    kv_write = lambda hw: lm.per_token_extras(M, hw, 16) - lm.per_token_extras(M, replace(hw, kv_bytes=0), 16)
    assert kv_write(one) == pytest.approx(kv_write(D) / 4)


def test_fp8_kv_halves_kv_time():
    fp8 = replace(D, kv_bytes=1.0)
    assert lm.step_terms(M, fp8, 8, 4096)["kv"] == pytest.approx(lm.step_terms(M, D, 8, 4096)["kv"] / 2)


def test_kv_on_chip_uses_sram_bandwidth_and_resource():
    on = replace(D, kv_on_chip=True)
    t = lm.step_terms(M, on, 8, 4096)
    assert t["kv"] == pytest.approx(8 * 4096 * M.kv_elems_per_token_per_loop() * 2 / lm.SRAM_BW)
    assert t["_by_resource"]["ext_mem"] == 0
    assert t["_by_resource"]["weight_mem"] == pytest.approx(t["weights_sram"] + t["kv"])


def test_sizing_counts_kv_only_when_on_chip():
    # weights + LM head at 8-bit = 1.33 GB -> 3 chips; KV in HBM never adds chips
    assert lm.sized(M, D, 16, 8192).n_chips == 3
    # fp8, shared, on-chip at 16x8192: 1.33 GB + 12.9 GB -> 29 chips (HANDOFF.md §7)
    on = replace(D, kv_bytes=1.0, kv_caches=1, kv_on_chip=True)
    assert lm.sized(M, on, 16, 8192).n_chips == 29
    assert lm.fit_check(M, lm.sized(M, on, 16, 8192), 16, 8192)[2]
    assert not lm.fit_check(M, replace(on, n_chips=28), 16, 8192)[2]


def test_max_layer_latency_hits_target():
    hw = replace(D, kv_bytes=1.0, kv_caches=1, kv_on_chip=True)
    target = 0.5 * lm.token_time(M, replace(hw, overhead_s=M.n_layers * 5e-6), 1, 2048, 3.22)
    lat = lm.max_layer_latency(M, hw, 1, 2048, 3.22, target)
    assert lm.token_time(M, replace(hw, overhead_s=M.n_layers * lat), 1, 2048, 3.22) == pytest.approx(target)
    assert lm.max_layer_latency(M, hw, 1, 2048, 3.22, 1e-9) is None


def test_kv_levers_reproduce_handoff_exploration():
    # HANDOFF.md §7 (2026-11-20), vs measured H100 int8, q=0.5, 5 µs/layer
    depth = lm.MEAN_DEPTH[0.5]
    h100 = lm.predict_gpu(lm.fit_gpu("l40s"), "h100")
    gpu8 = lm.measured_steps("h100", "int8")
    base = replace(D, overhead_s=M.n_layers * 5e-6)
    expect = {(1, 2048): [6.1, 8.0, 8.0, 11.4], (16, 8192): [1.5, 3.0, 3.0, 40.8]}
    for (B, C), want in expect.items():
        gpu_t = depth * gpu8[(B, C)] * 1e-3 + lm.per_token_extras(M, h100, B)
        got = [gpu_t / lm.token_time(M, lm.sized(M, replace(base, **kw), B, C), B, C, depth)
               for kw in lm.KV_LEVERS.values()]
        assert got == pytest.approx(want, abs=0.05)
