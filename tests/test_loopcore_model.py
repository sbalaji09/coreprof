"""Unit tests for loopcore_model.py (pure Python, no GPU)."""
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


def test_l40s_calibration_reproduces_small_batch_step():
    t, _ = lm.step_time(M, lm.L40S_MEASURED, 1, 128)
    assert t == pytest.approx(5.82e-3, rel=0.03)


def test_l40s_grid_within_6_percent():
    # residual: B=2-32 run ~0.3 ms slower than linear gamma predicts (see validate())
    errs = [abs(lm.step_time(M, lm.L40S_MEASURED, B, C)[0] * 1e3 - ms) / ms for B, C, ms in lm.L40S_GRID]
    assert max(errs) < 0.06
    assert sum(errs) / len(errs) < 0.03


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
    for hw in lm.LOOPCORE_CONFIGS.values():
        assert lm.fit_check(M, hw)[2]


def test_kv_write_independent_of_exit_depth():
    hw = lm.L40S_MEASURED
    step, _ = lm.step_time(M, hw, 8, 512)
    shallow = lm.token_time(M, hw, 8, 512, 2.62)
    deep = lm.token_time(M, hw, 8, 512, 4.0)
    assert deep - shallow == pytest.approx((4.0 - 2.62) * step)


def test_bad_combine_rejected():
    with pytest.raises(ValueError):
        lm.step_time(M, replace(lm.L40S_MEASURED, combine="avg"), 1, 128)
