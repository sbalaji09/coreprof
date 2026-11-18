"""int8 weight-only linear for decode: y = (x @ int8_w.T) * scale, reading int8 weights directly.

torch.compile in torch 2.5.1 does not fuse the int8 -> bf16 cast of Int8WeightOnlyLinear into the matmul
(it writes bf16 weights every step; results/modal/delta_int8-*), so this Triton kernel loads int8 tiles,
converts them in registers and accumulates in fp32. One program per BLOCK_N output columns (and per
BLOCK_M rows; BLOCK_M covers all of M <= 64, so weights are read once). Autotuned per (M, N, K).

Split-K variant: with N = 2048 there are only ~32-128 column blocks, too few to keep an H100's memory busy
(int8_probe: 0.9-1.1x bf16 at B=2-16). The split-K kernel also splits K across programs and atomically adds
fp32 partial sums (scale already applied), then casts; that costs a zero-fill and a cast kernel. The first
call for each (M, N, K) times both paths and keeps the faster (eager, before any CUDA-graph capture).

Registered as the custom op torch.ops.loopcore.int8_wo_linear (opaque to torch.compile, CUDA-graph safe
once autotuned; delta_split's first call compiles and autotunes before capture).
Self-check: python int8_kernel.py   (needs a GPU)
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
             for bn in (16, 32, 64) for bk in (128, 256, 512) for w in (4,) for s in (3, 4)],
    key=["M", "N", "K"],
)
@triton.jit
def _int8_wo_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc * s[None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=4, num_stages=st)
             for bn in (16, 32, 64) for bk in (128, 256) for sk in (2, 4, 8) for st in (3, 4)],
    key=["M", "N", "K"], reset_to_zero=["y_ptr"],
)
@triton.jit
def _int8_wo_splitk_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           SPLIT_K: tl.constexpr):
    pid_n, pid_m, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_per = tl.cdiv(K, SPLIT_K * BLOCK_K) * BLOCK_K          # multiple of BLOCK_K: slices do not overlap
    k_lo = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_lo + k_per, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K), other=0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    tl.atomic_add(y_ptr + rm[:, None] * N + rn[None, :], acc * s[None, :],
                  mask=(rm[:, None] < M) & (rn[None, :] < N))


def _plain(x2, w, s, M, N, K, bm):
    y = torch.empty(M, N, device=x2.device, dtype=x2.dtype)
    _int8_wo_kernel[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(M, bm))](x2, w, s, y, M, N, K, BLOCK_M=bm)
    return y


def _splitk(x2, w, s, M, N, K, bm):
    acc = torch.zeros(M, N, device=x2.device, dtype=torch.float32)
    _int8_wo_splitk_kernel[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(M, bm), meta["SPLIT_K"])](
        x2, w, s, acc, M, N, K, BLOCK_M=bm)
    return acc.to(x2.dtype)


PATHS = {"plain": _plain, "splitk": _splitk}
CHOICE = {}          # (M, N, K) -> "plain" | "splitk", decided on first call


def _choose(x2, w, s, M, N, K, bm):
    times = {}
    for name, fn in PATHS.items():
        fn(x2, w, s, M, N, K, bm)                            # autotune this path first
        times[name] = triton.testing.do_bench(lambda: fn(x2, w, s, M, N, K, bm), warmup=10, rep=50)
    return min(times, key=times.get)


@torch.library.custom_op("loopcore::int8_wo_linear", mutates_args=())
def int8_wo_linear(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x [..., K] bf16, w [N, K] int8, s [N] bf16 -> [..., N] bf16."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    N = w.shape[0]
    assert M <= 64, "decode kernel: one BLOCK_M covers M <= 64"
    bm = min(64, max(16, triton.next_power_of_2(M)))
    key = (M, N, K)
    if key not in CHOICE:
        CHOICE[key] = _choose(x2, w, s, M, N, K, bm)
    return PATHS[CHOICE[key]](x2, w, s, M, N, K, bm).reshape(*x.shape[:-1], N)


@int8_wo_linear.register_fake
def _(x, w, s):
    return x.new_empty(*x.shape[:-1], w.shape[0])


def check(shapes=((2048, 2048), (5632, 2048), (2048, 5632)), ms=(1, 16, 64), seed=0):
    """Max relative error vs the bf16 reference F.linear(x, w.bf16) * s, over Ouro's linear shapes, for both
    paths and the op; also reports which path the op chose per (M, N, K)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    worst = 0.0
    for N, K in shapes:
        w = torch.randint(-128, 128, (N, K), device="cuda", dtype=torch.int8, generator=g)
        s = (torch.rand(N, device="cuda", generator=g) * 1e-2).to(torch.bfloat16)
        for M in ms:
            x = torch.randn(M, 1, K, device="cuda", generator=g).to(torch.bfloat16)
            ref = (torch.nn.functional.linear(x.float(), w.float()) * s.float()).float()
            x2 = x.reshape(M, K)
            bm = min(64, max(16, triton.next_power_of_2(M)))
            for fn in PATHS.values():
                fn(x2, w, s, M, N, K, bm)    # first call autotunes: split-K's output from it is not valid
                out = fn(x2, w, s, M, N, K, bm).float()
                worst = max(worst, float((out - ref).norm() / ref.norm()))
            out = int8_wo_linear(x, w, s).float().reshape(M, N)
            worst = max(worst, float((out - ref).norm() / ref.norm()))
    return dict(max_rel_err=worst, ok=worst < 1e-2, choice={f"{m}x{n}x{k}": c for (m, n, k), c in CHOICE.items()})


if __name__ == "__main__":
    print(check())
