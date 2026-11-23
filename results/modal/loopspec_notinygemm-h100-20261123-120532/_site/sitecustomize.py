import os
import sglang_recurrent.integrations.flashinfer as _f
_orig = _f.install_linear_backend
def _wrapped(**kw):
    if os.environ.get("LOOPSPEC_BENCH_NO_TINYGEMM"):
        st = {"active": "torch", "reason": "TinyGEMM disabled by loopspec_bench --no-tinygemm"}
    else:
        st = _orig(**kw)
    print("loopspec_bench linear backend:", st, flush=True)
    return st
_f.install_linear_backend = _wrapped
