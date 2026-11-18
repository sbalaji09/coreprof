"""Run the LoopCore scripts on Modal GPUs instead of the NRP JupyterHub pod.

Usage (from the repo root, after `modal setup`):

    modal run modal_app.py                           # task "check" on an L40S
    modal run modal_app.py --task harness            # Week 1 latency sweep (B = 1..64, C = 512)
    modal run modal_app.py --task equivalence        # Week 2 fp32 equivalence test
    modal run modal_app.py --task tests              # pytest unit tests
    modal run modal_app.py --task quality            # GSM8K accuracy check
    modal run modal_app.py --task loopprof           # Week 4: B x C sweep + fit (~30-45 min)
    modal run modal_app.py --task delta_split        # Week 5: split delta (overhead vs weights)
    modal run modal_app.py --task delta_grid         # Week 5 v1: kernel breakdown + CUDA-graph B x C fit
    modal run --detach modal_app.py --task bench_q03 # Week 3: q=0.3 sweep + q=1.0 control (hours)
    modal run modal_app.py --task check --gpu A100-80GB
    modal run modal_app.py --task bench_q03 --run-id <id> --pull   # download a detached run

Every task runs selfcheck.py first and stops unless it prints ALL OK.
Outputs are written to the Modal volume "loopcore-vol" under results/<run-id>/ and then
downloaded to ./results/modal/<run-id>/ (including log.txt with the full console output).
Re-running with the same --run-id resumes: bench_cdb.py skips points it already finished.

Model weights and datasets are cached on the volume (HF_HOME=/vol/hf), so only the first run
downloads Ouro-1.4B (~3 GB).
"""

import os
import pathlib
import subprocess
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent
VOL_NAME = "loopcore-vol"

app = modal.App("loopcore")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "procps", "build-essential")   # C compiler: Triton builds its launcher (torch.compile)
    # Same pins as the NRP runs (torch 2.5.1 wheels on PyPI are the CUDA 12.4 build).
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .env({
        "HF_HOME": "/vol/hf",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(
        str(REPO), "/root/loopcore",
        ignore=[".git", ".venv", "results", "**/__pycache__", ".pytest_cache", "*.pyc"],
    )
)

# Each task is a list of shell commands run in order from the repo root.
# $OUT is the run's output dir on the volume, $RID the run id.
CHECK = ["python selfcheck.py 2>&1 | tee $OUT/selfcheck.txt; grep -q 'ALL OK' $OUT/selfcheck.txt"]
TASKS = {
    "check": CHECK + ["python smoke_test.py 2>&1 | tee $OUT/smoke_test.txt"],
    "harness": CHECK + [
        "python harness.py --out-dir $OUT",
        "python plot.py $(ls $OUT/*.csv | head -1)",
    ],
    "equivalence": CHECK + ["python equivalence_test.py --only fp32 --out-dir $OUT"],
    "tests": ["python -m pytest -q tests 2>&1 | tee $OUT/pytest.txt"],
    "quality": CHECK + ["python quality_check.py --out-dir $OUT"],
    # Week 3 remainder: q=0.3 on this GPU (S=1 skipped: with one slot CDB == baseline by
    # construction), plus a q=1.0 control at a few slot counts to confirm the noise floor.
    # Week 4: LoopProf sweep over batch x context, then fit t = δ + γB + αC (+ B·C variants).
    # Re-running with the same --run-id resumes the sweep where it stopped.
    "loopprof": CHECK + [
        "python loopprof.py --out-dir $OUT",
        "python fit_loopprof.py $OUT/loopprof.csv --target ut",
        "python fit_loopprof.py $OUT/loopprof.csv --target decode",
    ],
    # Week 5: split δ into weight streaming vs software overhead (layer sweep, CUDA graphs,
    # static cache, launch microbenchmarks), all in one container. ~10-15 min on an L40S.
    "delta_split": CHECK + ["python delta_split.py --out-dir $OUT"],
    # Week 5 v1: kernel-time breakdown + CUDA-graph static-cache B x C grid and fit (~15-20 min).
    "delta_grid": CHECK + ["python delta_split.py --grid --out-dir $OUT"],
    # Stronger GPU baseline: the same grid with the step torch.compile'd (kernel fusion) + CUDA graph,
    # plus the uncompiled graph at 3 points in the same container. One compile per grid point (~1 h).
    "delta_compile": CHECK + ["python delta_split.py --grid --compile --out-dir $OUT"],
    # int8 weight-only looped block (fused cast under torch.compile) vs the bf16 compiled step, same grid (~1 h).
    "delta_int8": CHECK + ["python delta_split.py --grid --compile --int8 --out-dir $OUT"],
    # Which int8 path reads int8 bytes? inductor settings vs a Triton int8 kernel, 3 points (~15 min).
    "int8_probe": CHECK + ["python int8_kernel.py | tee $OUT/int8_kernel_check.txt",
                           "python delta_split.py --int8-probe --out-dir $OUT"],
    # Full int8 grid with the Triton int8 kernel (int8_kernel.py) instead of the unfused cast (~1 h).
    "delta_int8_triton": CHECK + ["python delta_split.py --grid --compile --int8 --int8-impl triton --out-dir $OUT"],
    # Full int8 grid, cast path + coordinate_descent_tuning (best at B=1 in int8_probe) (~1 h).
    "delta_int8_cd": CHECK + ["python delta_split.py --grid --compile --int8 --coord-descent --out-dir $OUT"],
    # Is the compiled step's bf16 difference rounding or a bug? eager vs compiled, both against fp32 (~3 min).
    "compile_check": CHECK + ["python delta_split.py --compile-check --out-dir $OUT"],
    "bench_q03": CHECK + [
        "python bench_cdb.py --q 0.3 --slots 2 4 8 16 32 64 --repeats 3 --run-id $RID --out-dir $OUT",
        "python bench_cdb.py --q 1.0 --slots 4 16 64 --repeats 2 --run-id $RID --out-dir $OUT",
        "python plot_bench.py $OUT/bench_cdb_$RID",
    ],
}


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, volumes={"/vol": vol},
              timeout=12 * 3600)
def run(task: str, run_id: str, git: str = "") -> int:
    out = f"/vol/results/{run_id}"
    os.makedirs(out, exist_ok=True)
    env = {**os.environ, "OUT": out, "RID": run_id, "GIT": git}
    log_path = f"{out}/log.txt"

    def sh(cmd: str) -> int:
        with open(log_path, "a") as log:
            log.write(f"\n$ {cmd}\n")
            log.flush()
            p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], cwd="/root/loopcore",
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            for line in p.stdout:
                print(line, end="")
                log.write(line)
            rc = p.wait()
            log.write(f"[exit {rc}]\n")
        vol.commit()  # make partial results durable after every command
        return rc

    sh("{ echo task=$RID; echo git=$GIT; date -u; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version "
       "--format=csv; lscpu | grep -E 'Model name|^CPU\\(s\\)'; nproc; "
       "python -c 'import torch,transformers;print(\"torch\",torch.__version__,"
       "\"transformers\",transformers.__version__)'; } 2>&1 | tee $OUT/env.txt")
    for cmd in TASKS[task]:
        rc = sh(cmd)
        if rc != 0:
            print(f"\n*** stopped: '{cmd}' exited {rc} ***")
            return rc
    print(f"\n*** task '{task}' finished OK ***")
    return 0


@app.local_entrypoint()
def main(task: str = "check", gpu: str = "L40S", run_id: str = "", pull: bool = False):
    """--pull with --run-id only downloads that run's results (e.g. after a --detach run)."""
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    rid = run_id or f"{task}-{gpu.lower()}-{time.strftime('%Y%m%d-%H%M%S')}"
    rc = 0
    if pull:
        if not run_id:
            raise SystemExit("--pull needs --run-id")
    else:
        print(f"run id: {rid}  (gpu {gpu})")
        try:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip()
            git = head + ("-dirty" if dirty else "")
        except OSError:
            git = ""
        rc = run.with_options(gpu=gpu).remote(task, rid, git)

    # Pull everything this run wrote back into the repo.
    dest = REPO / "results" / "modal" / rid
    n = 0
    for entry in vol.listdir(f"/results/{rid}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        rel = pathlib.Path(entry.path.lstrip("/")).relative_to(f"results/{rid}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        n += 1
    print(f"downloaded {n} files to {dest}")
    if rc != 0:
        raise SystemExit(rc)
