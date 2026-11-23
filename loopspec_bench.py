"""B=1 decode speed of Ouro-1.4B under LoopSpec vs normal decoding, both in LoopSpec's SGLang stack
(results/NOTEBOOK_LOOPSPEC.md). Runs inside the Modal `loopspec` image, not our torch 2.5.1 image.

For each mode (normal decoding, LoopSpec "1", LoopSpec "1:2") it starts LoopSpec's server, sends the same GSM8K prompts
one at a time (greedy), and reads the server's own per-request decode timing (`generation_metrics` log lines: decode
tokens after the first, host-side seconds). Writes loopspec_bench.json with per-prompt texts, timings and summaries.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.request

LOOPSPEC_DIR = "/opt/loopspec"
MODEL_DIR = "/vol/models/Ouro-1.4B"   # LoopSpec identifies the model by the directory name
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"   # = harness.REVISION
GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"
MODES = {"baseline": [], "1": ["--loopspec", "--first", "1"], "1:2": ["--loopspec", "--first", "1", "--second", "2"]}
PORT = 30000
METRICS_RE = re.compile(r"generation_metrics\s+(\{.*\})\s*$")
# Loaded by every server process (via PYTHONPATH): logs which linear backend LoopSpec's install_linear_backend picks
# (FlashInfer TinyGEMM where supported, else SGLang's), and with --no-tinygemm keeps SGLang's (TinyGEMM crashed CUDA-graph
# capture on H100 twice: "unspecified launch failure"). Applies to all three modes alike.
SITECUSTOMIZE = """import os
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
"""


def gsm8k(split):
    with urllib.request.urlopen(GSM8K.format(split), timeout=60) as r:
        return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]


def prompts(n, shots=3):
    train, test = gsm8k("train"), gsm8k("test")
    head = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train[:shots])
    return [f"{head}Question: {e['question']}\nAnswer:" for e in test[:n]]


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, log_path, limit_s=900):
    """Seconds until SGLang reports ready. Its own start-up warm-up request finishes (and logs a metrics line) before
    this line, so metrics lines already in the log at that point are not ours."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}; see {log_path}")
        if "The server is fired up and ready to roll" in open(log_path).read():
            return time.time() - t0
        time.sleep(5)
    raise TimeoutError(f"server not ready after {limit_s} s; see {log_path}")


def run_mode(mode, texts, warmup, max_new, out_dir, no_tinygemm):
    log_path = os.path.join(out_dir, f"server_{mode.replace(':', '-')}.log")
    site = os.path.join(out_dir, "_site")
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    env = {**os.environ, "PYTHONPATH": f"{site}:{LOOPSPEC_DIR}"}
    if no_tinygemm:
        env["LOOPSPEC_BENCH_NO_TINYGEMM"] = "1"
    cmd = ["python", "-m", "sglang_recurrent.server", "--model", MODEL_DIR, "--host", "127.0.0.1",
           "--port", str(PORT)] + MODES[mode]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=LOOPSPEC_DIR, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env=env)
    try:
        ready_s = wait_ready(proc, log_path)
        skip = sum(1 for line in open(log_path) if METRICS_RE.search(line))
        outs = []
        for text in texts[:warmup] + texts:
            r = post("/generate", {"text": text, "sampling_params": {
                "temperature": 0, "max_new_tokens": max_new, "stop": ["Question:"]}})
            outs.append(dict(text=r["text"], prompt_tokens=r["meta_info"].get("prompt_tokens"),
                             completion_tokens=r["meta_info"].get("completion_tokens")))
        time.sleep(2)   # let the scheduler flush its last metrics line
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:   # server already gone (crashed)
            pass
        proc.wait(timeout=120)
    metrics = [json.loads(m.group(1)) for line in open(log_path) if (m := METRICS_RE.search(line))][skip:]
    if len(metrics) != len(outs):
        raise RuntimeError(f"{mode}: {len(metrics)} metrics lines for {len(outs)} requests; see {log_path}")
    rows = [dict(o, **mm) for o, mm in zip(outs, metrics)][warmup:]
    tok = sum(r["decode_tokens"] for r in rows)
    sec = sum(r["decode_seconds"] for r in rows)
    per = [r["decode_seconds"] / r["decode_tokens"] * 1e3 for r in rows if r["decode_tokens"]]
    summary = dict(ms_per_token=sec / tok * 1e3, median_ms_per_token=statistics.median(per), decode_tokens=tok,
                   mean_prompt_tokens=statistics.mean(r["prompt_tokens"] for r in rows),
                   mean_completion_tokens=statistics.mean(r["completion_tokens"] for r in rows),
                   server_ready_s=ready_s)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--no-tinygemm", action="store_true", help="keep SGLang's linear kernels in every mode")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download("ByteDance/Ouro-1.4B", revision=REVISION, local_dir=MODEL_DIR)
    import torch
    texts = prompts(a.n)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, model_revision=REVISION,
               loopspec_commit=open(os.path.join(LOOPSPEC_DIR, "COMMIT")).read().strip(), n=a.n, warmup=a.warmup,
               max_new_tokens=a.max_new_tokens, no_tinygemm=a.no_tinygemm, modes={})
    path = os.path.join(a.out_dir, "loopspec_bench.json")
    for mode in a.modes:   # warm-up = the first `warmup` measured prompts, sent once before the measured set
        rows, summary = run_mode(mode, texts, a.warmup, a.max_new_tokens, a.out_dir, a.no_tinygemm)
        res["modes"][mode] = dict(summary=summary, rows=rows)
        if "baseline" in res["modes"] and mode != "baseline":
            base = res["modes"]["baseline"]
            summary["identical_text_vs_baseline"] = sum(
                r["text"] == b["text"] for r, b in zip(rows, base["rows"])) / len(rows)
            summary["speedup_vs_baseline"] = base["summary"]["ms_per_token"] / summary["ms_per_token"]
        print(mode, json.dumps(summary))
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
