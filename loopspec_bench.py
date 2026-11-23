"""B=1 decode speed of Ouro-1.4B under LoopSpec vs normal decoding, both in LoopSpec's SGLang stack
(results/NOTEBOOK_LOOPSPEC.md). Runs inside the Modal `loopspec` image, not our torch 2.5.1 image.

For each mode (normal decoding, LoopSpec "1", LoopSpec "1:2") it starts LoopSpec's server, sends the same GSM8K prompts
one at a time (greedy), and reads the server's own per-request decode timing (`generation_metrics` log lines: decode
tokens after the first, host-side seconds). Writes loopspec_bench.json with per-prompt texts, timings and summaries.
"""
import argparse
import json
import math
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
MODES = {"baseline": [], "1": ["--loopspec", "--first", "1"], "1:2": ["--loopspec", "--first", "1", "--second", "2"],
         # a second normal-decoding server in the same container: how much text drift is ordinary run-to-run noise
         "baseline_repeat": []}
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


def prompts(n, shots=3, strip=False):
    """(prompt texts, gold answer strings). strip: remove <<...>> calculator annotations from the shots, as
    quality_check.py (Week 6) does."""
    train, test = gsm8k("train"), gsm8k("test")
    clean = (lambda a: re.sub(r"<<.*?>>", "", a)) if strip else (lambda a: a)
    head = "".join(f"Question: {e['question']}\nAnswer: {clean(e['answer'])}\n\n" for e in train[:shots])
    return ([f"{head}Question: {e['question']}\nAnswer:" for e in test[:n]],
            [e["answer"].split("####")[-1].strip() for e in test[:n]])


# Answer extraction and paired statistics copied from quality_check.py (Week 6); that module imports torch and the
# batching engine, which this image and the laptop do not have.
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_number(s):
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract(text):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(" + NUM.pattern + ")", text)
    if m:
        return to_number(m.group(1))
    nums = NUM.findall(text)
    return to_number(nums[-1]) if nums else None


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def grade(path, ref="baseline"):
    """Accuracy per mode and the paired change vs ref (Week 6 rule 1: keeps accuracy if the 95 % CI's lower end > −2 pts)."""
    res = json.load(open(path))
    ok = {}
    for mode, r in res["modes"].items():
        ok[mode] = [(p := extract(x["text"])) is not None and abs(p - to_number(x["gold"])) < 1e-6 for x in r["rows"]]
        print(f"{mode}: {sum(ok[mode])}/{len(ok[mode])} = {100 * sum(ok[mode]) / len(ok[mode]):.1f} %")
    for mode in ok:
        if mode == ref:
            continue
        a, b = ok[ref], ok[mode]
        only_ref, only_var = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
        n, diff = len(a), (only_var - only_ref) / len(a)
        se = math.sqrt(max(0.0, only_ref + only_var - n * diff * diff)) / n
        same = sum(extract(x["text"]) == extract(y["text"]) for x, y in zip(res["modes"][mode]["rows"], res["modes"][ref]["rows"]))
        lo = 100 * (diff - 1.96 * se)
        print(f"{mode} vs {ref}: {100 * diff:+.2f} pts (95 % CI {lo:+.2f} to {100 * (diff + 1.96 * se):+.2f}); only {ref} right "
              f"{only_ref}, only {mode} right {only_var}, McNemar p={mcnemar_p(only_ref, only_var):.3f}; same answer {same}/{n}; "
              f"rule (lower end > -2): {'PASS' if lo > -2 else 'FAIL'}")


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


def run_mode(mode, texts, golds, warmup, max_new, out_dir, no_tinygemm):
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
            outs.append(dict(text=r["text"], gold=golds[i] if (i := len(outs) - warmup) >= 0 else None,
                             prompt_tokens=r["meta_info"].get("prompt_tokens"),
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
    ap.add_argument("--out-dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--shots", type=int, default=3, help="GSM8K examples before each question (sets the context)")
    ap.add_argument("--strip-annotations", action="store_true", help="remove <<...>> from the shots (Week 6 format)")
    ap.add_argument("--grade", metavar="JSON", help="laptop: accuracy and paired comparison from a loopspec_bench.json")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--no-tinygemm", action="store_true", help="keep SGLang's linear kernels in every mode")
    a = ap.parse_args()
    if a.grade:
        return grade(a.grade)
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download("ByteDance/Ouro-1.4B", revision=REVISION, local_dir=MODEL_DIR)
    import torch
    texts, golds = prompts(a.n, a.shots, a.strip_annotations)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, model_revision=REVISION,
               loopspec_commit=open(os.path.join(LOOPSPEC_DIR, "COMMIT")).read().strip(), n=a.n, warmup=a.warmup,
               max_new_tokens=a.max_new_tokens, shots=a.shots, strip_annotations=a.strip_annotations,
               no_tinygemm=a.no_tinygemm, modes={})
    path = os.path.join(a.out_dir, "loopspec_bench.json")
    for mode in a.modes:   # warm-up = the first `warmup` measured prompts, sent once before the measured set
        rows, summary = run_mode(mode, texts, golds, a.warmup, a.max_new_tokens, a.out_dir, a.no_tinygemm)
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
