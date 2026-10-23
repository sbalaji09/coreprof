"""Step 1: load Ouro-1.4B in bf16 and generate ~50 tokens."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"  # same pin as harness.py

torch.manual_seed(0)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()
torch.cuda.synchronize()
load_s = time.perf_counter() - t0

torch.cuda.reset_peak_memory_stats()
inputs = tok("The future of AI is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
print(f"load time: {load_s:.1f} s (includes download on first run)")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
