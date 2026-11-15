"""Print Ouro-1.4B parameter shapes without downloading weights (meta device, runs on a laptop).

Usage: python inspect_params.py
Checks the hand-derived SRAM budget: layer-0 shapes, non-layer params, and the total.
"""
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"  # same pin as smoke_test.py / harness.py

cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
with init_empty_weights():  # shapes only, no memory
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

layer_total = 0
for name, p in model.named_parameters():
    if ".layers.0." in name or ".layers." not in name:  # layer 0 + everything outside the layers
        print(f"{name:55s} {tuple(p.shape)}")
    if ".layers.0." in name:
        layer_total += p.numel()

total = sum(p.numel() for p in model.parameters())
print(f"\nlayer 0 params: {layer_total:,}")
print(f"total params:   {total:,}")
