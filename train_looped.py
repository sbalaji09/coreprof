"""Week 6b: is KV sharing across loops free when it is TRAINED IN? Small looped LMs from scratch.

Ouro (trained with one KV cache per loop) collapses when its loops share a cache at inference (week6_share_check.py).
Here the sharing is part of training. Same data, same order, same seed, same hyperparameters for every variant:

  perloop        each loop attends to its own K/V (Ouro's layout): KV cache = loops x layers
  sharefirst     loops 2..L reuse loop 1's K/V of the same layer, including the current token (Mixture-of-Recursions
                 "recursive KV sharing"; trainable in one parallel pass, whereas "last loop" sharing needs later
                 loops of earlier positions and is not). KV cache = layers
  sharefirst_w64 sharefirst + each later loop also attends to its OWN K/V of the last 64 positions, in the same
                 softmax (Monea et al. 2026, arXiv 2610.02383). KV cache = layers + a 64-token window per loop
  pairs          loops 1-2 use loop 1's K/V, loops 3-4 use loop 3's (Cross-Layer-Attention-style). KV = 2 x layers
  mqa3           per-loop K/V with 3 KV heads instead of 12: the SAME KV bytes as sharefirst, the standard alternative
  noloop         the same weights run once (does looping help at this scale at all?)

Model (Ouro-like): token embedding (tied LM head), L loops of [6 pre-norm layers (RoPE attention, SwiGLU) + RMSNorm],
LM head on the last loop. d=768, 12 heads x 64, FFN 2048: 42.5M looped params + 38.6M embedding.
Data: prep_data.py (FineWeb-Edu, GPT-2 tokens, uint16). Batches are drawn from a seed + step RNG, so a resumed run
sees exactly the batches an uninterrupted one would. Writes $OUT/<variant>/log.jsonl, final.json, ckpt.pt.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {    # kv_every: loop u computes K/V if u % kv_every == 0, else reuses the last computed loop's (same layer)
    "perloop": dict(kv_every=1, n_kv_heads=12, loops=4),
    "sharefirst": dict(kv_every=4, n_kv_heads=12, loops=4),
    "sharefirst_w64": dict(kv_every=4, n_kv_heads=12, loops=4, window=64),
    "pairs": dict(kv_every=2, n_kv_heads=12, loops=4),
    "mqa3": dict(kv_every=1, n_kv_heads=3, loops=4),
    "noloop": dict(kv_every=1, n_kv_heads=12, loops=1),
}
VOCAB = 50257


def rope(x, cos, sin):                       # x [B, H, T, D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Layer(nn.Module):
    def __init__(self, d, h, hkv, ff):
        super().__init__()
        self.h, self.hkv, self.hd = h, hkv, d // h
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * hkv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 2 * ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def own_kv(self, a, cos, sin):
        B, T, _ = a.shape
        k, v = self.kv(a).view(B, T, 2, self.hkv, self.hd).permute(2, 0, 3, 1, 4)
        k = rope(k, cos, sin)
        if self.hkv < self.h:
            k, v = (t.repeat_interleave(self.h // self.hkv, 1) for t in (k, v))
        return k, v

    def forward(self, x, cos, sin, kv=None, win_mask=None):
        """kv=None: attend to own K/V (returned for reuse). kv given: attend to it, plus own K/V within the
        window when win_mask ([T, 2T] bool: shared causal | own causal-and-recent) is given."""
        B, T, _ = x.shape
        a = self.n1(x)
        q = rope(self.q(a).view(B, T, self.h, self.hd).transpose(1, 2), cos, sin)
        if kv is None:
            kv = self.own_kv(a, cos, sin)
            y = F.scaled_dot_product_attention(q, *kv, is_causal=True)
        elif win_mask is None:
            y = F.scaled_dot_product_attention(q, *kv, is_causal=True)
        else:
            k, v = self.own_kv(a, cos, sin)
            y = F.scaled_dot_product_attention(q, torch.cat((kv[0], k), 2), torch.cat((kv[1], v), 2),
                                               attn_mask=win_mask)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, -1))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u), kv


class LoopedLM(nn.Module):
    def __init__(self, kv_every, n_kv_heads, loops, window=0, d=768, h=12, layers=6, ff=2048, ctx=1024):
        super().__init__()
        self.kv_every, self.loops, self.window = kv_every, loops, window
        self.emb = nn.Embedding(VOCAB, d)
        self.layers = nn.ModuleList(Layer(d, h, n_kv_heads, ff) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        hd = d // h
        f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(ctx).float(), f)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.apply(self._init)
        for l in self.layers:                     # GPT-2-style scaled init of residual outputs (by effective depth)
            for p in (l.o.weight, l.down.weight):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers * loops))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        win = None
        if self.window:
            t = torch.arange(T, device=idx.device)
            causal = t[:, None] >= t[None, :]
            win = torch.cat((causal, causal & (t[:, None] - t[None, :] < self.window)), 1)
        x, shared = self.emb(idx), None
        for u in range(self.loops):
            fresh = u % self.kv_every == 0
            kvs = []
            for i, layer in enumerate(self.layers):
                x, kv = layer(x, cos, sin, None if fresh else shared[i], None if fresh else win)
                kvs.append(kv)
            if fresh:
                shared = kvs
            x = self.norm(x)
        return F.linear(x, self.emb.weight)


def batch(data, rng, bsz, ctx, device):
    ix = rng.integers(0, len(data) - ctx - 1, bsz)
    b = torch.from_numpy(np.stack([data[i:i + ctx + 1] for i in ix]).astype(np.int64)).to(device, non_blocking=True)
    return b[:, :-1], b[:, 1:]


@torch.no_grad()
def evaluate(model, val, n_batches, bsz, ctx):
    rng, tot = np.random.default_rng(12345), 0.0      # the same held-out windows for every variant and every eval
    for _ in range(n_batches):
        x, y = batch(val, rng, bsz, ctx, "cuda")
        with torch.autocast("cuda", torch.bfloat16):
            tot += F.cross_entropy(model(x).float().view(-1, VOCAB), y.reshape(-1)).item()
    return tot / n_batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--data", default="/vol/data/fineweb_edu_gpt2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", type=float, default=2e9)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--micro-bsz", type=int, default=32)
    ap.add_argument("--accum", type=int, default=16)             # 32 x 1024 x 16 = 524k tokens per step
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--final-eval-batches", type=int, default=400)   # 13M held-out tokens
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    out = os.path.join(args.out_dir, args.variant if args.seed == 0 else f"{args.variant}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    model = LoopedLM(**VARIANTS[args.variant], ctx=args.ctx).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    steps = int(args.tokens // (args.micro_bsz * args.ctx * args.accum))
    lr_at = lambda s: args.lr * (s + 1) / args.warmup if s < args.warmup else \
        args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, steps - args.warmup))))

    start, ckpt = 0, os.path.join(out, "ckpt.pt")
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"resumed at step {start}", flush=True)
    fwd = model if args.no_compile else torch.compile(model)
    meta = dict(variant=args.variant, **VARIANTS[args.variant], params=n_params, steps=steps,
                tokens_per_step=args.micro_bsz * args.ctx * args.accum, args=vars(args),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    print(json.dumps(meta), flush=True)
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    for step in range(start, steps):
        rng = np.random.default_rng([args.seed, step])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tot = 0.0
        for _ in range(args.accum):
            x, y = batch(train, rng, args.micro_bsz, args.ctx, "cuda")
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(fwd(x).float().view(-1, VOCAB), y.reshape(-1)) / args.accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(tot):
            raise SystemExit(f"non-finite loss at step {step}")
        rec = dict(step=step + 1, train_loss=tot, grad_norm=gn, lr=lr_at(step))
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            model.eval()
            rec["val_loss"] = evaluate(fwd, val, args.eval_batches, args.micro_bsz, args.ctx)
            model.train()
            rec["tok_per_s"] = (step + 1 - start) * meta["tokens_per_step"] / (time.time() - t0)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1), ckpt)
            print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    model.eval()
    final = evaluate(fwd, val, args.final_eval_batches, args.micro_bsz, args.ctx)
    res = dict(meta, final_val_loss=final, final_val_ppl=math.exp(final), hours=(time.time() - t0) / 3600)
    json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
