"""Week 6b data: FineWeb-Edu (sample-10BT) -> GPT-2 tokens, uint16, one EOS (50256) after each document.

train.bin = documents from the first shards until --train-tokens; val.bin = --val-tokens from the LAST shard
(disjoint documents). The dataset commit is pinned on first use and recorded in meta.json. Idempotent.
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

REPO, EOS = "HuggingFaceFW/fineweb-edu", 50256


def write(path, shards, sha, target, tok):
    n = 0
    with open(path, "wb") as f:
        for shard in shards:
            pf = pq.ParquetFile(hf_hub_download(REPO, shard, repo_type="dataset", revision=sha))
            for rb in pf.iter_batches(batch_size=4096, columns=["text"]):
                ids = tok(rb.column("text").to_pylist())["input_ids"]
                arr = np.fromiter((t for doc in ids for t in (*doc, EOS)), dtype=np.uint16)
                f.write(arr[: target - n].tobytes())
                n += min(len(arr), target - n)
                if n >= target:
                    return n
            print(f"{path}: {n / 1e9:.2f}B tokens after {shard}", flush=True)
    raise SystemExit(f"ran out of data for {path}: {n} < {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vol/data/fineweb_edu_gpt2")
    ap.add_argument("--train-tokens", type=float, default=2.1e9)
    ap.add_argument("--val-tokens", type=float, default=20e6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "meta.json")):
        print(open(os.path.join(args.out, "meta.json")).read())
        return
    api = HfApi()
    sha = api.dataset_info(REPO).sha
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", revision=sha)
                    if f.startswith("sample/10BT/") and f.endswith(".parquet"))
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 10**9
    val = write(os.path.join(args.out, "val.bin"), shards[-1:], sha, int(args.val_tokens), tok)
    train = write(os.path.join(args.out, "train.bin"), shards[:-1], sha, int(args.train_tokens), tok)
    meta = dict(repo=REPO, revision=sha, n_shards=len(shards), val_shard=shards[-1], train_tokens=train,
                val_tokens=val, tokenizer="gpt2", eos=EOS)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
