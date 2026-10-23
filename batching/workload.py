"""Fixed workload (DESIGN_CDB.md §6): the first N GSM8K test questions, zero-shot, tokenized."""
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .common import MODEL, REVISION

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROMPT = "Question: {question}\nAnswer:"


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def gsm8k_prompts(n=128):
    """Token-id lists for the first n GSM8K test questions. Cached under $HF_HOME (/tmp/hf)."""
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=GSM8K_REVISION)
    questions = pd.read_parquet(path)["question"].head(n).tolist()
    tok = tokenizer()
    return [tok(PROMPT.format(question=q))["input_ids"] for q in questions]
