import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batching.common import MODEL, REVISION, set_numerics  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model():
    """A 2-layer, randomly initialised Ouro built from Ouro's own remote code (fp32).
    Same architecture and loop/cache code paths as the real model, but runs in milliseconds."""
    set_numerics()
    cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=4, head_dim=16, vocab_size=256, max_window_layers=2,
                    layer_types=["full_attention"] * 2, total_ut_steps=4))
    cls = get_class_from_dynamic_module("modeling_ouro.OuroForCausalLM", MODEL, revision=REVISION)
    torch.manual_seed(0)
    model = cls._from_config(cfg, attn_implementation="sdpa", torch_dtype=torch.float32)
    return model.cuda().eval()
