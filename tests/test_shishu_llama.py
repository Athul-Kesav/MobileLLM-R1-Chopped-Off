"""ShiShu MobileLLM-R1 architecture and checkpoint layout."""

import json
from pathlib import Path

import pytest
import torch

from src.shishu_llama import (
    ShishuLlamaConfig,
    ShishuLlamaForCausalLM,
    layer_kinds,
    register_shishu_llama,
    required_weight_keys,
    validate_weight_keys,
)
from src.shishu_llama.modeling_shishu_llama import ShishuMLPLayer


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "shishu_mobilellm_r1_140m.json"


def test_checked_in_config_matches_the_surgery_stack():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    register_shishu_llama()
    config = ShishuLlamaConfig(**raw)
    assert config.model_type == "shishu-llama"
    assert layer_kinds(config) == ["decoder"] * 11 + ["mlp"] * 5
    assert config.stack_size == 16
    assert config.rope_parameters["rope_theta"] == 8000000.0
    assert config.use_qk_norm is True
    assert config.tie_word_embeddings is True


def test_weight_key_contract_rejects_attention_inside_mlp_blocks():
    config = ShishuLlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_decoder_front=1,
        num_mlp=2,
        num_decoder_end=0,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )
    keys = set(required_weight_keys(config))
    validate_weight_keys(config, keys)
    keys.add("model.layers.1.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="attention"):
        validate_weight_keys(config, keys)


def test_tiny_model_forward_uses_mlp_blocks_and_tied_embeddings():
    config = ShishuLlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_decoder_front=2,
        num_mlp=2,
        num_mlp_shared=1,
        num_decoder_end=0,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_qk_norm=True,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = ShishuLlamaForCausalLM(config)
    assert isinstance(model.model.layers[2], ShishuMLPLayer)
    assert not hasattr(model.model.layers[2], "self_attn")
    assert model.model.layers[2].mlp.gate_proj.weight is model.model.layers[3].mlp.gate_proj.weight
    assert model.lm_head.weight is model.model.embed_tokens.weight
    batch = torch.randint(0, config.vocab_size, (2, 6))
    output = model(input_ids=batch, labels=batch)
    assert tuple(output.logits.shape) == (2, 6, config.vocab_size)
    assert torch.isfinite(output.loss)
