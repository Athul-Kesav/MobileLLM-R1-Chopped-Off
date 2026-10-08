"""Register ShishuLlama with Hugging Face Auto classes."""

from transformers import AutoConfig, AutoModelForCausalLM

from .checkpoint import (
    layer_kinds,
    read_weight_keys,
    required_weight_keys,
    validate_weight_keys,
    weight_file,
)
from .configuration_shishu_llama import ShishuLlamaConfig
from .modeling_shishu_llama import ShishuLlamaForCausalLM, ShishuLlamaModel


def register_shishu_llama() -> None:
    AutoConfig.register("shishu-llama", ShishuLlamaConfig, exist_ok=True)
    AutoModelForCausalLM.register(ShishuLlamaConfig, ShishuLlamaForCausalLM, exist_ok=True)


__all__ = [
    "ShishuLlamaConfig",
    "ShishuLlamaForCausalLM",
    "ShishuLlamaModel",
    "layer_kinds",
    "read_weight_keys",
    "register_shishu_llama",
    "required_weight_keys",
    "validate_weight_keys",
    "weight_file",
]
