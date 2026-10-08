"""Checkpoint layout expected by ShishuLlamaForCausalLM."""

from __future__ import annotations

from pathlib import Path

from .configuration_shishu_llama import ShishuLlamaConfig


def layer_kinds(config: ShishuLlamaConfig) -> list[str]:
    return (
        ["decoder"] * config.num_decoder_front
        + ["mlp"] * config.num_mlp
        + ["decoder"] * config.num_decoder_end
    )


def _suffixes(kind: str) -> tuple[str, ...]:
    mlp = (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
    )
    if kind == "mlp":
        return mlp
    return (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "input_layernorm.weight",
        *mlp,
    )


def required_weight_keys(config: ShishuLlamaConfig) -> list[str]:
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    for index, kind in enumerate(layer_kinds(config)):
        keys.extend(f"model.layers.{index}.{suffix}" for suffix in _suffixes(kind))
    if not config.tie_word_embeddings:
        keys.append("lm_head.weight")
    return keys


def unexpected_mlp_attention_keys(config: ShishuLlamaConfig, keys: set[str]) -> list[str]:
    """MLP-only blocks must not carry attention or the pre-attention norm."""
    found = []
    offset = config.num_decoder_front
    for index in range(offset, offset + config.num_mlp):
        for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight"):
            name = f"model.layers.{index}.{suffix}"
            if name in keys:
                found.append(name)
    return found


def weight_file(path: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def read_weight_keys(path: Path) -> set[str]:
    weight = weight_file(path)
    if weight is None:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {path}"
        )
    if weight.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(weight, framework="pt") as handle:
            return set(handle.keys())
    import torch

    state = torch.load(weight, map_location="cpu", weights_only=True)
    return set(state.keys())


def validate_weight_keys(config: ShishuLlamaConfig, keys: set[str]) -> None:
    missing = [key for key in required_weight_keys(config) if key not in keys]
    unexpected = unexpected_mlp_attention_keys(config, keys)
    if missing or unexpected:
        details = []
        if missing:
            preview = ", ".join(missing[:8])
            details.append(f"missing {len(missing)} required tensors ({preview})")
        if unexpected:
            details.append(
                "MLP-only layers still have attention tensors: "
                + ", ".join(unexpected[:4])
            )
        raise ValueError(
            "Checkpoint does not match the ShiShu MobileLLM-R1 layout: "
            + "; ".join(details)
        )
