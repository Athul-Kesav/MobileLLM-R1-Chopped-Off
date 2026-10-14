"""Model loading, transformation, and checkpoint persistence seams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def apply_model_transform(model: Any, cfg: dict[str, Any] | None = None) -> Any:
    """Apply the configured model transform.

    ``none`` is deliberately the only supported transform until the surgery
    phase defines a concrete architecture and serialization contract.
    """
    transform = (cfg or {}).get("model_transform", "none")
    if transform in (None, "none", "identity"):
        return model
    raise ValueError(f"Unsupported model_transform: {transform}")


def align_causal_lm_backbone(model: Any) -> Any:
    """Point ``base_model`` at the submodule that actually holds the backbone.

    ``Llama4ForCausalLM`` keeps the text stack on ``self.model`` while declaring
    ``base_model_prefix = "language_model"``. ``model.base_model`` then falls
    back to the causal LM, and TRL's fused head crashes reading
    ``last_hidden_state`` from ``MoeCausalLMOutputWithPast``.
    """
    prefix = getattr(type(model), "base_model_prefix", None)
    children = {name for name, _ in model.named_children()}
    if prefix and prefix not in children and "model" in children:
        model.base_model_prefix = "model"
    return model


def disable_dense_router_loss(model: Any) -> Any:
    """Turn off the MoE load-balancing loss on a dense Llama4 checkpoint.

    MobileLLM-R1-140M-base still sets ``router_aux_loss_coef`` and declares
    ``output_router_logits``, so TRL enables the auxiliary loss and then reads
    ``num_experts``. This checkpoint has ``num_local_experts`` 0 and no MoE
    layers, and its config uses that name rather than ``num_experts``.
    """
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    if not hasattr(config, "router_aux_loss_coef"):
        return model
    n_experts = getattr(config, "num_experts", None)
    if n_experts is None:
        n_experts = getattr(config, "num_local_experts", None)
    moe_layers = getattr(config, "moe_layers", None)
    has_experts = isinstance(n_experts, int) and n_experts > 0
    has_moe_layers = isinstance(moe_layers, (list, tuple)) and len(moe_layers) > 0
    if has_experts or has_moe_layers:
        return model
    config.router_aux_loss_coef = 0.0
    if hasattr(config, "output_router_logits"):
        config.output_router_logits = False
    return model


def checkpoint_has_weights(path: str | Path) -> bool:
    root = Path(path)
    return any(
        (root / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )


def load_causal_lm(path: str | Path, **kwargs: Any) -> Any:
    """Load a local causal LM, including a ShiShu MobileLLM checkpoint."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.shishu_llama import ShishuLlamaForCausalLM, register_shishu_llama

    register_shishu_llama()
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if getattr(config, "model_type", None) == "shishu-llama":
        model = ShishuLlamaForCausalLM.from_pretrained(str(path), **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    model = align_causal_lm_backbone(model)
    return disable_dense_router_loss(model)


def load_model(
    path: str | Path,
    dtype: Any | None = None,
    device: Any = "cpu",
    *,
    output_loading_info: bool = False,
    trust_remote_code: bool = False,
) -> Any:
    """Load a local Hugging Face causal LM."""
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "output_loading_info": output_loading_info,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    result = load_causal_lm(path, **kwargs)
    if output_loading_info:
        model, info = result
        return model.to(device), info
    return result.to(device)


def save_model(
    model: Any,
    tokenizer: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save model/tokenizer in HF format and write explicit experiment metadata."""
    root = Path(path)
    model_dir, tokenizer_dir = root / "model", root / "tokenizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    payload = dict(metadata or {})
    payload.setdefault("model_transform", "none")
    (root / "metadata.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return root
