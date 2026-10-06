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


def load_model(
    path: str | Path,
    dtype: Any | None = None,
    device: Any = "cpu",
    *,
    output_loading_info: bool = False,
    trust_remote_code: bool = False,
) -> Any:
    """Load a local Hugging Face causal LM, leaving room for a future builder."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "output_loading_info": output_loading_info,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    result = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
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
