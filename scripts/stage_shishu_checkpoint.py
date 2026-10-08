#!/usr/bin/env python3
"""Stage a local ShiShu MobileLLM-R1 directory for the training scripts.

Copies the checked-in config and, when present, the original MobileLLM
tokenizer. It does not download or convert weights. Place your
``model.safetensors`` in the destination directory, or pass ``--weights``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shishu_llama import (  # noqa: E402
    ShishuLlamaConfig,
    read_weight_keys,
    register_shishu_llama,
    validate_weight_keys,
    weight_file,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
DEFAULT_CONFIG = ROOT / "configs" / "shishu_mobilellm_r1_140m.json"
DEFAULT_DEST = ROOT / "models" / "MobileLLM-R1-140M-shishu"
DEFAULT_TOKENIZER = ROOT / "models" / "MobileLLM-R1-140M"


def _copy_if_present(source: Path, dest: Path) -> None:
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def stage(dest: Path, config_path: Path, tokenizer_dir: Path, weights: Path | None) -> int:
    register_shishu_llama()
    dest.mkdir(parents=True, exist_ok=True)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    if config.get("model_type") != "shishu-llama":
        raise SystemExit(f"{config_path} is not a shishu-llama config.")
    (dest / "config.json").write_text(config_text if config_text.endswith("\n") else config_text + "\n", encoding="utf-8")
    loaded = ShishuLlamaConfig.from_pretrained(dest)
    print(
        f"Config: {loaded.num_decoder_front} decoder + {loaded.num_mlp} MLP "
        f"+ {loaded.num_decoder_end} decoder-end layers "
        f"(hidden {loaded.hidden_size}, shared MLP pairs {loaded.num_mlp_shared})."
    )
    if loaded.stack_size != loaded.num_hidden_layers:
        print(
            f"Warning: num_hidden_layers={loaded.num_hidden_layers} but the stack has "
            f"{loaded.stack_size} blocks. Training uses the stack counts."
        )

    copied = []
    for name in TOKENIZER_FILES:
        source = tokenizer_dir / name
        if source.is_file() and not (dest / name).is_file():
            _copy_if_present(source, dest / name)
            copied.append(name)
    if copied:
        print(f"Copied tokenizer files from {tokenizer_dir}: {', '.join(copied)}")
    elif not (dest / "tokenizer.json").is_file():
        print(
            f"Tokenizer not found in {dest} or {tokenizer_dir}. "
            "Copy tokenizer.json from the original MobileLLM-R1-140M snapshot."
        )

    if weights is not None:
        if not weights.is_file():
            raise SystemExit(f"Weights file does not exist: {weights}")
        target_name = "model.safetensors" if weights.suffix == ".safetensors" else weights.name
        shutil.copy2(weights, dest / target_name)
        print(f"Copied weights to {dest / target_name}")

    current = weight_file(dest)
    if current is None:
        print(f"Weights are not loaded yet. Copy your file to:\n  {dest / 'model.safetensors'}")
        return 0

    try:
        keys = read_weight_keys(dest)
        validate_weight_keys(loaded, keys)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Weights at {current} match the ShiShu layout ({len(keys)} tensors).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--weights", type=Path, default=None, help="Optional model.safetensors to copy in.")
    args = parser.parse_args()
    raise SystemExit(stage(args.dest, args.config, args.tokenizer_dir, args.weights))


if __name__ == "__main__":
    main()
