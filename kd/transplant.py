#!/usr/bin/env python3
"""Build a ShiShu student by copying the fully trained MobileLLM-R1-140M.

Blocks 0-10 are copied whole. Blocks 11-14 keep the teacher's MLP and drop its
attention. Block 15 copies that last MLP and then zeros ``down_proj``, so the
extra block starts as a residual identity instead of scrambling the copied stream.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.shishu_llama.checkpoint import validate_weight_keys
from src.shishu_llama.configuration_shishu_llama import ShishuLlamaConfig

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
CONFIG = _ROOT / "configs" / "shishu_mobilellm_r1_140m.json"
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=_ROOT / "models" / "MobileLLM-R1-140M")
    parser.add_argument("--dest", type=Path, default=_ROOT / "kd" / "runs" / "shishu-transplant")
    parser.add_argument("--config", type=Path, default=CONFIG)
    return parser.parse_args()


def copy_mlp(teacher: dict[str, torch.Tensor], student: dict[str, torch.Tensor], teacher_idx: int, student_idx: int) -> None:
    source = f"model.layers.{teacher_idx}"
    target = f"model.layers.{student_idx}"
    student[f"{target}.post_attention_layernorm.weight"] = teacher[f"{source}.post_attention_layernorm.weight"].clone()
    for name in MLP:
        student[f"{target}.mlp.{name}.weight"] = teacher[f"{source}.feed_forward.{name}.weight"].clone()


def transplant(teacher_dir: Path, dest: Path, config_path: Path) -> None:
    weights = teacher_dir / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"Missing {weights}")
    if not config_path.is_file():
        raise SystemExit(f"Missing {config_path}")
    teacher = load_file(str(weights))
    student: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": teacher["model.embed_tokens.weight"].clone(),
        "model.norm.weight": teacher["model.norm.weight"].clone(),
    }
    for index in range(11):
        source = f"model.layers.{index}"
        for name in ATTENTION:
            student[f"{source}.self_attn.{name}.weight"] = teacher[f"{source}.self_attn.{name}.weight"].clone()
        student[f"{source}.input_layernorm.weight"] = teacher[f"{source}.input_layernorm.weight"].clone()
        copy_mlp(teacher, student, index, index)
    for index in range(11, 15):
        copy_mlp(teacher, student, index, index)
    copy_mlp(teacher, student, 14, 15)
    student["model.layers.15.mlp.down_proj.weight"].zero_()

    config = ShishuLlamaConfig.from_pretrained(str(config_path))
    validate_weight_keys(config, set(student))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, dest / "config.json")
    for name in TOKENIZER_FILES:
        source = teacher_dir / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    save_file(student, str(dest / "model.safetensors"))

    copied = student["model.layers.0.self_attn.q_proj.weight"].float()
    original = teacher["model.layers.0.self_attn.q_proj.weight"].float()
    cosine = torch.nn.functional.cosine_similarity(copied.flatten(), original.flatten(), dim=0)
    tail = student["model.layers.15.mlp.down_proj.weight"]
    print(
        f"Wrote {dest / 'model.safetensors'} "
        f"({sum(tensor.numel() for tensor in student.values())} parameters) "
        f"layer0 q_proj cosine {float(cosine):.6f} "
        f"layer15 down_proj abs {float(tail.abs().sum()):.1f}"
    )


def main() -> None:
    args = parse_args()
    transplant(args.teacher, args.dest, args.config)


if __name__ == "__main__":
    main()
