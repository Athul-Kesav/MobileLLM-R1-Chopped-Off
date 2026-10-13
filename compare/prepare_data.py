#!/usr/bin/env python3
"""Split the local Facebook SFT subsets into identical train and eval files.

General SFT is a slice of allenai/tulu-3-sft-olmo-2-mixture-0225.
Reasoning SFT is a slice of nvidia/OpenMathReasoning.
GSM8K test questions are downloaded when the Hub is reachable; accuracy
falls back to the held-out reasoning set when they are not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare.common import REPO_ROOT, load_env, read_jsonl, resolve, write_jsonl


def split_rows(rows: list[dict], eval_rows: int, seed: int) -> tuple[list[dict], list[dict]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    holdout = min(eval_rows, max(1, len(rows) // 5))
    if len(rows) - holdout < 1:
        raise SystemExit(f"Need at least 2 rows to split, found {len(rows)}.")
    eval_idx = set(order[:holdout])
    train = [rows[i] for i in range(len(rows)) if i not in eval_idx]
    held = [rows[i] for i in range(len(rows)) if i in eval_idx]
    return train, held


def maybe_gsm8k(dest: Path, rows: int, seed: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"GSM8K skipped: {exc}", file=sys.stderr)
        return 0
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as exc:
        print(f"GSM8K download skipped: {exc}", file=sys.stderr)
        return 0
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    chosen = []
    for index in indices[:rows]:
        row = dataset[index]
        chosen.append({"question": row["question"], "answer": row["answer"]})
    write_jsonl(dest, chosen)
    print(f"gsm8k: {len(chosen)} test questions -> {dest}")
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / "compare/configs/default.env")
    args = parser.parse_args()
    env = load_env(args.env)
    data_root = resolve(env, "COMPARE_DATA", "compare/data")
    seed = int(env.get("SEED", "140"))
    eval_rows = int(env.get("EVAL_ROWS", "128"))

    sources = {
        "general": (
            resolve(env, "SOURCE_GENERAL", "data/sft/general/train.jsonl"),
            "allenai/tulu-3-sft-olmo-2-mixture-0225",
            "Facebook general SFT. Paper uses 866K rows; this run uses the local slice.",
        ),
        "reasoning": (
            resolve(env, "SOURCE_REASONING", "data/sft/reasoning/train.jsonl"),
            "nvidia/OpenMathReasoning",
            "Facebook reasoning SFT also includes OpenScienceReasoning-2 and OpenCodeReasoning-2. This run uses OpenMathReasoning.",
        ),
    }
    manifest: dict = {"seed": seed, "stages": {}}
    for name, (source, upstream, note) in sources.items():
        if not source.is_file():
            raise SystemExit(
                f"Missing {source}. Build it with "
                "python scripts/prepare_training_data.py --stages sft/general sft/reasoning"
            )
        rows = read_jsonl(source)
        train, held = split_rows(rows, eval_rows, seed if name == "general" else seed + 1)
        write_jsonl(data_root / name / "train.jsonl", train)
        write_jsonl(data_root / name / "eval.jsonl", held)
        manifest["stages"][name] = {
            "upstream": upstream,
            "source": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
            "train_rows": len(train),
            "eval_rows": len(held),
            "note": note,
        }
        print(f"{name}: {len(train)} train / {len(held)} eval from {source}")

    gsm_path = data_root / "gsm8k" / "test.jsonl"
    gsm_count = maybe_gsm8k(gsm_path, int(env.get("GSM8K_ROWS", "64")), seed)
    manifest["gsm8k"] = {
        "upstream": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "rows": gsm_count,
        "path": str(gsm_path.relative_to(REPO_ROOT)) if gsm_count else None,
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
