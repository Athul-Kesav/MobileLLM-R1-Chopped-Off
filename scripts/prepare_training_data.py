#!/usr/bin/env python3
"""Stream Hugging Face datasets into the layout the training stages read.

Pre-/mid-training (one directory per node, one file per rank on that node):

    <data_root>/pretrain/phase1/<node>/<source>:<weight>/<chunk>.jsonl   {"text": ...}

SFT (read by sft/sft.py through ``load_dataset(<dir>)``):

    <data_root>/sft/general/train.jsonl     {"messages": [...]}
    <data_root>/sft/reasoning/train.jsonl   {"messages": [...]}

Sizes are controlled per run, so the same script builds a small local
pipeline-check dataset or a large shard for a real run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    config: str | None
    split: str
    text_field: str
    weight: float


# Ungated subsets of the MobileLLM-R1 mixes. Weights follow the paper's mix
# ratios for these sources, renormalized by the loader.
MIXES: dict[str, list[Source]] = {
    "pretrain/phase1": [
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 63.75),
        Source("open_web_math", "open-web-math/open-web-math", None, "train", "text", 6.93),
        Source("wikipedia", "wikimedia/wikipedia", "20231101.en", "train", "text", 5.03),
    ],
    "pretrain/phase2": [
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 54.83),
        Source("open_web_math", "open-web-math/open-web-math", None, "train", "text", 23.33),
        Source("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text", 8.01),
    ],
    "midtrain/phase1": [
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 37.03),
        Source("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text", 11.63),
        Source("open_web_math", "open-web-math/open-web-math", None, "train", "text", 3.45),
    ],
    "midtrain/phase2": [
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 6.51),
        Source("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text", 29.10),
        Source("open_web_math", "open-web-math/open-web-math", None, "train", "text", 19.40),
    ],
}

GENERAL_SFT = ("allenai/tulu-3-sft-olmo-2-mixture-0225", None, "train")
REASONING_SFT = ("nvidia/OpenMathReasoning", None, "cot")


def stream(path: str, config: str | None, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    return iter(load_dataset(path, name=config, split=split, streaming=True))


def write_lm_stage(stage: str, root: Path, nodes: int, chunks: int, docs: int, offset: int) -> None:
    """Round-robin ``docs`` documents per source over nodes x chunks files."""
    for source in MIXES[stage]:
        dirs = [root / stage / str(node) / f"{source.name}:{source.weight}" for node in range(1, nodes + 1)]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        handles = [
            (directory / f"{chunk:03d}.jsonl").open("w", encoding="utf-8")
            for directory in dirs
            for chunk in range(chunks)
        ]
        written = 0
        try:
            rows = islice(stream(source.path, source.config, source.split), offset, offset + docs)
            for index, row in enumerate(rows):
                text = row.get(source.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                handles[index % len(handles)].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
        finally:
            for handle in handles:
                handle.close()
        if written < len(handles):
            raise SystemExit(f"{stage}/{source.name}: only {written} docs for {len(handles)} files.")
        print(f"{stage}/{source.name}: {written} docs -> {nodes} nodes x {chunks} files")


def general_messages(rows: Iterable[dict]) -> Iterator[list[dict]]:
    for row in rows:
        messages = row.get("messages")
        if messages:
            yield [{"role": m["role"], "content": m["content"]} for m in messages]


def reasoning_messages(rows: Iterable[dict]) -> Iterator[list[dict]]:
    for row in rows:
        problem, solution = row.get("problem"), row.get("generated_solution")
        if problem and solution:
            yield [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]


def write_sft(name: str, spec: tuple, convert, root: Path, rows: int) -> None:
    out = root / "sft" / name
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for messages in islice(convert(stream(*spec)), rows):
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    print(f"sft/{name}: {count} conversations -> {out / 'train.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Repo-relative mix directory (default: ./data).",
    )
    parser.add_argument("--nodes", type=int, default=1, help="Training nodes; one shard directory per node.")
    parser.add_argument(
        "--chunks", type=int, default=1,
        help="Files per source per node. Must divide GPUs per node (3 locally, 8 on an A100 node).",
    )
    parser.add_argument("--docs", type=int, default=3000, help="Documents per source per LM stage.")
    parser.add_argument("--sft_rows", type=int, default=2000, help="Conversations per SFT stage.")
    parser.add_argument(
        "--stages", nargs="+",
        default=[*MIXES, "sft/general", "sft/reasoning"],
        choices=[*MIXES, "sft/general", "sft/reasoning"],
    )
    args = parser.parse_args()

    for offset, stage in enumerate(s for s in args.stages if s in MIXES):
        # Different offsets keep phases from training on identical documents.
        write_lm_stage(stage, args.data_root, args.nodes, args.chunks, args.docs, offset * args.docs)
    if "sft/general" in args.stages:
        write_sft("general", GENERAL_SFT, general_messages, args.data_root, args.sft_rows)
    if "sft/reasoning" in args.stages:
        write_sft("reasoning", REASONING_SFT, reasoning_messages, args.data_root, args.sft_rows)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # Streaming datasets leave background threads that crash interpreter teardown.
    os._exit(0)
