#!/usr/bin/env python3
"""Score ShiShu and MobileLLM-R1-140M-base on the same held-out data.

Metrics, measured with identical prompts and token budgets:
  perplexity  — assistant-token NLL on held-out Tulu-3 and OpenMathReasoning
  accuracy    — exact match on GSM8K and on held-out OpenMath boxed answers
  latency     — median batch-1 generation time
  throughput  — generated tokens per second at a fixed batch size
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from compare.common import (
    REPO_ROOT,
    answers_match,
    extract_final,
    gsm8k_gold,
    load_env,
    read_jsonl,
    resolve,
)

PROMPTS = (
    "Compute 17 * 23.",
    "What is the sum of the first 10 positive integers?",
    "If a train travels 120 km in 2 hours, what is its average speed in km/h?",
    "Solve 3x + 5 = 20 for x.",
    "How many prime numbers are there between 10 and 30?",
    "A rectangle has width 4 and length 9. What is its area?",
    "Write the next number in the sequence 2, 4, 8, 16.",
    "What is 15% of 80?",
)
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def checkpoint_map(env: dict[str, str]) -> dict[str, dict[str, Path]]:
    runs = resolve(env, "RUN_ROOT", "compare/runs")
    return {
        "shishu": {
            "init": resolve(env, "SHISHU_DIR", "models/MobileLLM-R1-140M-shishu"),
            "general": runs / "shishu" / "general",
            "reasoning": runs / "shishu" / "reasoning",
        },
        "base": {
            "init": resolve(env, "BASE_DIR", "models/MobileLLM-R1-140M-base"),
            "general": runs / "base" / "general",
            "reasoning": runs / "base" / "reasoning",
        },
    }


def has_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(path: Path, fallback: Path):
    source = path if (path / "tokenizer.json").is_file() else fallback
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.decode(tokenizer.vocab_size - 1)
        tokenizer.pad_token_id = tokenizer.vocab_size - 1
    return tokenizer


def load_model(path: Path, device: torch.device):
    from src.model_io import load_causal_lm

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = load_causal_lm(path, dtype=dtype, attn_implementation="sdpa")
    except Exception:
        model = load_causal_lm(path, dtype=dtype)
    model.to(device)
    model.eval()
    return model


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def context_window(model) -> int | None:
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    positions = getattr(config, "max_position_embeddings", None)
    if isinstance(positions, int) and positions > 0:
        return positions
    return None


def fit_length(requested: int, window: int | None, reserved: int = 0) -> int:
    """Longest input that stays inside the model position table after ``reserved`` new tokens."""
    if window is None:
        return max(1, requested)
    return max(1, min(requested, window - reserved))


def chat_ids(
    tokenizer,
    messages: list[dict],
    add_generation_prompt: bool,
    max_length: int | None = None,
    truncation_side: str = "right",
) -> torch.Tensor:
    previous_side = tokenizer.truncation_side
    previous_max = tokenizer.model_max_length
    tokenizer.truncation_side = truncation_side
    try:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": "pt",
        }
        if max_length is None:
            # Measuring the uncropped length must not trip the model_max_length warning.
            tokenizer.model_max_length = max(int(previous_max or 0), 1_000_000_000)
        else:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_length
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    finally:
        tokenizer.truncation_side = previous_side
        tokenizer.model_max_length = previous_max
    if isinstance(encoded, torch.Tensor):
        return encoded
    return encoded["input_ids"]


def perplexity(model, tokenizer, rows: list[dict], device: torch.device, max_length: int, max_seqs: int) -> dict:
    nll_sum = 0.0
    tokens = 0
    used = 0
    for row in rows[:max_seqs]:
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            continue
        full = chat_ids(tokenizer, messages, False)
        prompt = chat_ids(tokenizer, messages[:-1], True)
        prompt_len = int(prompt.shape[1])
        if prompt_len > full.shape[1] or not torch.equal(full[0, : min(prompt_len, full.shape[1])], prompt[0, : min(prompt_len, full.shape[1])]):
            prompt_len = max(1, full.shape[1] // 2)
        overflow = int(full.shape[1] - max_length)
        if overflow > 0:
            # Keep the end of the conversation so the assistant span stays in the window.
            full = full[:, -max_length:]
            prompt_len = max(0, prompt_len - overflow)
        if prompt_len >= full.shape[1]:
            continue
        labels = full.clone()
        labels[:, :prompt_len] = -100
        full = full.to(device)
        labels = labels.to(device)
        with torch.inference_mode():
            loss = model(input_ids=full, labels=labels).loss
        count = int((labels != -100).sum().item())
        if count < 1 or not math.isfinite(float(loss)):
            continue
        nll_sum += float(loss) * count
        tokens += count
        used += 1
    mean = nll_sum / tokens if tokens else float("nan")
    return {
        "sequences": used,
        "assistant_tokens": tokens,
        "loss": mean if tokens else None,
        "perplexity": math.exp(mean) if tokens else None,
    }


def generate_one(model, tokenizer, messages: list[dict], device: torch.device, max_new: int, max_length: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_limit = fit_length(max_length, context_window(model), reserved=max_new)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_limit)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_problems(model, tokenizer, problems: list[tuple[list[dict], str]], device, max_new: int, max_length: int) -> dict:
    correct = 0
    for messages, gold in problems:
        prediction = generate_one(model, tokenizer, messages, device, max_new, max_length)
        correct += int(answers_match(prediction, gold))
    total = len(problems)
    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
    }


def reasoning_problems(rows: list[dict], limit: int) -> list[tuple[list[dict], str]]:
    problems = []
    for row in rows:
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            continue
        gold = extract_final(messages[-1]["content"])
        if not gold:
            continue
        problems.append((messages[:-1], gold))
        if len(problems) >= limit:
            break
    return problems


def gsm8k_problems(path: Path, limit: int) -> list[tuple[list[dict], str]]:
    if not path.is_file():
        return []
    problems = []
    for row in read_jsonl(path)[:limit]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        problems.append((messages, gsm8k_gold(row["answer"])))
    return problems


def user_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def latency(model, tokenizer, device: torch.device, repeats: int, max_new: int, max_length: int) -> dict:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(repeats)]
    generate_one(model, tokenizer, user_messages(prompts[0]), device, max_new, max_length)
    samples = []
    for text in prompts:
        sync(device)
        start = time.perf_counter()
        prediction = generate_one(model, tokenizer, user_messages(text), device, max_new, max_length)
        sync(device)
        elapsed = time.perf_counter() - start
        produced = max(1, len(tokenizer.encode(prediction, add_special_tokens=False)))
        samples.append({"seconds": elapsed, "ms_per_token": 1000.0 * elapsed / produced, "tokens": produced})
    ordered = sorted(sample["seconds"] for sample in samples)
    mid = ordered[len(ordered) // 2]
    per_token = sorted(sample["ms_per_token"] for sample in samples)
    return {
        "batch_size": 1,
        "max_new_tokens": max_new,
        "repeats": repeats,
        "median_latency_s": mid,
        "median_ms_per_token": per_token[len(per_token) // 2],
    }


def throughput(model, tokenizer, device: torch.device, batch_size: int, max_new: int, max_length: int) -> dict:
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(user_messages(PROMPTS[i % len(PROMPTS)]), tokenize=False, add_generation_prompt=True)
        for i in range(batch_size)
    ]
    prompt_limit = fit_length(max_length, context_window(model), reserved=max_new)
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=prompt_limit)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        model.generate(**encoded, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    new_tokens = output[:, encoded["input_ids"].shape[1] :]
    count = int((new_tokens != tokenizer.pad_token_id).sum().item())
    tokenizer.padding_side = "right"
    return {
        "batch_size": batch_size,
        "max_new_tokens": max_new,
        "generated_tokens": count,
        "seconds": elapsed,
        "tokens_per_s": count / elapsed,
    }


def evaluate_checkpoint(path: Path, tokenizer_fallback: Path, env: dict[str, str], data_root: Path, device: torch.device) -> dict:
    tokenizer = load_tokenizer(path, tokenizer_fallback)
    model = load_model(path, device)
    max_length = fit_length(int(env.get("SEQ_LEN", "2048")), context_window(model))
    general_eval = read_jsonl(data_root / "general" / "eval.jsonl")
    reasoning_eval = read_jsonl(data_root / "reasoning" / "eval.jsonl")
    accuracy_n = int(env.get("ACCURACY_PROBLEMS", "32"))
    metrics = {
        "path": str(path),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "perplexity": {
            "general": perplexity(model, tokenizer, general_eval, device, max_length, int(env.get("PPL_MAX_SEQS", "64"))),
            "reasoning": perplexity(model, tokenizer, reasoning_eval, device, max_length, int(env.get("PPL_MAX_SEQS", "64"))),
        },
        "accuracy": {
            "gsm8k": score_problems(
                model,
                tokenizer,
                gsm8k_problems(data_root / "gsm8k" / "test.jsonl", accuracy_n),
                device,
                int(env.get("ACCURACY_MAX_NEW_TOKENS", "256")),
                max_length,
            ),
            "reasoning_heldout": score_problems(
                model,
                tokenizer,
                reasoning_problems(reasoning_eval, accuracy_n),
                device,
                int(env.get("ACCURACY_MAX_NEW_TOKENS", "256")),
                max_length,
            ),
        },
        "latency": latency(
            model,
            tokenizer,
            device,
            int(env.get("LATENCY_REPEATS", "8")),
            int(env.get("LATENCY_NEW_TOKENS", "32")),
            max_length,
        ),
        "throughput": throughput(
            model,
            tokenizer,
            device,
            int(env.get("THROUGHPUT_BATCH", "4")),
            int(env.get("THROUGHPUT_NEW_TOKENS", "32")),
            max_length,
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# ShiShu vs MobileLLM-R1-140M-base",
        "",
        "Both models ran the same Facebook post-training loop: Tulu-3 general SFT, then OpenMathReasoning SFT.",
        "Accuracy, perplexity, latency, and throughput use the same held-out rows, prompts, and token budgets.",
        "",
        f"Device: `{payload['device']}`",
        "",
        "| Model | Stage | PPL general | PPL reasoning | GSM8K acc | Reasoning acc | Median latency (s) | Throughput (tok/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, stages in payload["models"].items():
        for stage, metrics in stages.items():
            if "error" in metrics:
                lines.append(f"| {model_name} | {stage} |  |  |  |  |  | {metrics['error']} |")
                continue
            if metrics.get("status") == "missing":
                lines.append(f"| {model_name} | {stage} |  |  |  |  |  | missing checkpoint |")
                continue
            ppl = metrics["perplexity"]
            acc = metrics["accuracy"]
            lines.append(
                "| {model} | {stage} | {pg} | {pr} | {ag} | {ar} | {lat} | {thr} |".format(
                    model=model_name,
                    stage=stage,
                    pg=fmt(ppl["general"]["perplexity"]),
                    pr=fmt(ppl["reasoning"]["perplexity"]),
                    ag=fmt(acc["gsm8k"]["accuracy"]),
                    ar=fmt(acc["reasoning_heldout"]["accuracy"]),
                    lat=fmt(metrics["latency"]["median_latency_s"]),
                    thr=fmt(metrics["throughput"]["tokens_per_s"]),
                )
            )
    lines.extend(["", "Raw numbers are in `metrics.json` beside this file.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / "compare/configs/default.env")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Score this checkpoint only.")
    args = parser.parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    env = load_env(args.env)
    set_seed(int(env.get("SEED", "140")))
    data_root = resolve(env, "COMPARE_DATA", "compare/data")
    if not (data_root / "general" / "eval.jsonl").is_file():
        raise SystemExit(f"Missing {data_root}. Run: bash compare/run.sh prepare")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (resolve(env, "RESULT_ROOT", "compare/results") / stamp)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    payload = {
        "created_utc": stamp,
        "device": str(device),
        "env": str(args.env),
        "models": {},
    }
    if args.checkpoint is not None:
        path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
        if not has_weights(path):
            raise SystemExit(f"Missing weights in {path}. Wait until KD finishes writing kd/runs/shishu.")
        fallback = resolve(env, "CHAT_TEMPLATE", "models/MobileLLM-R1-140M/chat_template.jinja").parent
        print(f"== checkpoint: {path}")
        payload["models"]["shishu"] = {
            "kd": evaluate_checkpoint(path, fallback, env, data_root, device)
        }
        (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_report(payload, out / "report.md")
        print(f"Wrote {out / 'report.md'}")
        return
    for model_name, stages in checkpoint_map(env).items():
        payload["models"][model_name] = {}
        fallback = stages["init"]
        for stage, path in stages.items():
            print(f"== {model_name} {stage}: {path}")
            if not has_weights(path):
                payload["models"][model_name][stage] = {"status": "missing", "path": str(path)}
                print(f"   missing weights, skipped")
                continue
            try:
                payload["models"][model_name][stage] = evaluate_checkpoint(path, fallback, env, data_root, device)
            except Exception as exc:
                payload["models"][model_name][stage] = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "path": str(path)}
                print(f"   failed: {exc}", file=sys.stderr)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload, out / "report.md")
    print(f"Wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
