"""Shared path, env, and answer-normalization helpers for the comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def load_env(path: str | Path) -> dict[str, str]:
    """Read KEY=VALUE lines. Later files do not exist; comments are skipped."""
    env: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def resolve(env: dict[str, str], key: str, default: str = "") -> Path:
    value = env.get(key, default)
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    text = text.strip().strip("`").rstrip(".")
    if "=" in text:
        text = text.split("=")[-1].strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def extract_final(text: str) -> str:
    boxed = BOXED.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    numbers = NUMBER.findall(text.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_answer(answer)


def answers_match(prediction: str, gold: str) -> bool:
    predicted = extract_final(prediction)
    return bool(predicted) and predicted == normalize_answer(gold)


def self_check() -> None:
    assert normalize_answer("1,024") == "1024"
    assert normalize_answer("72.0") == "72"
    assert extract_final(r"so \\boxed{42}") == "42"
    assert extract_final("the total is 1,000.") == "1000"
    assert gsm8k_gold("work\n#### 72") == "72"
    assert answers_match(r"\boxed{72}", "72.0")
    assert not answers_match("the answer is 3", "4")
    print("compare.common self-check ok")
