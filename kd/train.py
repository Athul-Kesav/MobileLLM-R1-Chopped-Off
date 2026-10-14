#!/usr/bin/env python3
"""Distill the fully trained MobileLLM-R1-140M into ShiShu.

The two models share a 128,256-token vocabulary and differ in the block
layout. The loss is forward KL from the frozen teacher to the student on
assistant tokens, plus a small next-token term on those same tokens. Sequences
are teacher-forced on the SFT chats. On-policy sampling is a poor fit while
ShiShu is still near a uniform distribution.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Trainer, TrainingArguments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--train-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--ce-coef", type=float, default=0.1)
    parser.add_argument("--layer-coef", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=140)
    return parser.parse_args()


def chat_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> torch.Tensor:
    previous = tokenizer.model_max_length
    tokenizer.model_max_length = max(int(previous or 0), 1_000_000_000)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    finally:
        tokenizer.model_max_length = previous
    if isinstance(encoded, torch.Tensor):
        return encoded.view(-1)
    return encoded["input_ids"].view(-1)


def build_example(tokenizer, messages: list[dict], max_length: int) -> dict[str, list[int]] | None:
    if not messages or messages[-1].get("role") != "assistant":
        return None
    full = chat_ids(tokenizer, messages, False)
    prompt = chat_ids(tokenizer, messages[:-1], True)
    prompt_len = int(prompt.shape[0])
    if prompt_len > full.shape[0] or not torch.equal(full[: min(prompt_len, full.shape[0])], prompt[: min(prompt_len, full.shape[0])]):
        prompt_len = max(1, int(full.shape[0]) // 2)
    overflow = int(full.shape[0] - max_length)
    if overflow > 0:
        full = full[-max_length:]
        prompt_len = max(0, prompt_len - overflow)
    if prompt_len >= full.shape[0]:
        return None
    labels = full.clone()
    labels[:prompt_len] = -100
    if int((labels != -100).sum()) < 1:
        return None
    return {"input_ids": full.tolist(), "labels": labels.tolist()}


def load_rows(paths: list[Path], tokenizer, max_length: int) -> Dataset:
    import json

    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                example = build_example(tokenizer, json.loads(line)["messages"], max_length)
                if example is not None:
                    rows.append(example)
    if not rows:
        raise SystemExit(f"No trainable conversations in {paths}")
    return Dataset.from_list(rows)


def collate(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in features]
    labels = [torch.tensor(row["labels"], dtype=torch.long) for row in features]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _chunked_logsumexp(hidden: torch.Tensor, weight: torch.Tensor, temperature: float, chunk_size: int) -> torch.Tensor:
    hidden = hidden.float()
    weight = weight.float()
    running_max = None
    running_sum = None
    for start in range(0, weight.shape[0], chunk_size):
        logits = hidden.matmul(weight[start : start + chunk_size].T) / temperature
        chunk_max = logits.amax(dim=-1)
        if running_max is None:
            running_max = chunk_max
            running_sum = torch.exp(logits - chunk_max.unsqueeze(-1)).sum(dim=-1)
            continue
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
            logits - new_max.unsqueeze(-1)
        ).sum(dim=-1)
        running_max = new_max
    return running_max + torch.log(running_sum.clamp_min(1e-20))


def chunked_forward_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Mean forward KL(teacher || student) over rows, scaled by T^2."""
    student_lse = _chunked_logsumexp(student_hidden, student_weight, temperature, chunk_size)
    with torch.no_grad():
        teacher_lse = _chunked_logsumexp(teacher_hidden.detach(), teacher_weight.detach(), temperature, chunk_size)
    kl = torch.zeros(student_hidden.shape[0], device=student_hidden.device, dtype=torch.float32)
    student_hidden_f = student_hidden.float()
    teacher_hidden_f = teacher_hidden.detach().float()
    student_weight_f = student_weight.float()
    teacher_weight_f = teacher_weight.detach().float()
    for start in range(0, student_weight.shape[0], chunk_size):
        student_logits = student_hidden_f.matmul(student_weight_f[start : start + chunk_size].T) / temperature
        with torch.no_grad():
            teacher_logits = teacher_hidden_f.matmul(teacher_weight_f[start : start + chunk_size].T) / temperature
            log_teacher = teacher_logits - teacher_lse.unsqueeze(-1)
            teacher_probs = log_teacher.exp()
        log_student = student_logits - student_lse.unsqueeze(-1)
        kl = kl + (teacher_probs * (log_teacher - log_student)).sum(dim=-1)
    return kl.mean() * (temperature ** 2), student_lse


def token_ce(student_hidden: torch.Tensor, student_weight: torch.Tensor, token_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    logsumexp = _chunked_logsumexp(student_hidden, student_weight, temperature=1.0, chunk_size=chunk_size)
    gold = torch.sum(student_hidden.float() * student_weight.float()[token_ids], dim=-1)
    return (logsumexp - gold).mean()


def _backbone(model):
    inner = model.module if hasattr(model, "module") else model
    backbone = inner.base_model if inner.base_model is not inner else inner.model
    return inner, backbone


def hidden_states(model, input_ids, attention_mask):
    inner, backbone = _backbone(model)
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return inner, outputs.last_hidden_state


def block_states(model, input_ids, attention_mask):
    """Return each block output, in order, plus the final normalized hidden state."""
    inner, backbone = _backbone(model)
    states: list[torch.Tensor] = []

    def remember(_module, _inputs, output):
        states.append(output if torch.is_tensor(output) else output[0])

    handles = [layer.register_forward_hook(remember) for layer in backbone.layers]
    try:
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return inner, states, outputs.last_hidden_state


def normalized_block_mse(student_states: list[torch.Tensor], teacher_states: list[torch.Tensor], token_mask: torch.Tensor):
    """Match student blocks 0..T-1 to teacher blocks of the same index.

    The teacher block output already includes its attention residual and its MLP
    residual. The extra student block past the teacher depth is left to the logit loss.
    """
    if len(student_states) < len(teacher_states):
        raise RuntimeError(
            f"Student has {len(student_states)} blocks and teacher has {len(teacher_states)}"
        )
    losses = []
    for index, teacher_state in enumerate(teacher_states):
        student = student_states[index][:, :-1][token_mask]
        teacher = teacher_state[:, :-1][token_mask]
        student = F.layer_norm(student.float(), (student.shape[-1],))
        teacher = F.layer_norm(teacher.detach().float(), (teacher.shape[-1],))
        losses.append(F.mse_loss(student, teacher))
    stacked = torch.stack(losses)
    front = stacked[:11].mean() if stacked.shape[0] >= 11 else stacked.mean()
    tail = stacked[11:].mean() if stacked.shape[0] > 11 else stacked.new_zeros(())
    return stacked.mean(), front, tail


class DistillationTrainer(Trainer):
    def __init__(self, teacher, temperature: float, kl_coef: float, ce_coef: float, layer_coef: float, chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.kl_coef = kl_coef
        self.ce_coef = ce_coef
        self.layer_coef = layer_coef
        self.chunk_size = chunk_size
        self._last_kl = None
        self._last_ce = None
        self._last_layer = None
        self._last_layer_front = None
        self._last_layer_tail = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        if self.layer_coef:
            student, student_states, student_hidden = block_states(model, input_ids, attention_mask)
            with torch.no_grad():
                _, teacher_states, teacher_hidden = block_states(self.teacher, input_ids, attention_mask)
        else:
            student_states = teacher_states = None
            student, student_hidden = hidden_states(model, input_ids, attention_mask)
            with torch.no_grad():
                _, teacher_hidden = hidden_states(self.teacher, input_ids, attention_mask)
        student_tokens = student_hidden[:, :-1]
        teacher_tokens = teacher_hidden[:, :-1]
        target = labels[:, 1:]
        mask = target.ne(-100)
        if int(mask.sum()) < 1:
            zero = student_hidden.sum() * 0.0
            self._last_kl = 0.0
            self._last_ce = 0.0
            self._last_layer = 0.0
            self._last_layer_front = 0.0
            self._last_layer_tail = 0.0
            return (zero, None) if return_outputs else zero
        flat_mask = mask.reshape(-1)
        student_rows = student_tokens.reshape(-1, student_tokens.shape[-1])[flat_mask]
        teacher_rows = teacher_tokens.reshape(-1, teacher_tokens.shape[-1])[flat_mask]
        gold = target.reshape(-1)[flat_mask]
        kl, student_lse = chunked_forward_kl(
            student_rows,
            teacher_rows,
            student.lm_head.weight,
            self.teacher.lm_head.weight,
            self.temperature,
            self.chunk_size,
        )
        if self.temperature == 1.0:
            gold_logit = torch.sum(student_rows.float() * student.lm_head.weight.float()[gold], dim=-1)
            ce = (student_lse - gold_logit).mean()
        else:
            ce = token_ce(student_rows, student.lm_head.weight, gold, self.chunk_size)
        loss = self.kl_coef * kl + self.ce_coef * ce
        self._last_kl = float(kl.detach())
        self._last_ce = float(ce.detach())
        if self.layer_coef:
            layer, front, tail = normalized_block_mse(student_states, teacher_states, mask)
            loss = loss + self.layer_coef * layer
            self._last_layer = float(layer.detach())
            self._last_layer_front = float(front.detach())
            self._last_layer_tail = float(tail.detach())
        return (loss, None) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._last_kl is not None:
            logs["kl"] = self._last_kl
            logs["ce"] = self._last_ce
        if self._last_layer is not None:
            logs["layer"] = self._last_layer
            logs["layer_front"] = self._last_layer_front
            logs["layer_tail"] = self._last_layer_tail
        super().log(logs, start_time)


def prepare_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.decode(tokenizer.vocab_size - 1)
        tokenizer.pad_token_id = tokenizer.vocab_size - 1
    return tokenizer


def load_pair(path: Path, dtype):
    try:
        return load_causal_lm(path, dtype=dtype, attn_implementation="sdpa", local_files_only=True)
    except Exception:
        return load_causal_lm(path, dtype=dtype, local_files_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
        )
    for path in (args.student, args.teacher, *args.train_files):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = prepare_tokenizer(args.teacher)
    student = load_pair(args.student, dtype)
    teacher = load_pair(args.teacher, dtype)
    student_vocab = student.config.get_text_config().vocab_size
    teacher_vocab = teacher.config.get_text_config().vocab_size
    if student_vocab != teacher_vocab:
        raise SystemExit(f"Vocab mismatch: student {student_vocab}, teacher {teacher_vocab}")
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.config.use_cache = False
    teacher.config.use_cache = False

    train_dataset = load_rows(args.train_files, tokenizer, args.seq_len)
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    updates_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * world * args.grad_accum))
    total_updates = args.max_steps if args.max_steps > 0 else math.ceil(updates_per_epoch * args.epochs)
    warmup_steps = int(round(args.warmup_ratio * total_updates))
    if local_rank == 0:
        print(
            f"KD student={args.student} teacher={args.teacher} "
            f"examples={len(train_dataset)} seq={args.seq_len} "
            f"kl={args.kl_coef} ce={args.ce_coef} layer={args.layer_coef} T={args.temperature} "
            f"warmup_steps={warmup_steps}"
        )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=device.type == "cuda",
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = DistillationTrainer(
        teacher=teacher,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        ce_coef=args.ce_coef,
        layer_coef=args.layer_coef,
        chunk_size=args.chunk_size,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda features: collate(features, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
