"""Run a local-only forward, backward, checkpoint reload, and generation test."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.download_model import verify_model_directory
from src.model_io import load_model, save_model


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def load_dataset(path: str | Path) -> list[str]:
    examples: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid dataset row at {path}:{line_number}") from exc
            if not isinstance(text, str) or not text:
                raise ValueError(f"Dataset row at {path}:{line_number} has invalid text")
            examples.append(text)
    if not examples:
        raise ValueError(f"Dataset is empty: {path}")
    return examples


def mask_padding_labels(input_ids: Any, attention_mask: Any) -> Any:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


def validate_forward(output: Any, batch_size: int, sequence_length: int, vocab_size: int) -> float:
    import torch

    logits = output.logits
    if tuple(logits.shape) != (batch_size, sequence_length, vocab_size):
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    loss = output.loss
    if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
        raise ValueError("Forward pass produced a missing, non-scalar, or non-finite loss")
    return float(loss.detach().cpu())


def _device_and_dtype(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch

    requested = config.get("device", "auto")
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    dtype_name = config.get("dtype", "auto")
    if dtype_name == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, str(dtype_name).replace("torch.", ""))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device, dtype


def _versions() -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import __version__ as hub_version

    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": hub_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    config_root = Path(config_path).resolve().parent.parent
    report_path = Path(config["report_file"])
    if not report_path.is_absolute():
        report_path = config_root / report_path
    report: dict[str, Any] = {"status": "failed", "download_status": "failed"}
    stage = "preflight"
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer

        model_path = Path(config["model_path"])
        tokenizer_path = Path(config["tokenizer_path"])
        train_file = Path(config["train_file"])
        output_dir = Path(config["output_dir"])
        model_path = model_path if model_path.is_absolute() else config_root / model_path
        tokenizer_path = tokenizer_path if tokenizer_path.is_absolute() else config_root / tokenizer_path
        train_file = train_file if train_file.is_absolute() else config_root / train_file
        verification = verify_model_directory(model_path)
        report["download_status"] = "passed"
        versions = _versions()
        device, dtype = _device_and_dtype(config)
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        report.update(versions, device=str(device), dtype=str(dtype), model_path=str(model_path),
                      tokenizer_path=str(tokenizer_path), resolved_revision=config.get("model_revision"),
                      **verification, seed=config["seed"])

        stage = "load_model"
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=config.get("trust_remote_code", False)
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model, loading_info = load_model(
            model_path, dtype=dtype, device=device, output_loading_info=True,
            trust_remote_code=config.get("trust_remote_code", False),
        )
        tied_allowlist = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
        missing = set(loading_info.get("missing_keys", [])) - tied_allowlist
        unexpected = set(loading_info.get("unexpected_keys", [])) - tied_allowlist
        if missing or unexpected or loading_info.get("mismatched_keys"):
            raise RuntimeError(f"Incompatible checkpoint keys: {loading_info}")
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer and model vocabulary sizes are incompatible")
        model.train()
        report.update(model_class=type(model).__name__, tokenizer_class=type(tokenizer).__name__,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

        stage = "prepare_data"
        texts = load_dataset(train_file)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=config["max_seq_length"],
                            return_tensors="pt")
        labels = mask_padding_labels(encoded["input_ids"], encoded["attention_mask"])
        if not (labels == -100).any():
            raise ValueError("Batch did not exercise padding label masking")
        loader = DataLoader(TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels),
                            batch_size=config["batch_size"], shuffle=False)
        batch = next(iter(loader))

        stage = "training"
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        input_ids, attention_mask, labels = (value.to(device) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        initial_loss = validate_forward(output, input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)
        output.loss.backward()
        params_without_grad = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not any(torch.isfinite(g).all().item() and torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError("No finite, nonzero trainable gradient was produced")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"]))
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        optimizer.step()
        deltas = [
            (old - new.detach()).abs().max().item()
            for old, new in zip(before, (p for p in model.parameters() if p.requires_grad))
        ]
        if not any(delta > 0 for delta in deltas):
            raise RuntimeError("Optimizer step did not change any trainable parameter")

        stage = "save_checkpoint"
        output_dir = output_dir if output_dir.is_absolute() else config_root / output_dir
        (output_dir / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
        save_model(model, tokenizer, output_dir, {
            "base_repo_id": config.get("model_repo_id"),
            "base_revision": config.get("model_revision"),
            "model_transform": config.get("model_transform", "none"),
            "transform_config": config.get("transform_config", {}),
            "seed": config["seed"],
        })

        stage = "reload_checkpoint"
        reloaded_model, reload_info = load_model(
            output_dir / "model", dtype=dtype, device=device, output_loading_info=True,
        )
        reload_missing = set(reload_info.get("missing_keys", [])) - tied_allowlist
        reload_unexpected = set(reload_info.get("unexpected_keys", [])) - tied_allowlist
        if reload_missing or reload_unexpected or reload_info.get("mismatched_keys"):
            raise RuntimeError(f"Incompatible reloaded checkpoint keys: {reload_info}")
        reloaded_model.eval()
        with torch.no_grad():
            memory_output = model.eval()(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            reload_output = reloaded_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        reload_loss = validate_forward(reload_output, input_ids.shape[0], input_ids.shape[1],
                                       reloaded_model.config.vocab_size)

        reload_logit_max_diff = float((memory_output.logits - reload_output.logits).abs().max().cpu())
        if reload_logit_max_diff > config.get("reload_logit_tolerance", 1e-5):
            raise RuntimeError(f"Reload logits differ by {reload_logit_max_diff}")

        stage = "generation"
        prompt = tokenizer(config["generation_prompt"], return_tensors="pt").to(device)
        generated = reloaded_model.generate(**prompt, max_new_tokens=config["max_new_tokens"],
                                            min_new_tokens=config.get("min_new_tokens", 0),
                                            do_sample=config["do_sample"], pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.decode(generated[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True)
        if not generated_text.strip():
            raise RuntimeError("Generation returned empty text")
        report.update(status="passed", initial_loss=initial_loss, reload_loss=reload_loss,
                      gradient_norm=gradient_norm, parameter_changed=True, checkpoint_reload_passed=True,
                      generation_passed=True, generated_text=generated_text, dataset_examples=len(texts),
                      params_without_grad=params_without_grad, max_abs_param_delta=max(deltas),
                      reload_logit_max_diff=reload_logit_max_diff,
                      tied_key_allowlist=sorted(tied_allowlist), native_repo_path_used=False)
    except Exception as exc:
        report.update(failed_stage=stage, exception_type=type(exc).__name__, error=str(exc))
        _write_report(report_path, report)
        raise RuntimeError(f"Smoke test failed during {stage}: {exc}") from exc
    _write_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()
    try:
        report = run(args.config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report": args.config}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
