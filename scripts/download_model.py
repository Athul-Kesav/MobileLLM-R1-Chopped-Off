"""Download and verify a complete Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def verify_model_directory(local_dir: str | Path) -> dict[str, Any]:
    """Return verification details, raising ValueError for an incomplete snapshot."""
    root = Path(local_dir)
    if not root.is_dir():
        raise ValueError(f"Model directory does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    names = {path.name for path in files}
    config_found = (root / "config.json").is_file()
    tokenizer_found = bool(
        names & {"tokenizer.json", "tokenizer.model", "vocab.json"}
    ) or ("vocab.txt" in names and "tokenizer_config.json" in names)
    weight_files = [
        path for path in files if path.suffix.lower() in WEIGHT_SUFFIXES and path.stat().st_size > 0
    ]
    if not config_found:
        raise ValueError(f"Missing config.json in {root}")
    if not tokenizer_found:
        raise ValueError(f"Missing tokenizer files in {root}")
    index_files = list(root.glob("*.index.json"))
    indexed_shards: list[str] = []
    for index_file in index_files:
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
            indexed_shards.extend(index.get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"Invalid weight index {index_file}: {exc}") from exc
    missing_shards = sorted(
        {
            shard for shard in indexed_shards
            if not (root / shard).is_file() or (root / shard).stat().st_size == 0
        }
    )
    if missing_shards:
        raise ValueError(f"Missing or empty indexed weight shards: {missing_shards}")
    if not weight_files:
        raise ValueError(f"Missing non-empty model weight files in {root}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return {
        "files_found": sorted(str(path.relative_to(root)) for path in files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "config_found": True,
        "tokenizer_found": True,
        "weights_found": True,
        "weight_files": sorted(str(path.relative_to(root)) for path in weight_files),
        "indexed_shards": sorted(set(indexed_shards)),
        "architectures": config.get("architectures", []),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
    }


def download_model(
    repo_id: str,
    local_dir: str | Path,
    revision: str | None = None,
    token: str | None = None,
    report_file: str | Path = "reports/download_report.json",
) -> dict[str, Any]:
    """Download a resumable snapshot and write a verification report."""
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "failed",
        "repo_id": repo_id,
        "requested_revision": revision,
        "local_dir": str(Path(local_dir).resolve()),
        "failed_stage": "download",
    }
    try:
        from huggingface_hub import HfApi, snapshot_download

        target = Path(local_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            token=token,
            resume_download=True,
        )
        details = verify_model_directory(target)
        resolved_revision = HfApi(token=token).model_info(repo_id, revision=revision).sha
        report.update(details, resolved_revision=resolved_revision, status="passed", failed_stage=None)
    except Exception as exc:
        report.update(exception_type=type(exc).__name__, error=str(exc))
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Model download/verification failed for {repo_id}: {exc}. "
            "If authentication is required, use `hf auth login` or HF_TOKEN."
        ) from exc
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="facebook/MobileLLM-R1-140M")
    parser.add_argument("--local_dir", default="models/MobileLLM-R1-140M")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="HF token; prefer HF_TOKEN or `hf auth login`.")
    parser.add_argument("--report_file", default="reports/download_report.json")
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    try:
        report = download_model(args.repo_id, args.local_dir, args.revision, token, args.report_file)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Downloaded and verified model at {report['local_dir']}")
    print(f"Resolved revision: {report['resolved_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
