"""Write non-secret runtime and storage information for the smoke reports."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    import torch
    import transformers
    from huggingface_hub import __version__ as hub_version

    gpu = []
    cuda_probe_error = None
    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu.append({"name": props.name, "vram_bytes": props.total_memory})
    except RuntimeError as exc:
        cuda_probe_error = f"{type(exc).__name__}: {exc}"
    root = Path(__file__).resolve().parents[1]
    usage = shutil.disk_usage(root)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver = []
    report = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu,
        "cuda_probe_error": cuda_probe_error,
        "driver_versions": driver,
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": hub_version,
        "trl_version": _version("trl"),
        "vllm_version": _version("vllm"),
        "lm_eval_version": _version("lm_eval"),
        "disk": {"path": str(root), "free_bytes": usage.free, "total_bytes": usage.total},
    }
    output = root / "reports/env_report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _version(module: str) -> str | None:
    try:
        imported = __import__(module)
    except ImportError:
        return None
    return getattr(imported, "__version__", "installed")


if __name__ == "__main__":
    raise SystemExit(main())
