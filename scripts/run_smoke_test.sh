#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
if [[ -x "../venv/bin/python" ]]; then
  PYTHON="../venv/bin/python"
fi

skip_download=false
if [[ "${1:-}" == "--skip-download" ]]; then
  skip_download=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--skip-download]" >&2
  exit 2
fi

"$PYTHON" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "huggingface_hub", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + ". Install requirements-smoke.txt.")
PY

model_dir="models/MobileLLM-R1-140M"
if [[ "$skip_download" == false ]]; then
  "$PYTHON" scripts/download_model.py --repo_id facebook/MobileLLM-R1-140M --local_dir "$model_dir"
else
  "$PYTHON" - <<'PY'
from scripts.download_model import verify_model_directory
try:
    verify_model_directory("models/MobileLLM-R1-140M")
except Exception as exc:
    raise SystemExit("Cannot use --skip-download: local model is missing or invalid: " + str(exc))
PY
fi

"$PYTHON" -m src.smoke_test --config configs/smoke_test.yaml
echo "Download report: reports/download_report.json"
echo "Smoke-test report: reports/smoke_test_report.json"
echo "Checkpoint: checkpoints/smoke_test"
