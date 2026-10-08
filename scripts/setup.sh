#!/usr/bin/env bash
# Create ./.venv and install training deps. Run from anywhere; cds to repo root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv}"

bash scripts/assemble_model.sh
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install -r requirements-full.txt
echo
echo "Environment ready: $PWD/$VENV"
echo "Launch with:"
echo "  $VENV/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"
echo "  PYTHON=$VENV/bin/python bash scripts/prepare_full_training.sh configs/train.env"
echo "  PYTHON=$VENV/bin/python bash scripts/run_full_training.sh configs/train.env all"
echo "On an A100 node use configs/a100.env instead of configs/train.env."
