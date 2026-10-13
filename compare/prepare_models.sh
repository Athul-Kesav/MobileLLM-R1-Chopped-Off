#!/usr/bin/env bash
# Assemble the ShiShu checkpoint and fetch facebook/MobileLLM-R1-140M-base.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source compare/env.sh
load_compare_env "${1:-compare/configs/default.env}"

bash scripts/assemble_model.sh "$SHISHU_DIR"
[[ -f "$SHISHU_DIR/config.json" && -f "$SHISHU_DIR/model.safetensors" ]] || {
  echo "ShiShu checkpoint is incomplete: $SHISHU_DIR" >&2
  exit 2
}

if [[ ! -f "$BASE_DIR/config.json" || ! -f "$BASE_DIR/model.safetensors" ]]; then
  echo "Downloading $BASE_REPO -> $BASE_DIR"
  "$PYTHON" scripts/download_model.py \
    --repo_id "$BASE_REPO" \
    --local_dir "$BASE_DIR" \
    --report_file compare/logs/base_download.json
fi

if [[ ! -f "$BASE_DIR/chat_template.jinja" ]]; then
  [[ -f "$CHAT_TEMPLATE" ]] || { echo "Missing chat template $CHAT_TEMPLATE" >&2; exit 2; }
  cp "$CHAT_TEMPLATE" "$BASE_DIR/chat_template.jinja"
  echo "Copied chat template onto $BASE_DIR"
fi

echo "Models ready:"
echo "  shishu $SHISHU_DIR"
echo "  base   $BASE_DIR"
