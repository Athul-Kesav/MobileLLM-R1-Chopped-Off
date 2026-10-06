#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${1:-configs/full_training.env}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${BASE_MODEL_ID:?}"
: "${BASE_MODEL_REVISION:?}"
: "${FINAL_MODEL_ID:?}"
: "${FINAL_MODEL_REVISION:?}"
: "${MODEL_ROOT:?}"
python scripts/download_model.py --repo_id "$BASE_MODEL_ID" \
  --revision "$BASE_MODEL_REVISION" --local_dir "$MODEL_ROOT/${BASE_MODEL_ID##*/}-base" \
  --report_file "$MODEL_ROOT/download_base_report.json"
python scripts/download_model.py --repo_id "$FINAL_MODEL_ID" \
  --revision "$FINAL_MODEL_REVISION" --local_dir "$MODEL_ROOT/${FINAL_MODEL_ID##*/}" \
  --report_file "$MODEL_ROOT/download_final_report.json"
echo "Models downloaded and verified outside the repository at $MODEL_ROOT."
