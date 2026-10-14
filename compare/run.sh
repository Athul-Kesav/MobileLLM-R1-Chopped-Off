#!/usr/bin/env bash
# prepare | finetune | eval | all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CMD="${1:-all}"
ENV_FILE="${2:-compare/configs/default.env}"
# shellcheck disable=SC1091
source compare/env.sh
load_compare_env "$ENV_FILE"
export PYTHONHASHSEED="${SEED:-140}"

prepare() {
  mkdir -p "$LOG_ROOT"
  bash compare/prepare_models.sh "$ENV_FILE"
  "$PYTHON" compare/prepare_data.py --env "$ENV_FILE"
}

evaluate() {
  local stamp out
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out="$RESULT_ROOT/$stamp"
  "$PYTHON" compare/evaluate.py --env "$ENV_FILE" --out "$out"
  ln -sfn "$stamp" "$RESULT_ROOT/latest"
  echo "Report: $out/report.md"
}

case "$CMD" in
  prepare) prepare ;;
  finetune) bash compare/finetune.sh "$ENV_FILE" ;;
  eval) evaluate ;;
  all) prepare; bash compare/finetune.sh "$ENV_FILE"; evaluate ;;
  *)
    echo "Usage: bash compare/run.sh {prepare|finetune|eval|all} [env-file]" >&2
    exit 2
    ;;
esac
