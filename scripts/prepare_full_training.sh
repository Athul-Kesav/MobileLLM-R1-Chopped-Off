#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${1:-configs/full_training.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy configs/full_training.env.example and fill all values." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MODEL_SIZE:?MODEL_SIZE is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${BASE_MODEL_REVISION:?BASE_MODEL_REVISION is required}"
: "${FINAL_MODEL_REVISION:?FINAL_MODEL_REVISION is required}"

for value in "$BASE_MODEL_REVISION" "$FINAL_MODEL_REVISION"; do
  if [[ "$value" == REPLACE_* || ! "$value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Model revisions must be 40-character resolved Hub SHAs: $value" >&2
    exit 2
  fi
done
if [[ "${NNODES}" != "16" || "${NPROC_PER_NODE}" != "8" ]]; then
  echo "This setup targets the paper recipe: NNODES=16 and NPROC_PER_NODE=8." >&2
  exit 2
fi

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$HF_HOME"
cat > "$OUTPUT_ROOT/SETUP_MANIFEST.txt" <<EOF
MobileLLM-R1 full-training setup
model_size=$MODEL_SIZE
base_model=$BASE_MODEL_ID@$BASE_MODEL_REVISION
final_model=$FINAL_MODEL_ID@$FINAL_MODEL_REVISION
teacher=$TEACHER_MODEL_ID
world_size=$((NNODES * NPROC_PER_NODE))
data_root=$DATA_ROOT
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

for command in torchrun python; do
  command -v "$command" >/dev/null || { echo "Missing executable: $command" >&2; exit 1; }
done
python - <<'PY'
import importlib.util
required = ("torch", "transformers", "datasets", "trl", "accelerate", "tensorboard")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing) +
                     ". Install requirements-full.txt in the approved environment.")
PY

echo "Setup directories and manifest created under $OUTPUT_ROOT."
echo "No datasets or models were downloaded. Run scripts/download_full_models.sh after reviewing the manifest."
