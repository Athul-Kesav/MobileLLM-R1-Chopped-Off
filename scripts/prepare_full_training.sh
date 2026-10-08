#!/usr/bin/env bash
# Validate the env file, checkpoint, Python packages, and data shards.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${1:-configs/train.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Use configs/train.env or configs/a100.env." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1091
source scripts/env_lib.sh
resolve_training_paths

: "${MODEL_SIZE:=140m}"

if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ && "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES and NPROC_PER_NODE must be positive integers (got ${NNODES}x${NPROC_PER_NODE})." >&2
  exit 2
fi

if [[ "${USE_LOCAL_SHISHU_MODEL:-0}" == "1" ]]; then
  : "${SHISHU_MODEL_DIR:?SHISHU_MODEL_DIR is required}"
  bash scripts/assemble_model.sh "$SHISHU_MODEL_DIR"
  [[ -f "$SHISHU_MODEL_DIR/config.json" ]] || {
    echo "Missing $SHISHU_MODEL_DIR/config.json. Run scripts/stage_shishu_checkpoint.py first." >&2
    exit 2
  }
  if [[ ! -f "$SHISHU_MODEL_DIR/model.safetensors" && ! -f "$SHISHU_MODEL_DIR/pytorch_model.bin" ]]; then
    echo "Weights are not in $SHISHU_MODEL_DIR yet. Copy model.safetensors there, then rerun this script." >&2
    exit 2
  fi
  "$PYTHON" scripts/stage_shishu_checkpoint.py --dest "$SHISHU_MODEL_DIR" --config "$SHISHU_MODEL_DIR/config.json"
fi

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$HF_HOME"
world_size=$((NNODES * NPROC_PER_NODE))
cat > "$OUTPUT_ROOT/SETUP_MANIFEST.txt" <<EOF
ShiShu MobileLLM-R1 training setup
model_size=$MODEL_SIZE
python=$PYTHON
shishu_model=${SHISHU_MODEL_DIR:-}
world_size=$world_size
nodes=${NNODES}x${NPROC_PER_NODE}
data_root=$DATA_ROOT
output_root=$OUTPUT_ROOT
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

command -v "$PYTHON" >/dev/null || { echo "Missing python: $PYTHON. Run bash scripts/setup.sh" >&2; exit 1; }
"$PYTHON" - <<'PY'
import importlib.util
required = ("torch", "transformers", "datasets", "trl", "accelerate", "tensorboard", "safetensors")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing packages: " + ", ".join(missing) +
        ". Run bash scripts/setup.sh (creates ./.venv) or pip install -r requirements-full.txt."
    )
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
PY

check_lm_data() {
  local label="$1" root="$2"
  [[ -n "$root" && -d "$root" ]] || { echo "WARN: $label data missing: $root" >&2; return 0; }
  local node
  for ((node = 1; node <= NNODES; node++)); do
    if [[ ! -d "$root/$node" ]]; then
      echo "WARN: $label missing shard $root/$node (need 1..$NNODES)." >&2
      return 0
    fi
  done
  echo "OK $label: $root"
}

check_lm_data pretrain-phase1 "${PRETRAIN_PHASE1_DATA:-}"
check_lm_data pretrain-phase2 "${PRETRAIN_PHASE2_DATA:-}"
check_lm_data midtrain-phase1 "${MIDTRAIN_PHASE1_DATA:-}"
check_lm_data midtrain-phase2 "${MIDTRAIN_PHASE2_DATA:-}"

echo "Setup complete. Manifest: $OUTPUT_ROOT/SETUP_MANIFEST.txt"
echo "Launch: bash scripts/run_full_training.sh $ENV_FILE all"
