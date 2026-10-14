#!/usr/bin/env bash
# Distill models/MobileLLM-R1-140M into the ShiShu checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source compare/env.sh
load_compare_env "${1:-kd/configs/default.env}"

if [[ -n "${TRANSPLANT_FROM:-}" && ! -f "$STUDENT/model.safetensors" ]]; then
  echo "Transplanting $TRANSPLANT_FROM -> $STUDENT"
  "$PYTHON" kd/transplant.py --teacher "$TRANSPLANT_FROM" --dest "$STUDENT"
fi

[[ -f "$STUDENT/model.safetensors" && -f "$TEACHER/model.safetensors" ]] || {
  echo "Missing student or teacher weights." >&2
  echo "Student: $STUDENT" >&2
  echo "Teacher: $TEACHER" >&2
  exit 2
}
[[ -f "$DATA_GENERAL" && -f "$DATA_REASONING" ]] || {
  echo "Missing SFT jsonl. Run: bash compare/run.sh prepare" >&2
  exit 2
}
[[ "${NPROC_PER_NODE}" -ge 1 ]] || { echo "KD needs a GPU. Set NPROC_PER_NODE." >&2; exit 2; }

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export MASTER_ADDR MASTER_PORT TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${SEED:-140}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "KD $STUDENT <- $TEACHER on GPUs ${CUDA_VISIBLE_DEVICES:-all} (nproc=$NPROC_PER_NODE)"
"$PYTHON" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --node_rank=0 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  kd/train.py \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --train-files "$DATA_GENERAL" "$DATA_REASONING" \
  --output-dir "$OUTPUT_DIR" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LR" \
  --warmup-ratio "$WARMUP" \
  --temperature "$TEMPERATURE" \
  --kl-coef "$KL_COEF" \
  --ce-coef "$CE_COEF" \
  --layer-coef "${LAYER_COEF:-0}" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --seed "$SEED" \
  2>&1 | tee -a "$LOG_DIR/train.log"
echo "Student saved under $OUTPUT_DIR"
