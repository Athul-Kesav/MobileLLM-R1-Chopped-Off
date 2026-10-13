#!/usr/bin/env bash
# Run the same two-stage SFT on ShiShu and MobileLLM-R1-140M-base.
# Stage 1 starts from each original checkpoint. Stage 2 starts from stage 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source compare/env.sh
load_compare_env "${1:-compare/configs/default.env}"

[[ -f "$COMPARE_DATA/general/train.jsonl" && -f "$COMPARE_DATA/reasoning/train.jsonl" ]] || {
  echo "Missing comparison splits. Run: bash compare/run.sh prepare" >&2
  exit 2
}
[[ -f "$SHISHU_DIR/model.safetensors" && -f "$BASE_DIR/model.safetensors" ]] || {
  echo "Missing model weights. Run: bash compare/run.sh prepare" >&2
  exit 2
}
[[ "${NPROC_PER_NODE}" -ge 1 ]] || { echo "Finetuning needs a GPU. Set NPROC_PER_NODE." >&2; exit 2; }
"$PYTHON" - <<'PY' || { echo "Finetuning needs a visible CUDA device." >&2; exit 2; }
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export MASTER_ADDR MASTER_PORT PYTHONPATH="$PWD:$PWD/pretrain${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TORCHRUN=("$PYTHON" -m torch.distributed.run
  --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --node_rank=0
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")

run_sft() {
  local name="$1" stage="$2" model="$3" data="$4" seq="$5" lr="$6" warmup="$7"
  local epochs="$8" steps="$9" batch="${10}" accum="${11}" scheduler="${12}"
  local output="$RUN_ROOT/$name/$stage"
  local logfile="$LOG_ROOT/${name}_${stage}.log"
  mkdir -p "$LOG_ROOT/$name/$stage"
  echo "== $name $stage: $model -> $output"
  "${TORCHRUN[@]}" sft/sft.py \
    --model_name_or_path "$model" \
    --dataset_name "$data" \
    --output_dir "$output" \
    --max_length "$seq" \
    --learning_rate "$lr" \
    --lr_scheduler_type "$scheduler" \
    --warmup_ratio "$warmup" \
    --weight_decay 0.0 \
    --num_train_epochs "$epochs" \
    --max_steps "$steps" \
    --per_device_train_batch_size "$batch" \
    --gradient_accumulation_steps "$accum" \
    --gradient_checkpointing \
    --bf16 True \
    --dtype bfloat16 \
    --eos_token '<|eot_id|>' \
    --use_liger_kernel "$USE_LIGER_KERNEL" \
    --eval_strategy no \
    --logging_steps 10 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 2 \
    --logging_dir "$LOG_ROOT/$name/$stage" \
    --report_to none \
    --resume_from_checkpoint auto \
    2>&1 | tee -a "$logfile"
}

for name in shishu base; do
  if [[ "$name" == shishu ]]; then
    init="$SHISHU_DIR"
  else
    init="$BASE_DIR"
  fi
  run_sft "$name" general "$init" "$COMPARE_DATA/general" \
    "$SEQ_LEN" "$GENERAL_LR" "$GENERAL_WARMUP" "$GENERAL_EPOCHS" "$GENERAL_MAX_STEPS" \
    "$GENERAL_BATCH" "$GENERAL_ACCUM" "$GENERAL_SCHEDULER"
  run_sft "$name" reasoning "$RUN_ROOT/$name/general" "$COMPARE_DATA/reasoning" \
    "$SEQ_LEN" "$REASONING_LR" "$REASONING_WARMUP" "$REASONING_EPOCHS" "$REASONING_MAX_STEPS" \
    "$REASONING_BATCH" "$REASONING_ACCUM" "$REASONING_SCHEDULER"
done

echo "Finetune finished under $RUN_ROOT"
