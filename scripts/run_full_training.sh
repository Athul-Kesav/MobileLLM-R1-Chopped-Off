#!/usr/bin/env bash
# Launch MobileLLM-R1 training stages on any node/GPU layout.
#
# Paths are repo-relative (configs/train.env). GPU count is auto unless set.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${1:-configs/train.env}"
STAGE="${2:-}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE; use configs/train.env or configs/a100.env." >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1091
source scripts/env_lib.sh
resolve_training_paths

usage() {
  echo "Usage: $0 <env-file> {pretrain-phase1|pretrain-phase2|midtrain-phase1|midtrain-phase2|sft-general|sft-reasoning|all}" >&2
  exit 2
}
[[ -n "$STAGE" ]] || usage

: "${NNODES:=1}" "${NPROC_PER_NODE:=8}" "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}" "${MASTER_PORT:=29500}"
: "${OUTPUT_ROOT:?}" "${LOG_ROOT:?}"
: "${PYTHON:=python}"
: "${PRETRAIN_INIT:=checkpoint}"

: "${PRETRAIN_SEQ_LEN:=2048}" "${PRETRAIN_BATCH:=4}" "${PRETRAIN_GRAD_ACCUM:=4}"
: "${PRETRAIN_GRAD_CKPT:=True}"
: "${PRETRAIN_LR:=4e-3}" "${PRETRAIN_WARMUP:=2000}" "${PRETRAIN_MIN_LR_RATIO:=0.1}"
: "${PRETRAIN_PHASE1_STEPS:=500000}" "${PRETRAIN_PHASE2_STEPS:=500000}"
: "${PRETRAIN_SAVE_STEPS:=1000}" "${PRETRAIN_BUFFER_SIZE:=2048}"

: "${MIDTRAIN_SEQ_LEN:=4096}" "${MIDTRAIN_BATCH:=4}" "${MIDTRAIN_GRAD_ACCUM:=1}"
: "${MIDTRAIN_LR:=3.6e-4}" "${MIDTRAIN_WARMUP:=0}"
: "${MIDTRAIN_PHASE1_STEPS:=50000}" "${MIDTRAIN_PHASE2_STEPS:=50000}"
: "${MIDTRAIN_SAVE_STEPS:=1000}" "${MIDTRAIN_BUFFER_SIZE:=2048}"

: "${SFT_GENERAL_SEQ_LEN:=4096}" "${SFT_GENERAL_BATCH:=4}" "${SFT_GENERAL_GRAD_ACCUM:=1}"
: "${SFT_GENERAL_LR:=5e-6}" "${SFT_GENERAL_EPOCHS:=2}" "${SFT_GENERAL_MAX_STEPS:=-1}"
: "${SFT_REASONING_SEQ_LEN:=32768}" "${SFT_REASONING_BATCH:=8}" "${SFT_REASONING_GRAD_ACCUM:=1}"
: "${SFT_REASONING_LR:=8e-5}" "${SFT_REASONING_EPOCHS:=4}" "${SFT_REASONING_MAX_STEPS:=-1}"
: "${SFT_SAVE_STEPS:=1000}" "${USE_LIGER_KERNEL:=True}"

if [[ "${USE_LOCAL_SHISHU_MODEL:-0}" == "1" ]]; then
  INPUT_MODEL="$SHISHU_MODEL_DIR"
  [[ -f "$INPUT_MODEL/model.safetensors" || -f "$INPUT_MODEL/pytorch_model.bin" ]] || {
    echo "Copy model.safetensors into $INPUT_MODEL before training." >&2
    exit 2
  }
else
  INPUT_MODEL="$MODEL_ROOT/${BASE_MODEL_ID##*/}-base"
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export MASTER_ADDR MASTER_PORT
[[ -n "${HF_HOME:-}" ]] && export HF_HOME
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd):$(pwd)/pretrain${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TORCHRUN=("$PYTHON" -m torch.distributed.run
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")

require_model() {
  [[ -f "$1/config.json" ]] || { echo "Stage input $1 has no config.json; run the previous stage first." >&2; exit 2; }
}

require_shards() {
  local data="$1" node
  for ((node = 1; node <= NNODES; node++)); do
    [[ -d "$data/$node" ]] || { echo "Missing data shard $data/$node (need 1..$NNODES)." >&2; exit 2; }
  done
}

# args: phase data input output steps seq batch accum lr warmup min_ratio save buffer
run_lm_stage() {
  local phase="$1" data="$2" input="$3" output="$4" steps="$5" seq="$6" batch="$7"
  local accum="$8" lr="$9" warmup="${10}" min_ratio="${11}" save="${12}" buffer="${13}"
  require_shards "$data"
  require_model "$input"
  echo "== $phase: $input -> $output ($steps steps, ${NNODES}x${NPROC_PER_NODE} GPUs)"
  "${TORCHRUN[@]}" pretrain/pretrain.py \
    --input_model_filename "$input" --init_from "$([[ $phase == pretrain_phase1 ]] && echo "$PRETRAIN_INIT" || echo checkpoint)" \
    --train_data_local_path "$data" --output_dir "$output" \
    --do_train True --do_eval False --model_max_length "$seq" --buffer_size "$buffer" \
    --fp16 False --bf16 True --log_on_each_node False \
    --average_tokens_across_devices False --ddp_find_unused_parameters False \
    --logging_dir "$LOG_ROOT/$phase" --per_device_train_batch_size "$batch" \
    --gradient_accumulation_steps "$accum" \
    --save_steps "$save" --logging_steps 10 --eval_strategy no \
    --save_strategy steps --report_to tensorboard --save_total_limit 2 \
    --learning_rate "$lr" --weight_decay 0.1 --adam_beta1 0.9 \
    --adam_beta2 0.95 --adam_epsilon 1e-8 --lr_scheduler_type linear --min_lr_ratio "$min_ratio" \
    --gradient_checkpointing "$PRETRAIN_GRAD_CKPT" --max_steps "$steps" --warmup_steps "$warmup" \
    --dataloader_num_workers 0 --resume_from_checkpoint auto --ignore_data_skip True \
    2>&1 | tee -a "$LOG_ROOT/$phase.log"
}

run_sft() {
  local phase="$1" input="$2" output="$3"
  shift 3
  require_model "$input"
  echo "== $phase: $input -> $output (${NNODES}x${NPROC_PER_NODE} GPUs)"
  "${TORCHRUN[@]}" sft/sft.py --model_name_or_path "$input" \
    --output_dir "$output" --bf16 True --dtype bfloat16 --resume_from_checkpoint auto \
    --eval_strategy no --logging_steps 10 --report_to tensorboard \
    --logging_dir "$LOG_ROOT/$phase" --save_steps "$SFT_SAVE_STEPS" --save_total_limit 2 \
    --gradient_checkpointing --eos_token '<|eot_id|>' --use_liger_kernel "$USE_LIGER_KERNEL" \
    "$@" 2>&1 | tee -a "$LOG_ROOT/$phase.log"
}

stage_pretrain_phase1() {
  run_lm_stage pretrain_phase1 "$PRETRAIN_PHASE1_DATA" "$INPUT_MODEL" "$OUTPUT_ROOT/pretrain_phase1" \
    "$PRETRAIN_PHASE1_STEPS" "$PRETRAIN_SEQ_LEN" "$PRETRAIN_BATCH" "$PRETRAIN_GRAD_ACCUM" \
    "$PRETRAIN_LR" "$PRETRAIN_WARMUP" "$PRETRAIN_MIN_LR_RATIO" "$PRETRAIN_SAVE_STEPS" "$PRETRAIN_BUFFER_SIZE"
}
stage_pretrain_phase2() {
  run_lm_stage pretrain_phase2 "$PRETRAIN_PHASE2_DATA" "$OUTPUT_ROOT/pretrain_phase1" "$OUTPUT_ROOT/pretrain_phase2" \
    "$PRETRAIN_PHASE2_STEPS" "$PRETRAIN_SEQ_LEN" "$PRETRAIN_BATCH" "$PRETRAIN_GRAD_ACCUM" \
    "$PRETRAIN_LR" "$PRETRAIN_WARMUP" "$PRETRAIN_MIN_LR_RATIO" "$PRETRAIN_SAVE_STEPS" "$PRETRAIN_BUFFER_SIZE"
}
stage_midtrain_phase1() {
  run_lm_stage midtrain_phase1 "$MIDTRAIN_PHASE1_DATA" "$OUTPUT_ROOT/pretrain_phase2" "$OUTPUT_ROOT/midtrain_phase1" \
    "$MIDTRAIN_PHASE1_STEPS" "$MIDTRAIN_SEQ_LEN" "$MIDTRAIN_BATCH" "$MIDTRAIN_GRAD_ACCUM" \
    "$MIDTRAIN_LR" "$MIDTRAIN_WARMUP" 0 "$MIDTRAIN_SAVE_STEPS" "$MIDTRAIN_BUFFER_SIZE"
}
stage_midtrain_phase2() {
  run_lm_stage midtrain_phase2 "$MIDTRAIN_PHASE2_DATA" "$OUTPUT_ROOT/midtrain_phase1" "$OUTPUT_ROOT/midtrain_phase2" \
    "$MIDTRAIN_PHASE2_STEPS" "$MIDTRAIN_SEQ_LEN" "$MIDTRAIN_BATCH" "$MIDTRAIN_GRAD_ACCUM" \
    "$MIDTRAIN_LR" "$MIDTRAIN_WARMUP" 0 "$MIDTRAIN_SAVE_STEPS" "$MIDTRAIN_BUFFER_SIZE"
}
stage_sft_general() {
  run_sft sft_general "$OUTPUT_ROOT/midtrain_phase2" "$OUTPUT_ROOT/sft_general" \
    --dataset_name "$GENERAL_SFT_DATASET" --max_length "$SFT_GENERAL_SEQ_LEN" \
    --learning_rate "$SFT_GENERAL_LR" --lr_scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.0 \
    --num_train_epochs "$SFT_GENERAL_EPOCHS" --max_steps "$SFT_GENERAL_MAX_STEPS" \
    --per_device_train_batch_size "$SFT_GENERAL_BATCH" --gradient_accumulation_steps "$SFT_GENERAL_GRAD_ACCUM"
}
stage_sft_reasoning() {
  local config_args=()
  [[ -n "${REASONING_SFT_DATASET_CONFIG:-}" ]] && config_args=(--dataset_config "$REASONING_SFT_DATASET_CONFIG")
  run_sft sft_reasoning "$OUTPUT_ROOT/sft_general" "$OUTPUT_ROOT/sft_reasoning" \
    --dataset_name "$REASONING_SFT_DATASET" "${config_args[@]}" --max_length "$SFT_REASONING_SEQ_LEN" \
    --learning_rate "$SFT_REASONING_LR" --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.0 \
    --num_train_epochs "$SFT_REASONING_EPOCHS" --max_steps "$SFT_REASONING_MAX_STEPS" \
    --per_device_train_batch_size "$SFT_REASONING_BATCH" --gradient_accumulation_steps "$SFT_REASONING_GRAD_ACCUM"
}

case "$STAGE" in
  pretrain-phase1) stage_pretrain_phase1 ;;
  pretrain-phase2) stage_pretrain_phase2 ;;
  midtrain-phase1) stage_midtrain_phase1 ;;
  midtrain-phase2) stage_midtrain_phase2 ;;
  sft-general) stage_sft_general ;;
  sft-reasoning) stage_sft_reasoning ;;
  all)
    stage_pretrain_phase1
    stage_pretrain_phase2
    stage_midtrain_phase1
    stage_midtrain_phase2
    stage_sft_general
    stage_sft_reasoning ;;
  *) usage ;;
esac
