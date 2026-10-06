#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE="${1:-configs/full_training.env}"
STAGE="${2:-all}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE; run prepare_full_training.sh first." >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"

world_size=$((NNODES * NPROC_PER_NODE))
[[ "$world_size" -eq 128 ]] || { echo "Expected 128 processes, got $world_size." >&2; exit 2; }
[[ -d "$MODEL_ROOT" && -d "$OUTPUT_ROOT" ]] || { echo "Run prepare_full_training.sh first." >&2; exit 2; }
export MASTER_ADDR MASTER_PORT HF_HOME

run_pretrain() {
  local phase="$1" data="$2" output="$3" max_steps="$4"
  [[ -d "$data/1" && -d "$data/$NNODES" ]] ||
    { echo "Pretraining data must contain node directories 1 and $NNODES: $data" >&2; exit 2; }
  torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    pretrain/pretrain.py \
    --input_model_filename "$MODEL_ROOT/${BASE_MODEL_ID##*/}-base" \
    --train_data_local_path "$data" --output_dir "$output" \
    --do_train True --do_eval False --model_max_length 2048 \
    --fp16 False --bf16 True --log_on_each_node False \
    --average_tokens_across_devices False --ddp_find_unused_parameters False \
    --logging_dir "$LOG_ROOT/$phase" --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 \
    --save_steps 1000 --logging_steps 10 --eval_strategy no \
    --save_strategy steps --report_to tensorboard --save_total_limit 1 \
    --learning_rate 4e-3 --weight_decay 0.1 --adam_beta1 0.9 \
    --adam_beta2 0.95 --adam_epsilon 1e-8 --lr_scheduler_type linear \
    --gradient_checkpointing False --max_steps "$max_steps" --warmup_step 2000 \
    2>&1 | tee "$LOG_ROOT/$phase.log"
}

run_sft() {
  local model="$1" output="$2"
  shift 2
  torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    sft/sft.py --model_name_or_path "$model" --model_revision "$FINAL_MODEL_REVISION" \
    --output_dir "$output" \
    --eval_strategy no --logging_steps 1 --report_to tensorboard \
    "$@"
}

case "$STAGE" in
  pretrain-phase1) run_pretrain pretrain_phase1 "$PRETRAIN_PHASE1_DATA" "$OUTPUT_ROOT/pretrain_phase1" 500000 ;;
  pretrain-phase2) run_pretrain pretrain_phase2 "$PRETRAIN_PHASE2_DATA" "$OUTPUT_ROOT/pretrain_phase2" 500000 ;;
  midtrain-phase1|midtrain-phase2)
    echo "Mid-training requires the repository's separate KD implementation and teacher-logit contract; it is not present in upstream MobileLLM-R1." >&2
    exit 3 ;;
  sft-general)
    run_sft "$OUTPUT_ROOT/pretrain_phase2" "$OUTPUT_ROOT/sft_general" \
      --dataset_name "$GENERAL_SFT_DATASET" --max_length 4096 --learning_rate 5e-6 \
      --lr_scheduler_type linear --warmup_ratio 0.03 --weight_decay 0.0 \
      --num_train_epochs 2 --per_device_train_batch_size 4 --gradient_checkpointing \
      --eos_token '<|eot_id|>' --saving_steps 1000 --use_liger_kernel True ;;
  sft-reasoning)
    run_sft "$OUTPUT_ROOT/sft_general" "$OUTPUT_ROOT/sft_reasoning" \
      --dataset_name "$REASONING_SFT_DATASET_ID" --dataset_config "$REASONING_SFT_DATASET_CONFIG" \
      --max_length 32768 --learning_rate 8e-5 --warmup_ratio 0.1 \
      --lr_scheduler_type cosine --num_train_epochs 4 --per_device_train_batch_size 8 \
      --gradient_checkpointing --eos_token '<|eot_id|>' --saving_steps 1000 \
      --use_liger_kernel True ;;
  eval)
    echo "Use evaluation/run_eval.sh after downloading evaluation data and installing vLLM." ;;
  all)
    echo "Run individual stages explicitly; full training is not chained automatically." >&2
    exit 2 ;;
  *) echo "Usage: $0 [env-file] {pretrain-phase1|pretrain-phase2|midtrain-phase1|midtrain-phase2|sft-general|sft-reasoning|eval|all}" >&2; exit 2 ;;
esac
