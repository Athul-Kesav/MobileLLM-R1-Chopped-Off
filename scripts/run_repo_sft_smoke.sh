#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
MODEL="${MODEL_PATH:-models/MobileLLM-R1-140M}"
DATA="${DATA_FILE:-data/smoke_train.jsonl}"
OUTPUT="${OUTPUT_DIR:-checkpoints/repo_sft_smoke}"
LOG="${LOG_FILE:-logs/repo_sft_smoke.log}"
mkdir -p "$(dirname "$LOG")" "$OUTPUT"
export HF_HUB_OFFLINE=1

# ASSUMPTION: TRL accepts a local JSONL through --datasets and one torchrun rank
# is sufficient to exercise the repository's native SFT entry point.
exec torchrun --standalone --nproc_per_node=1 sft/sft.py \
  --model_name_or_path "$MODEL" \
  --datasets "$DATA" \
  --dataset_train_split train \
  --max_steps 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_length 128 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 2 \
  --logging_steps 1 \
  --output_dir "$OUTPUT" \
  --report_to none 2>&1 | tee "$LOG"
