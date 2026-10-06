# Baseline evaluation command

The full baseline uses the repository harness (not lm-evaluation-harness):

```bash
cd evaluation
CUDA_VISIBLE_DEVICES=0 python3 -u math_eval.py \
  --model_name_or_path "$HOME/models/MobileLLM-R1-140M" \
  --data_name gsm8k,math,aime24 --data_dir ./data --split test \
  --prompt_type raw --max_tokens_per_call 30000 --seed 0 \
  --temperature 0.6 --n_sampling 1 --top_p 0.95 --use_vllm \
  --save_outputs --apply_chat_template
```

This full command is recorded only and was not run as part of the smoke
verification. A 5-example offline slice requires the evaluation data and vLLM,
which are not present in the checked-in smoke fixture.
