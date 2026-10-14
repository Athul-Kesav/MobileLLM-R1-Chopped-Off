# Distill MobileLLM-R1-140M into ShiShu

The teacher is the fully trained `models/MobileLLM-R1-140M`. The student starts from `compare/runs/shishu/reasoning`. Both use the same 128,256-token vocabulary.

Training is teacher-forced forward KL on assistant tokens of the local Tulu-3 and OpenMath chats, plus a 0.1 weight on the student's own next-token loss. The teacher is frozen. Sampling from ShiShu is not used: its current outputs are still close to uniform, so on-policy distillation would match the teacher on those samples.

From the repository root:

```bash
bash kd/run.sh kd/configs/default.env
```

The script keeps GPUs that have at least 8 GiB free. The student is written to `kd/runs/shishu`. After that job exits, score it on the same held-out set:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python compare/evaluate.py \
  --env compare/configs/default.env \
  --checkpoint kd/runs/shishu \
  --out kd/results
```

Layerwise KD starts from that student. Blocks 0–14 match the teacher block of the same index with normalized MSE. The teacher block output already includes its attention residual and its MLP residual. Student block 15 stays with the logit loss. The run writes `kd/runs/shishu-layerwise`.

```bash
bash kd/run.sh kd/configs/layerwise.env
```

The transplant run copies the fully trained 140M into a new ShiShu student, then distills with the same layerwise loss. Blocks 0–10 are copied whole. Blocks 11–14 keep the MLP and drop attention. Block 15 starts as a residual identity. The learning rate is 1e-5 so the copied blocks move less than in the from-scratch run. The student init is written to `kd/runs/shishu-transplant`, and training writes `kd/runs/shishu-transplant-kd`.

```bash
bash kd/run.sh kd/configs/transplant.env
```

A short check:

```bash
MAX_STEPS=2 bash kd/run.sh kd/configs/default.env
```
