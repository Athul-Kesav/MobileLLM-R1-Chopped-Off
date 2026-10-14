# How these results were produced

This is the local comparison between a ShiShu MobileLLM-R1-140M and `facebook/MobileLLM-R1-140M-base`, followed by distillation from the fully trained `facebook/MobileLLM-R1-140M`. The model card and the upstream training recipe stay in the repository `README.md`. The measurements and the reading of them are in `experiment/RESULTS.md`.

Commands are from the repository root. The reported runs used GPUs 0 and 2. GPU 1 was already taken, and the configs now pin those two devices so a later run does not silently pick up a third GPU and change the batch.

## Models

ShiShu is `models/MobileLLM-R1-140M-shishu`: 11 decoder blocks, then 5 MLP-only blocks, hidden size 576, SwiGLU width 2,048, 140,246,784 parameters, vocabulary 128,256. Its initial weights are uncorrelated with the 140M (cosine about 0 on the embeddings and on layer 0). The base model is `models/MobileLLM-R1-140M-base`: 15 decoder blocks, 140,248,512 parameters. The distillation teacher is `models/MobileLLM-R1-140M`, the fully trained release, with the same widths as the base.

## 1. Shared data

`compare/configs/default.env` points at the local Tulu-3 slice and the local OpenMathReasoning slice. `SEED=140` holds out 128 conversations from each and draws 64 GSM8K test questions. Reasoning uses seed 141 so the two holdouts are not the same shuffle.

```bash
bash compare/run.sh prepare compare/configs/default.env
```

That writes `compare/data/general/{train,eval}.jsonl`, `compare/data/reasoning/{train,eval}.jsonl`, `compare/data/gsm8k/test.jsonl`, and `compare/data/manifest.json`. Each training file has 1,372 rows.

## 2. Two-stage supervised finetuning

Both models get the same loop. Stage 1 is general SFT from the original checkpoint: learning rate 5e-6, linear schedule, warmup 0.03, 2 epochs, batch 2. Stage 2 is reasoning SFT from that checkpoint: learning rate 8e-5, cosine schedule, warmup 0.1, 4 epochs, batch 1 with gradient accumulation 2. Sequence length is 2,048. Weight decay is 0. The shuffle seed is `SFT_SEED=42`, which is the Trainer default that the reported runs actually used. `PYTHONHASHSEED` is exported to the same value.

```bash
bash compare/run.sh finetune compare/configs/default.env
```

Checkpoints land in `compare/runs/{shishu,base}/{general,reasoning}`. A second launch resumes a finished stage and returns immediately. Delete those directories before a clean replay.

## 3. Shared evaluation

Perplexity uses up to 64 held-out sequences per set. Accuracy uses the first 32 GSM8K rows in the seeded file and 32 held-out OpenMath problems, greedy decoding, 256 new tokens. Latency is the median of 8 single-sequence generations of 32 tokens. Throughput is one batch of 4. Eval generation is greedy, and `compare/evaluate.py` seeds Python and PyTorch from `SEED` anyway.

```bash
bash compare/run.sh eval compare/configs/default.env
```

The report for the six SFT checkpoints is `compare/results/20261013T140249Z/report.md`.

## 4. Logit distillation

`kd/configs/default.env` trains ShiShu after reasoning SFT (`compare/runs/shishu/reasoning`) against the frozen fully trained 140M. The loss is forward KL on assistant tokens of the same 2,744 training conversations, plus a 0.1 weight on the student's next-token loss. Learning rate 2e-5, cosine, warmup 0.03, 2 epochs, batch 1, accumulation 4, seed 140. Full vocab logits are computed in chunks of 4,096 so the 128,256-way vector is never materialized.

```bash
bash kd/run.sh kd/configs/default.env
```

The student is `kd/runs/shishu`. Score only that checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python compare/evaluate.py \
  --env compare/configs/default.env \
  --checkpoint kd/runs/shishu \
  --out kd/results
```

`bash compare/run.sh eval` scores the six SFT checkpoints and does not include this one. The logit-KD report is `kd/results/report.md`.

## 5. Layerwise distillation

`kd/configs/layerwise.env` starts from `kd/runs/shishu` and adds normalized MSE between student blocks 0–14 and the teacher block of the same index. The teacher block output already includes that block's attention residual and its MLP residual. Student block 15 has no paired teacher block. The MSE weight is 1. The other hyperparameters match the logit run, including seed 140.

```bash
bash kd/run.sh kd/configs/layerwise.env
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python compare/evaluate.py \
  --env compare/configs/default.env \
  --checkpoint kd/runs/shishu-layerwise \
  --out kd/results/layerwise
```

The report is `kd/results/layerwise/report.md`.

## 6. Transplant student, not yet trained

`kd/transplant.py` builds a new ShiShu initialization from the fully trained 140M. Blocks 0–10 are copied whole. Blocks 11–14 copy the MLP and the post-attention norm and drop attention. Block 15 copies the last MLP and then zeros `down_proj`, so that extra block starts as a residual identity. The init is already written to `kd/runs/shishu-transplant` (layer-0 query cosine 1, 140,246,784 parameters). Training it is:

```bash
bash kd/run.sh kd/configs/transplant.env
```

The launcher rebuilds the init only when `kd/runs/shishu-transplant/model.safetensors` is absent. Training writes `kd/runs/shishu-transplant-kd`. The learning rate there is 1e-5, with the same layerwise loss. There is no eval for this run yet.

## Seeds

| Knob | Value | What it fixes |
| --- | --- | --- |
| `SEED` | 140 | Held-out split, GSM8K slice, KD shuffle, eval process seed |
| Reasoning split | 141 | `SEED + 1` inside `compare/prepare_data.py` |
| `SFT_SEED` | 42 | SFT example shuffle. This was the Trainer default during the reported runs |
| KD `--seed` / `--data_seed` | 140 | Distillation shuffle |

`compare/finetune.sh` and `kd/run.sh` export `PYTHONHASHSEED` before Python starts. Perplexity and exact match are deterministic given the checkpoint, because decoding is greedy and the loss is teacher-forced. A second training run will follow the same data order and the same two-GPU batch, and bfloat16 SDPA will still move the weights slightly relative to these checkpoints. Timing numbers are not part of the seed contract.
