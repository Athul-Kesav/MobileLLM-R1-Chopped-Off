# ShiShu vs MobileLLM-R1-140M-base

Both checkpoints go through the same Facebook post-training loop, then the same measurements.

| Stage | Facebook dataset | What this run uses |
| --- | --- | --- |
| 1. General SFT | [Tulu-3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225) (866K) | `data/sft/general/train.jsonl`, held-out rows removed |
| 2. Reasoning SFT | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) (3.2M) | `data/sft/reasoning/train.jsonl`, held-out rows removed |

Stage 2 starts from the stage-1 checkpoint of that same model. Hyperparameters follow the paper (AdamW, zero weight decay, general LR `5e-6` linear with warmup `0.03` for 2 epochs, reasoning LR `8e-5` cosine with warmup `0.1` for 4 epochs). Sequence length and batch size in `configs/default.env` are the local 140M settings. Facebook also trains reasoning SFT on OpenScienceReasoning-2 and OpenCodeReasoning-2; those sets are not in this loop.

Measured on the shared held-out split, with the same prompts and token budgets:

- **Perplexity** on assistant tokens (Tulu-3 holdout and OpenMathReasoning holdout)
- **Accuracy** as exact match on GSM8K and on held-out OpenMath boxed answers
- **Latency** as median batch-1 generation time
- **Throughput** as generated tokens per second at a fixed batch

`init` is the untouched checkpoint. `general` and `reasoning` are the checkpoints after each stage.

## Run

From the repository root, with a CUDA machine:

```bash
bash compare/run.sh prepare compare/configs/default.env
bash compare/run.sh finetune compare/configs/default.env
bash compare/run.sh eval compare/configs/default.env
```

`bash compare/run.sh all` runs all three. `prepare` assembles `models/MobileLLM-R1-140M-shishu`, downloads `facebook/MobileLLM-R1-140M-base` into `models/MobileLLM-R1-140M-base`, copies the chat template onto the base snapshot if it has none, and writes `compare/data/`.

The report lands in `compare/results/<timestamp>/report.md` (`compare/results/latest` points at the newest one).

A shorter pass, for a pipeline check:

```bash
GENERAL_MAX_STEPS=20 REASONING_MAX_STEPS=20 bash compare/run.sh finetune
```
