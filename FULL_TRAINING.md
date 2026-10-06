# Full MobileLLM-R1 training setup

This workspace now preserves the upstream `pretrain/`, `sft/`, and
`evaluation/` folders and adds launch scaffolding for the paper's 16-node,
8-GPU-per-node recipe. It does **not** download multi-terabyte corpora or
install packages automatically.

## What is covered

The upstream code supports:

- pre-training phase 1 and phase 2: 500,000 steps each, 2,048 tokens;
- post-training general SFT: 2 epochs, 4,096 tokens;
- post-training reasoning SFT: 4 epochs, 32,768 tokens;
- repository evaluation through `evaluation/math_eval.py`.

The upstream repository does **not** contain the paper's mid-training
knowledge-distillation loop. The paper describes an 8B Llama-3.1-Instruct
teacher and KL-logit distillation, but this workspace will refuse to launch
mid-training rather than invent an incompatible implementation. A KD
implementation and its exact data/logit contract must be supplied separately.

## Setup

```bash
cp configs/full_training.env.example configs/full_training.env
# Edit model SHAs, external paths, and dataset identifiers.
bash scripts/prepare_full_training.sh configs/full_training.env
bash scripts/download_full_models.sh configs/full_training.env
```

Models are stored under `MODEL_ROOT`, outside the repository. Credentials are
read only from `HF_TOKEN` or the existing Hugging Face login state.

Pre-training data must match the upstream loader layout:

```text
$DATA_ROOT/pretrain/phase1/1/<dataset-name>:<weight>/data.jsonl
...
$DATA_ROOT/pretrain/phase1/16/<dataset-name>:<weight>/data.jsonl
```

The same numbered-node layout is required for phase 2 and both mid-training
data roots. The data mix ratios and source datasets are listed in the
upstream README; the launcher does not pretend to have downloaded them.

## Launch

Run stages explicitly and inspect each checkpoint before proceeding:

```bash
bash scripts/run_full_training.sh configs/full_training.env pretrain-phase1
bash scripts/run_full_training.sh configs/full_training.env pretrain-phase2
bash scripts/run_full_training.sh configs/full_training.env sft-general
bash scripts/run_full_training.sh configs/full_training.env sft-reasoning
```

`midtrain-phase1` and `midtrain-phase2` intentionally exit with a clear
unsupported-stage error until a verified KD implementation is added. The
`eval` stage points to the untouched repository evaluation harness and
requires its datasets and vLLM installation.

The launcher requires exactly `NNODES=16` and `NPROC_PER_NODE=8`, uses
`torchrun`, enables bf16 for pre-training, and writes logs/checkpoints outside
the repository. It does not automatically chain all stages or resume a
checkpoint without an explicit model path.
