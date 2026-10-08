# Train ShiShu MobileLLM-R1-140M from this repo

Everything the launchers need lives in this directory: the surgeried
checkpoint, a small data mix, configs, and scripts. After a clone, the A100
box should not need `/data/cs26e003/...` paths.

The 140M checkpoint is stored as `model.safetensors.part-*` because GitHub
blocks Git LFS on public forks. `scripts/setup.sh` and
`scripts/prepare_full_training.sh` concatenate the shards automatically.

```bash
git clone <this-repo>
cd MobileLLM-R1-Chopped-Off
```

## Run (Ada 6000 or A100)

```bash
bash scripts/setup.sh                          # creates ./.venv
PYTHON=.venv/bin/python bash scripts/prepare_full_training.sh configs/train.env
PYTHON=.venv/bin/python bash scripts/run_full_training.sh configs/train.env all
```

`configs/train.env` uses **repo-relative** paths (`models/`, `data/`,
`checkpoints/`, `logs/`) and `NPROC_PER_NODE=auto`. On the A100 node, either
keep that file or switch to the larger-batch recipe:

```bash
PYTHON=.venv/bin/python bash scripts/run_full_training.sh configs/a100.env all
```

The only usual A100 edits inside `configs/a100.env` are `NPROC_PER_NODE` (or
leave `auto`) and `MASTER_ADDR` if you go multi-node.

## What is in the tree

| Path | Role |
| --- | --- |
| `models/MobileLLM-R1-140M-shishu/` | Surgeried 140M checkpoint + tokenizer |
| `data/pretrain/`, `data/midtrain/`, `data/sft/` | Small JSONL mix (1 node shard `1/`) |
| `configs/train.env` | Portable short run, auto GPU count |
| `configs/a100.env` | Same paths, larger batch, paper-scale steps |
| `checkpoints/`, `logs/` | Created at runtime, gitignored |

One JSONL file per source divides any GPU count, so you do not re-shard when
moving from 3 Ada GPUs to 8 A100s.

## Grow the mix (optional)

```bash
.venv/bin/python scripts/prepare_training_data.py --nodes 1 --chunks 1 --docs 20000 --sft_rows 10000
```

`--data_root` defaults to `./data`. `--chunks` must divide GPUs per node (1
works everywhere).

Mid-training in the paper is KL distillation from Llama-3.1-8B-Instruct. That
teacher loop is not in this repo, so mid-training here is the same next-token
loop as pre-training, on the mid-training mix.

Paper-scale 2T-token pretrain still needs a large cluster disk; this repo
ships a small mix so the **code path** is identical after clone.
