# Reconnaissance

The native entry points are `pretrain/pretrain.py`, `sft/sft.py`, and
`evaluation/math_eval.py`, launched by their adjacent shell scripts. Pretraining
expects a node-sharded directory of JSONL data and the provided script launches
8 ranks. SFT is TRL-based and the checked-in recipes launch `torchrun` with 2
or 128 ranks; `sft.py` initializes a distributed process group unconditionally.
Its arguments include local model paths, dataset names/mixtures, sequence length,
batch size, and `max_steps`, but the local JSONL adapter is not documented by
upstream and must be verified by the wrapper.

Evaluation uses the repository's vLLM-backed math harness, registers
`Llama4ForCausalLM`, applies optional chat templates, and defaults to raw/tool
prompts, 0 shots, and up to 30,000 generated tokens in `run_eval.sh`.
Pretraining consumes tokenized-style JSONL text shards and a full-corpus run is
intentionally out of scope. The local smoke path uses raw text and the same
AutoModel loader, but it is not a benchmark.

The installed environment is recorded in `reports/env_report.json`; it differs
from the plan's assumed CUDA host. No upstream files were modified.
