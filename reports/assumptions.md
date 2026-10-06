# Assumptions

| Assumption | Result |
| --- | --- |
| One RTX 4090, CUDA 12.4, torch 2.7.0+cu126, transformers 5.x | REFUTED/UNVERIFIED: this host reports no CUDA device, torch 2.14.1+cu130, and transformers 5.18.0. |
| The model is `Llama4ForCausalLM` with 15 layers, hidden size 576, 9 attention heads, 3 KV heads, vocab 128256, and tied embeddings | CONFIRMED from the downloaded config. |
| The smoke test should use fp32 and a batch larger than one | CONFIRMED in `configs/smoke_test.yaml`; this exercises padding and avoids tiny bf16 updates. |
| The post-trained model requires a chat template for production use | CONFIRMED by the model snapshot; the raw-text smoke data intentionally does not apply it. |
| TRL's SFT entry point can consume a local JSONL and run one rank | UNVERIFIED until `scripts/run_repo_sft_smoke.sh` is run with all TRL dependencies. |
| Full evaluation uses the repository's vLLM math harness, 0-shot, 30,000 generated tokens, and sampling temperature 0.6 | CONFIRMED from `evaluation/run_eval.sh`; a tiny offline slice is UNVERIFIED when vLLM/data are unavailable. |

No ShishuLM behavior or architecture has been implemented or inferred.
