# ShiShu vs MobileLLM-R1-140M-base

Both models ran the same Facebook post-training loop: Tulu-3 general SFT, then OpenMathReasoning SFT.
Accuracy, perplexity, latency, and throughput use the same held-out rows, prompts, and token budgets.

Device: `cuda`

| Model | Stage | PPL general | PPL reasoning | GSM8K acc | Reasoning acc | Median latency (s) | Throughput (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shishu | kd | 791.999 | 16.854 | 0.000 | 0.062 | 0.288 | 388.596 |

Raw numbers are in `metrics.json` beside this file.
