# Results

All numbers below are from this checkout, on the same held-out split, unless a figure is marked as published. The local data is a 1,372-row training slice of Tulu-3 and a 1,372-row training slice of OpenMathReasoning, with 128 held-out conversations each. GSM8K accuracy is exact match on 32 questions drawn from the test set. Perplexity is the exponential of the mean assistant-token loss, and sequences longer than 2,048 tokens are cropped from the left so the answer stays in the window. A uniform distribution over the 128,256-token vocabulary has loss 11.76 nats.

The ShiShu student has 140,246,784 parameters. The MobileLLM-R1-140M base has 140,248,512. ShiShu is 11 full decoder blocks followed by 5 MLP-only blocks. The base model is 15 full decoder blocks. Residual width is 576 and the SwiGLU inner width is 2,048 on both. The ShiShu checkpoint that training started from is not a copy of the 140M weights: corresponding tensors have cosine similarity about 0.

## Supervised finetuning

Source: `compare/results/20261013T140249Z`. Device: CUDA, two RTX 6000 Ada GPUs.

| Model | Stage | General loss | General PPL | Reasoning loss | Reasoning PPL | GSM8K | OpenMath holdout | ms / token | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShiShu | init | 11.879 | 144240 | 11.861 | 141686 | 0/32 | 0/32 | 9.11 | 383.6 |
| ShiShu | after general SFT | 11.553 | 104040 | 11.841 | 138794 | 0/32 | 0/32 | 9.42 | 289.0 |
| ShiShu | after reasoning SFT | 11.094 | 65753 | 4.929 | 138.3 | 1/32 | 0/32 | 9.17 | 373.6 |
| Base | init | 3.015 | 20.38 | 1.288 | 3.63 | 0/32 | 4/32 | 10.95 | 298.7 |
| Base | after general SFT | 2.983 | 19.74 | 1.288 | 3.63 | 1/32 | 1/32 | 11.08 | 299.7 |
| Base | after reasoning SFT | 3.323 | 27.75 | 1.133 | 3.11 | 1/32 | 2/32 | 10.95 | 302.5 |

The last logged training losses, from the trainer state at the end of each stage, were 11.48 and then 4.26 for ShiShu, and 2.60 and then 0.82 for the base. Those sit next to the held-out losses above, so the eval gap is not a case of the training loss looking finished while the held-out loss is still large.

General SFT barely moves either model. ShiShu stays within 0.3 nats of a uniform vocabulary on both eval sets and still scores nothing. The base model's reasoning loss does not change at all in the reported precision (1.288), and its general loss moves from 3.015 to 2.983. On 1,372 Tulu-3 rows, two epochs at 5e-6 is not enough to teach a randomly initialized 140M network, and it is also not enough to move a model that is already trained.

Reasoning SFT is the only stage that changes ShiShu in a way that shows up on the eval. Held-out math loss falls from 11.86 to 4.93 (perplexity 141,686 to 138). That is real learning: the final training window was 4.26, so the model fit the small math set and generalized to the holdout at a similar loss. It does not become a general language model. General-text loss only falls from 11.88 to 11.09, still next to the uniform value of 11.76. Exact match is 1 of 32 GSM8K questions and 0 of 32 held-out OpenMath answers.

The base model is already strong on this math holdout before any local training (loss 1.29, perplexity 3.63, 4 of 32 exact). Reasoning SFT pushes that loss to 1.13 and the perplexity to 3.11, and it costs general-text perplexity, which rises from 20.4 to 27.8. Exact match does not follow the perplexity. OpenMath exact match goes 4/32, then 1/32, then 2/32. GSM8K is 0, then 1, then 1. With 32 questions, one extra hit is 3.1 points, so those exact-match differences do not rank the three base checkpoints. The perplexity does: local reasoning SFT helps the base model on math text and hurts it on general text.

ShiShu is faster per generated token at the same parameter count, about 9.1 ms against 11.0 ms for the base, roughly 17 percent less time. That fits the architecture: five of its sixteen blocks have no attention. The throughput column is a worse comparison. The ShiShu general checkpoint reports 289 tok/s while its latency stays at 0.29 s, which happens when generations emit padding or end early and the token counter drops. Latency is the stable speed measurement.

## Logit distillation

Source: `kd/results`. Student: ShiShu after reasoning SFT. Teacher: the fully trained `models/MobileLLM-R1-140M`, frozen. Loss: forward KL on assistant tokens plus a 0.1 weight on the student's own next-token loss. Two epochs, learning rate 2e-5, sequence length 2,048.

| Checkpoint | General loss | General PPL | Reasoning loss | Reasoning PPL | GSM8K | OpenMath holdout | ms / token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShiShu after reasoning SFT | 11.094 | 65753 | 4.929 | 138.3 | 1/32 | 0/32 | 9.17 | 373.6 |
| After logit KD | 9.811 | 18225 | 4.621 | 101.6 | 2/32 | 0/32 | 9.30 | 371.9 |

This is the best ShiShu GSM8K result in the set, 2 of 32. Reasoning perplexity moves from 138 to 102, toward the base model's 3.1 but still about 30 times higher. General loss moves from 11.09 to 9.81. That is the first time ShiShu is clearly below the uniform line on general text, and it is still far from the base model's 3.3. The teacher is a useful target for the math distribution the student had started to fit. It does not, on this data budget, install the rest of the teacher's language-model behavior into an uncorrelated initialization.

## Layerwise distillation

Source: `kd/results/layerwise`, checkpoint `kd/runs/shishu-layerwise`. Same student and teacher as the logit run. The extra loss is normalized MSE between student blocks 0–14 and teacher blocks 0–14. Block 15 has no teacher block and is left to the logit term. Coefficient 1.0 on that MSE, on top of the same KL and next-token terms.

| Checkpoint | General loss | General PPL | Reasoning loss | Reasoning PPL | GSM8K | OpenMath holdout | ms / token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| After logit KD | 9.811 | 18225 | 4.621 | 101.6 | 2/32 | 0/32 | 9.30 | 371.9 |
| After layerwise KD | 9.780 | 17681 | 4.727 | 113.0 | 1/32 | 1/32 | 8.98 | 391.6 |

Training ran 686 steps, two epochs, in 729 seconds. The combined logged loss fell from 30.3 to about 26.9 and then sat there. The hidden-state term fell from 1.97 to about 1.23. Blocks 0–10 finished near 1.14 and blocks 11–14 near 1.50, so the tail was harder, but both stayed order-1 after normalization. The token KL stayed around 4, which is where the logit-only run already was. Held-out math got worse (perplexity 102 to 113). GSM8K went from 2/32 back to 1/32. The one new exact match is on the OpenMath holdout.

The inference is that hidden-state matching did some of what it was asked to do, and that this was the wrong extra objective for this student. Normalized MSE of 1.2 means the block outputs are still far apart after layer-norm, which is what an uncorrelated initialization should look like: layers 0–10 are the same kind of block, but they are not the teacher's weights, so there is no free alignment to preserve. Spending gradient on that match competed with the token KL, and the KL did not improve. The small exact-match shuffle (2/32 to 1/32, and 0/32 to 1/32 on OpenMath) is one question each way. It should not be read as a ranking.

## What the 6 percent comparison is

The [MobileLLM-R1-140M model card](https://huggingface.co/facebook/MobileLLM-R1-140M) reports GSM8K at 4.1 percent for the post-trained 140M, 0-shot pass@1 on the full test set. The 6.2 percent on that card is MATH500. The 16.3 percent GSM8K figure is the base checkpoint under 8-shot exact match.

The local logit-KD result of 2/32 is 6.25 percent on a 32-question slice, 0-shot, with a 256-token budget. That slice was never run on the fully trained teacher. The local base model, which is the pretrained base rather than that teacher, scores 0 or 1 out of 32 here. So 2/32 is the best number this ShiShu line has produced on this slice. It is not a reproduction of the paper's GSM8K number, and it is not evidence that the student has caught the teacher.

## What was not run

`kd/runs/shishu-transplant` is a ShiShu student whose blocks 0–10 are copied from the fully trained 140M, whose blocks 11–14 keep that model's MLP weights and drop attention, and whose block 15 starts as a residual identity (`down_proj` is zero). It has 140,246,784 parameters and layer-0 query weights match the teacher with cosine 1. It has not been trained or evaluated. The config for that run is `kd/configs/transplant.env`.

## Reproducibility

The split, the GSM8K slice, and the distillation shuffle are seed 140. The reported SFT runs used the Trainer default seed, 42, for the example shuffle. Both are now set explicitly: `SEED=140` and `SFT_SEED=42` in `compare/configs/default.env`, and `SEED=140` in the KD configs. Eval generation is greedy (`do_sample` is false), so accuracy and perplexity do not depend on a sampling seed. The launchers export `PYTHONHASHSEED` to the same value as the training seed.

A rerun on this machine with GPUs 0 and 2 free follows the same data order and the same two-process batch. It will not be bit-exact. Training is bfloat16 SDPA, and those kernels are not deterministic. Latency and throughput move between processes even when the checkpoint is unchanged. `compare/finetune.sh` resumes a finished stage automatically, so a second launch on the existing `compare/runs` directories exits immediately. A clean replay needs empty output directories.

The world size is part of the result. These runs used two processes because GPU 1 was occupied. The configs now pin `CUDA_VISIBLE_DEVICES=0,2` and `NPROC_PER_NODE=2`. Leaving the process count on auto, and then finding three free GPUs, would change the batch and would not replay these curves.
