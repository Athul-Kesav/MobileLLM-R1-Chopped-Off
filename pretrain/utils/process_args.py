# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative path"}
    )
    init_from: str = field(
        default="checkpoint",
        metadata={"help": "'checkpoint' loads input weights; 'scratch' only uses its config."},
    )


@dataclass
class DataArguments:
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )
    buffer_size: int = field(
        default=2048,
        metadata={"help": "Sequences per shuffle buffer; must be filled before the first batch."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )
    min_lr_ratio: float = field(
        default=0.1,
        metadata={"help": "Final learning rate as a fraction of the peak."},
    )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "TensorBoard log directory (kept for transformers 5)."},
    )


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args
