# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
import sys
from logging import Logger
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[1]
_PRETRAIN_DIR = Path(__file__).resolve().parent
for _path in (str(_ROOT), str(_PRETRAIN_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from torch import distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from src.model_io import checkpoint_has_weights, load_causal_lm
from src.shishu_llama import register_shishu_llama
from utils.multi_jsonl import MultiJSONLIterator

from utils.pretrain_trainer import PretrainTrainer
from utils.process_args import process_args


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


log: Logger = get_logger("mobileLLM")


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def get_local_world_size() -> int:
    if os.environ.get("LOCAL_WORLD_SIZE"):
        return int(os.environ["LOCAL_WORLD_SIZE"])
    return torch.cuda.device_count() or 1


def get_folder_paths(directory: str) -> List[str]:
    folder_paths = [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    ]
    return folder_paths


def train() -> None:
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
    )
    model_args, data_args, training_args = process_args()

    global_rank = get_global_rank()
    local_rank = get_local_rank()

    log.info(f"Global Rank: {global_rank}")
    log.info(f"Local Rank: {local_rank}")
    register_shishu_llama()
    model_path = model_args.input_model_filename
    if model_args.init_from == "checkpoint" and not checkpoint_has_weights(model_path):
        raise FileNotFoundError(
            f"No model.safetensors in {model_path}. "
            "Copy the ShiShu MobileLLM-R1 checkpoint there before training."
        )
    if model_args.init_from == "scratch":
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_config(config)
        log.info("Initialized model weights from scratch using %s", model_path)
    elif model_args.init_from == "checkpoint":
        model = load_causal_lm(model_path, local_files_only=True)
    else:
        raise ValueError(f"--init_from must be 'checkpoint' or 'scratch', got {model_args.init_from}")
    if getattr(model.config, "_attn_implementation", None) in (None, "eager"):
        model.config._attn_implementation = "sdpa"
    log.info("attention implementation is %s", model.config._attn_implementation)
    log.info(
        "model size is "
        + str(sum(param.numel() for param in model.model.parameters()) / 1024 / 1024)
    )
    log.info("Start to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    log.info("Complete tokenizer loading...")

    # Each node reads its own shard directory; ranks on a node split that shard.
    procs_per_node = get_local_world_size()
    local_data_folder = os.path.join(
        data_args.train_data_local_path, str(global_rank // procs_per_node + 1)
    )
    log.info("world_rank for data loader is " + str(local_rank))
    log.info("world_size for data loader is " + str(procs_per_node))
    assert os.path.isdir(local_data_folder), local_data_folder
    folder_paths_string = ",".join(sorted(get_folder_paths(local_data_folder)))
    train_data = MultiJSONLIterator(
        tokenizer=tokenizer,
        data=folder_paths_string,
        instruct_data="",
        seq_len=training_args.model_max_length,
        batch_size=training_args.per_device_train_batch_size,
        buffer_size=data_args.buffer_size,
        world_rank=local_rank,
        world_size=procs_per_node,
        multiprocess=True,
        max_precompute=500,
        ignore_extra_chunks=False,
    )
    if training_args.resume_from_checkpoint == "auto":
        training_args.resume_from_checkpoint = (
            get_last_checkpoint(training_args.output_dir)
            if os.path.isdir(training_args.output_dir)
            else None
        )
    trainer = PretrainTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=None,
        data_collator=default_data_collator,
    )
    torch.distributed.barrier(device_ids=[local_rank])

    if training_args.do_train:
        _ = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
        train_data.close()

    torch.distributed.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    train()
