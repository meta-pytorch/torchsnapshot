# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time
import uuid
from enum import Enum
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from torchsnapshot.tricks.deepspeed import patch_engine_to_use_torchsnapshot
from transformers import OPTConfig, OPTModel
from transformers.deepspeed import HfDeepSpeedConfig

dschf: Optional[HfDeepSpeedConfig] = None


# https://arxiv.org/pdf/2205.01068.pdf
TRAIN_BATCH_SIZE = 1024**2
NUM_HIDDEN_LAYERS = 48
NUM_ATTENTION_HEADS = 56
HIDDEN_SIZE = 7168


class BenchmarkType(Enum):
    TORCHSNAPSHOT = "torchsnapshot"
    DEEPSPEED = "deepspeed"

    def __str__(self):
        return self.value


def rank_0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)


def initialize_deepspeed_opt() -> DeepSpeedEngine:
    ds_config = {
        "train_batch_size": TRAIN_BATCH_SIZE,
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-4,
                "weight_decay": 0.01,
            },
        },
    }
    # HfDeepSpeedConfig must be created before instantiating the model and and kept alive.
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    dschf = HfDeepSpeedConfig(ds_config)  # noqa

    with deepspeed.zero.Init():
        config = OPTConfig(
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
        )
        model = OPTModel(config)

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config_params=ds_config
    )
    return engine


def benchmark_torchsnapshot(
    engine: DeepSpeedEngine, save_dir: str, benchmark_load: bool
) -> None:
    patch_engine_to_use_torchsnapshot(engine)

    rank_0_print("Saving a checkpoint with torchsnapshot...")
    begin_ts = time.monotonic()
    engine.save_checkpoint(save_dir=save_dir)
    rank_0_print(
        f"Completed saving with torchsnapshot (snapshot path: {save_dir}).\n"
        f"Took {time.monotonic() - begin_ts:.2f} seconds."
    )

    if benchmark_load:
        del engine
        engine = initialize_deepspeed_opt()
        patch_engine_to_use_torchsnapshot(engine)

        rank_0_print("Loading the checkpoint with torchsnapshot...")
        begin_ts = time.monotonic()
        engine.load_checkpoint(load_dir=save_dir)
        rank_0_print(
            f"Completed loading with torchsnapshot.\n"
            f"Took {time.monotonic() - begin_ts:.2f} seconds."
        )


def benchmark_deepspeed(
    engine: DeepSpeedEngine, save_dir: str, benchmark_load: bool
) -> None:
    rank_0_print("Saving a checkpoint with DeepSpeedEngine.save_checkpoint()...")
    begin_ts = time.monotonic()
    engine.save_checkpoint(save_dir=save_dir)
    rank_0_print(
        f"Completed saving with DeepSpeedEngine.save_checkpoint() (save_dir: {save_dir}).\n"
        f"Took {time.monotonic() - begin_ts:.2f} seconds."
    )
    if benchmark_load:
        del engine
        engine = initialize_deepspeed_opt()
        rank_0_print("Loading the checkpoint with DeepSpeedEngine.save_checkpoint()...")
        begin_ts = time.monotonic()
        engine.load_checkpoint(load_dir=save_dir)
        rank_0_print(
            f"Completed loading with DeepSpeedEngine.load_checkpoint().\n"
            f"Took {time.monotonic() - begin_ts:.2f} seconds."
        )


def main(benchmark_type: BenchmarkType, work_dir: str, benchmark_load: bool) -> None:
    logger = logging.getLogger("torchsnapshot.scheduler")
    logger.setLevel(logging.DEBUG)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    save_dir = f"{work_dir}/{uuid.uuid4()}"
    object_list = [None] * dist.get_world_size()
    object_list[dist.get_rank()] = save_dir
    dist.broadcast_object_list(object_list=object_list, src=0)
    save_dir = object_list[0]

    engine = initialize_deepspeed_opt()
    if benchmark_type == BenchmarkType.TORCHSNAPSHOT:
        benchmark_torchsnapshot(
            engine=engine, save_dir=save_dir, benchmark_load=benchmark_load
        )
    elif benchmark_type == BenchmarkType.DEEPSPEED:
        benchmark_deepspeed(
            engine=engine, save_dir=save_dir, benchmark_load=benchmark_load
        )
    else:
        raise ValueError(f"Unrecognized benchmark type: {benchmark_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-type",
        type=BenchmarkType,
        choices=list(BenchmarkType),
        default=BenchmarkType.TORCHSNAPSHOT,
    )
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--benchmark-load", action="store_true", default=False)

    args: argparse.Namespace = parser.parse_args()
    main(
        benchmark_type=args.benchmark_type,
        work_dir=args.work_dir,
        benchmark_load=args.benchmark_load,
    )
