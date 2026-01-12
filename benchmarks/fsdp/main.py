# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
from enum import Enum
from functools import partial
from uuid import uuid4

import torch
from torch import distributed as dist, nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torchsnapshot import Snapshot


class BenchmarkType(Enum):
    TORCHSNAPSHOT = "torchsnapshot"
    TORCH_SAVE = "torch_save"

    def __str__(self):
        return self.value


def rank_0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)


def create_model() -> nn.Module:
    # 7.8GB model, 1.9B parameters
    model = nn.Transformer(
        d_model=864,
        num_encoder_layers=1,
        num_decoder_layers=20,
        nhead=12,
        dim_feedforward=50257,
    )

    # 80GB 21B parameters
    # model = nn.Transformer(
    #     d_model=4000,
    #     num_encoder_layers=1,
    #     num_decoder_layers=40,
    #     nhead=40,
    #     dim_feedforward=50257,
    # )

    model_size = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rank_0_print(f"model parameters: {model_params:,}")
    rank_0_print(f"model size: {model_size / (1024**3):.3} GB")

    return FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                nn.TransformerDecoderLayer,
                nn.TransformerEncoderLayer,
            },
        ),
        device_id=int(os.environ["LOCAL_RANK"]),
    )


def benchmark_torchsnapshot(
    model: nn.Module, save_dir: str, benchmark_load: bool
) -> None:
    rank_0_print("Saving a checkpoint with torchsnapshot...")
    app_state = {"model": model}
    begin_ts = time.monotonic()
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        Snapshot.take(
            path=save_dir,
            app_state=app_state,
        )
    dist.barrier()
    end_ts = time.monotonic()
    rank_0_print(
        f"Completed saving with torchsnapshot (snapshot path: {save_dir}).\n"
        f"Took {end_ts - begin_ts:.2f} seconds."
    )

    if benchmark_load:
        rank_0_print("Loading the checkpoint with torchsnapshot...")
        begin_ts = time.monotonic()
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            snapshot = Snapshot(path=save_dir)
            snapshot.restore(app_state)
        end_ts = time.monotonic()
        rank_0_print(
            f"Completed loading with torchsnapshot.\n"
            f"Took {end_ts - begin_ts:.2f} seconds."
        )


def benchmark_torchsave(model: nn.Module, save_dir: str, benchmark_load: bool) -> None:
    rank_0_print("Saving a checkpoint with torch.save...")

    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{save_dir}/state_dict-{dist.get_rank()}.pt"

    begin_ts = time.monotonic()
    with FSDP.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
    ):
        state_dict = model.state_dict()
        torch.save(state_dict, save_file)
    dist.barrier()
    end_ts = time.monotonic()
    rank_0_print(
        f"Completed saving with torch.save (path: {save_dir}).\n"
        f"Took {end_ts - begin_ts:.2f} seconds."
    )

    if benchmark_load:
        begin_ts = time.monotonic()
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model.load_state_dict(torch.load(save_file))
        dist.barrier()
        end_ts = time.monotonic()
        rank_0_print(
            f"Completed loading with torch.save.\nTook {end_ts - begin_ts:.2f} seconds."
        )


@record
def main(benchmark_type: BenchmarkType, work_dir: str, benchmark_load: bool) -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    save_dir = f"{work_dir}/{uuid4()}"
    object_list = [None] * dist.get_world_size()
    object_list[dist.get_rank()] = save_dir
    dist.broadcast_object_list(object_list=object_list, src=0)
    save_dir = object_list[0]

    model = create_model()
    model.to(device)

    if benchmark_type == BenchmarkType.TORCHSNAPSHOT:
        benchmark_torchsnapshot(model, save_dir, benchmark_load)
    elif benchmark_type == BenchmarkType.TORCH_SAVE:
        benchmark_torchsave(model, save_dir, benchmark_load)
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
