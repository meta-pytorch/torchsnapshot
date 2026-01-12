#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import os
import time
import uuid
from enum import Enum
from typing import List

import fsspec
import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchrec
import torchsnapshot
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed import DistributedModelParallel, ModuleSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchsnapshot.rss_profiler import measure_rss_deltas


NUM_TABLES = 2
EMBEDDING_DIM = 128
DENSE_IN_FEATURES = 128
NUM_CLASSES = 8


class BenchmarkType(Enum):
    TORCHSNAPSHOT = "torchsnapshot"
    TORCHSNAPSHOT_ASYNC = "torchsnapshot_async"
    TORCH_SAVE_PATH_MANAGER = "torch_save_path_manager"
    TORCH_SAVE_FSSPEC = "torch_save_fsspec"

    def __str__(self) -> str:
        return self.value


def rank_0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)


def initialize_dmp(device: torch.device, mb_per_gpu: int) -> DistributedModelParallel:
    num_embeddings_per_gpu = mb_per_gpu * 1024**2 // EMBEDDING_DIM // 4
    num_embeddings = num_embeddings_per_gpu * dist.get_world_size()
    tables = [
        torchrec.EmbeddingBagConfig(
            name=f"t{i}",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=num_embeddings // NUM_TABLES,
            feature_names=[f"f{i}"],
            pooling=torchrec.PoolingType.SUM,
        )
        for i in range(NUM_TABLES)
    ]
    sharders: List[ModuleSharder] = [
        EmbeddingBagCollectionSharder(
            fused_params={
                "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                "learning_rate": 0.01,
                "eps": 0.01,
            }
        ),
    ]

    dlrm_model = DLRM(
        embedding_bag_collection=torchrec.EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=tables,
        ),
        dense_in_features=DENSE_IN_FEATURES,
        dense_arch_layer_sizes=[64, EMBEDDING_DIM],
        over_arch_layer_sizes=[64, NUM_CLASSES],
    )
    model = DLRMTrain(dlrm_model)

    pg = dist.group.WORLD
    assert pg, "dist.group.WORLD set to None."
    plan = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=dist.get_world_size(),
            local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            compute_device=device.type,
        ),
        constraints={
            table.name: ParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value]
            )
            for table in tables
        },
    ).collective_plan(
        module=model,
        sharders=sharders,
        pg=pg,
    )

    dmp = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
    )
    return dmp


def benchmark_torchsnapshot(
    dmp: DistributedModelParallel, work_dir: str, benchmark_load: bool
) -> None:
    rank_0_print("Saving a checkpoint with torchsnapshot...")
    begin_ts = time.monotonic()
    snapshot = torchsnapshot.Snapshot.take(
        path=f"{work_dir}/{uuid.uuid4()}",
        app_state={"dmp": dmp},
        replicated=["**"],
    )
    rank_0_print(
        f"Completed saving with torchsnapshot (snapshot path: {snapshot.path})."
        f"Took {time.monotonic() - begin_ts:.2f} seconds."
    )
    if benchmark_load:
        raise NotImplementedError()


def benchmark_torchsnapshot_async(
    dmp: DistributedModelParallel, work_dir: str, benchmark_load: bool
) -> None:
    rank_0_print("Saving a checkpoint with torchsnapshot...")
    begin_ts = time.monotonic()
    future = torchsnapshot.Snapshot.async_take(
        path=f"{work_dir}/{uuid.uuid4()}",
        app_state={"dmp": dmp},
        replicated=["**"],
    )
    unblock_ts = time.monotonic()
    snapshot = future.wait()
    rank_0_print(f"Snapshot.async_take returned after {unblock_ts - begin_ts:.2f}")
    end_ts = time.monotonic()
    rank_0_print(
        f"Completed saving with torchsnapshot (snapshot path: {snapshot.path})."
        f"Took {end_ts - begin_ts:.2f} seconds "
        f"(blocked for {unblock_ts - begin_ts:.2f} seconds."
    )
    if benchmark_load:
        raise NotImplementedError()


def benchmark_torch_save_fsspec(
    dmp: DistributedModelParallel, work_dir: str, benchmark_load: bool
) -> None:
    rank_0_print("Saving a checkpoint with torch.save + fsspec...")
    begin_ts = time.monotonic()
    path = os.path.join(work_dir, str(uuid.uuid4()))
    with fsspec.open(path, "wb") as f:
        torch.save(dmp.state_dict(), f)
    dist.barrier()
    rank_0_print(
        "Completed saving with torch.save + fsspec."
        f"Took {time.monotonic() - begin_ts:.2f} seconds."
    )
    if benchmark_load:
        raise NotImplementedError()


def benchmark_torch_save_path_manager(
    dmp: DistributedModelParallel, work_dir: str, benchmark_load: bool
) -> None:
    from iopath.common.file_io import PathManager

    pm = PathManager()

    if work_dir.startswith("manifold://"):
        from iopath.fb.manifold import ManifoldPathHandler

        pm.register_handler(ManifoldPathHandler())

    rank_0_print("Saving a checkpoint with torch.save + PathManager...")
    begin_ts = time.monotonic()
    path = os.path.join(work_dir, str(uuid.uuid4()))
    with pm.open(path, "wb") as f:
        torch.save(dmp.state_dict(), f)
    dist.barrier()
    rank_0_print(
        "Completed saving with torch.save + PathManager."
        f"Took {time.monotonic() - begin_ts:.2f} seconds."
    )
    if benchmark_load:
        raise NotImplementedError()


def main(
    benchmark_type: BenchmarkType, work_dir: str, mb_per_gpu: int, benchmark_load: bool
) -> None:
    logger = logging.getLogger("torchsnapshot.scheduler")
    logger.setLevel(logging.DEBUG)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dmp = initialize_dmp(device=device, mb_per_gpu=mb_per_gpu)
    rss_deltas = []
    with measure_rss_deltas(rss_deltas=rss_deltas):
        if benchmark_type == BenchmarkType.TORCHSNAPSHOT:
            benchmark_torchsnapshot(
                dmp=dmp, work_dir=work_dir, benchmark_load=benchmark_load
            )
        elif benchmark_type == BenchmarkType.TORCHSNAPSHOT_ASYNC:
            benchmark_torchsnapshot_async(
                dmp=dmp, work_dir=work_dir, benchmark_load=benchmark_load
            )
        elif benchmark_type == BenchmarkType.TORCH_SAVE_PATH_MANAGER:
            benchmark_torch_save_path_manager(
                dmp=dmp, work_dir=work_dir, benchmark_load=benchmark_load
            )
        elif benchmark_type == BenchmarkType.TORCH_SAVE_FSSPEC:
            benchmark_torch_save_fsspec(
                dmp=dmp, work_dir=work_dir, benchmark_load=benchmark_load
            )
        else:
            raise ValueError(f"Unrecognized benchmark type: {benchmark_type}")
    print(f"Peak RSS delta: {max(rss_deltas) // 1024**2}MB")


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-type", type=BenchmarkType, choices=list(BenchmarkType)
    )
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--mb-per-gpu", type=int, default=4000)
    parser.add_argument("--benchmark-load", action="store_true", default=False)
    parser.add_argument("--use-pet", action="store_true", default=False)
    parser.add_argument("--num-processes", type=int, default=4)

    args: argparse.Namespace = parser.parse_args()
    if args.use_pet:
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=args.num_processes,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )
        pet.elastic_launch(lc, entrypoint=main)(
            args.benchmark_type, args.work_dir, args.mb_per_gpu, args.benchmark_load
        )
    else:
        main(
            benchmark_type=args.benchmark_type,
            work_dir=args.work_dir,
            mb_per_gpu=args.mb_per_gpu,
            benchmark_load=args.benchmark_load,
        )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
