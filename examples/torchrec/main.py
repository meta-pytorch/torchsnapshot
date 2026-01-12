#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import os
import uuid
from typing import Generator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchrec
import torchsnapshot
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.datasets.utils import Batch
from torchrec.distributed import ModuleSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingPlan, ShardingType
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

EPOCH_SIZE = 10
BATCH_SIZE = 8

EMBEDDING_DIM = 16
NUM_EMBEDDINGS = 200
DENSE_IN_FEATURES = 128
NUM_CLASSES = 8

TABLES = [
    torchrec.EmbeddingBagConfig(
        name="t1",
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        feature_names=["f1"],
        pooling=torchrec.PoolingType.SUM,
    ),
    torchrec.EmbeddingBagConfig(
        name="t2",
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        feature_names=["f2"],
        pooling=torchrec.PoolingType.SUM,
    ),
]

SHARDERS: List[ModuleSharder] = [
    EmbeddingBagCollectionSharder(
        fused_params={
            "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "learning_rate": 0.01,
            "eps": 0.01,
        }
    )
]


def get_data_loader(max_bag_size: int = 20) -> Generator[Batch, None, None]:
    for _ in range(EPOCH_SIZE):
        values = []
        lengths = []
        for _ in range(len(TABLES)):
            for _ in range(BATCH_SIZE):
                length = torch.randint(max_bag_size, (1,))
                values.append(torch.randint(EMBEDDING_DIM, (int(length.item()),)))
                lengths.append(length)
        yield Batch(
            dense_features=torch.rand((BATCH_SIZE, DENSE_IN_FEATURES)),
            sparse_features=torchrec.KeyedJaggedTensor(
                keys=["f1", "f2"],
                values=torch.cat(values),
                lengths=torch.cat(lengths),
            ),
            labels=torch.randn((BATCH_SIZE, NUM_CLASSES)),
        )


def get_rowwise_sharding_plan(
    module: torch.nn.Module, device: torch.device
) -> ShardingPlan:
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=dist.get_world_size(), compute_device=device.type),
        constraints={
            table.name: ParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value]
            )
            for table in TABLES
        },
    )
    pg = dist.group.WORLD
    assert pg is not None
    return planner.collective_plan(module=module, sharders=SHARDERS, pg=pg)


def train(work_dir: str, max_epochs: int, snapshot_path: Optional[str] = None) -> None:
    os.environ["TORCHSNAPSHOT_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ONLY"] = "1"
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(42)

    dlrm_model = DLRM(
        embedding_bag_collection=torchrec.EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=TABLES,
        ),
        dense_in_features=DENSE_IN_FEATURES,
        dense_arch_layer_sizes=[64, EMBEDDING_DIM],
        over_arch_layer_sizes=[64, NUM_CLASSES],
    )
    model = DLRMTrain(dlrm_model)

    dmp = torchrec.distributed.DistributedModelParallel(
        module=model,
        device=device,
        plan=get_rowwise_sharding_plan(model, device),
        sharders=SHARDERS,
    )

    optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(dmp.named_parameters())),
        lambda params: torch.optim.SGD(params, lr=0.01),
    )

    progress = torchsnapshot.StateDict(current_epoch=0)

    # torchsnapshot: define app state
    app_state = {
        "dmp": dmp,
        "optim": dmp.fused_optimizer,
        "progress": progress,
    }

    # torchsnapshot: restore from snapshot
    if snapshot_path is not None:
        snapshot = torchsnapshot.Snapshot(
            path=snapshot_path,
        )
        snapshot.restore(app_state=app_state)

    train_pipeline = torchrec.distributed.TrainPipelineSparseDist(
        model=dmp,
        optimizer=optimizer,
        device=device,
    )

    final_loss = None
    while progress["current_epoch"] < max_epochs:
        data_iter = iter(get_data_loader())
        while True:
            try:
                final_loss = train_pipeline.progress(data_iter)[0]
            except StopIteration:
                break

        progress["current_epoch"] += 1

        # torchsnapshot: take snapshot
        snapshot = torchsnapshot.Snapshot.take(
            path=f"{work_dir}/{uuid.uuid4()}",
            app_state=app_state,
            replicated=["**"],
        )
        print(f"Snapshot path: {snapshot.path}")
        print(f"Final loss: {final_loss}")

    # torchsnapshot: examine snapshot content
    if dist.get_rank() == 0:
        entries = snapshot.get_manifest()
        for path in entries.keys():
            print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--snapshot-path")
    parser.add_argument("--use-pet", action="store_true", default=False)
    parser.add_argument("--num-processes", type=int, default=2)
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
        pet.elastic_launch(lc, entrypoint=train)(
            args.work_dir, args.max_epochs, args.snapshot_path
        )
    else:
        train(
            work_dir=args.work_dir,
            max_epochs=args.max_epochs,
            snapshot_path=args.snapshot_path,
        )


if __name__ == "__main__":
    main()  # pragma: no cover
