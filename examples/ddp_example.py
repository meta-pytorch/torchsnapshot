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
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot import Snapshot, Stateful

NUM_EPOCHS = 4
EPOCH_SIZE = 16
BATCH_SIZE = 8


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


def train(
    work_dir: str,
    snapshot_path: Optional[str] = None,
) -> None:
    # initialize the process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.manual_seed(42)

    print(f"Running basic DDP example on device {device}.")
    model = Model().to(device)

    # DDP wrapper around model
    ddp_model = DDP(model)

    optim = torch.optim.Adagrad(ddp_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    progress = torchsnapshot.StateDict(current_epoch=0)

    # torchsnapshot: define app state
    app_state: Dict[str, Stateful] = {
        "rng_state": torchsnapshot.RNGState(),
        "model": ddp_model,
        "optim": optim,
        "progress": progress,
    }
    snapshot: Optional[Snapshot] = None

    if snapshot_path is not None:
        # torchsnapshot: restore app state
        snapshot = torchsnapshot.Snapshot(path=snapshot_path)
        print(f"Restoring snapshot from path: {snapshot.path}")
        snapshot.restore(app_state=app_state)

    while progress["current_epoch"] < NUM_EPOCHS:
        for _ in range(EPOCH_SIZE):
            X = torch.rand((BATCH_SIZE, 128), device=device)
            pred = ddp_model(X)
            label = torch.rand((BATCH_SIZE, 1), device=device)
            loss = loss_fn(pred, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        progress["current_epoch"] += 1

        # torchsnapshot: take snapshot
        snapshot = torchsnapshot.Snapshot.take(
            f"{work_dir}/run-{uuid.uuid4()}-epoch-{progress['current_epoch']}",
            app_state,
            replicated=["**"],  # this pattern treats all states as replicated
        )

        print(f"Snapshot path: {snapshot.path}")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--snapshot-path")
    args: argparse.Namespace = parser.parse_args()

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

    pet.elastic_launch(lc, entrypoint=train)(args.work_dir, args.snapshot_path)


if __name__ == "__main__":
    main()  # pragma: no cover
