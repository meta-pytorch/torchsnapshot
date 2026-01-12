#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from pathlib import Path
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot import Snapshot
from torchsnapshot.manifest_utils import is_fully_replicated_entry
from torchsnapshot.stateful import AppState
from torchsnapshot.test_utils import run_with_pet


@pytest.mark.parametrize(
    "replication_globs, expected_replicated_paths",
    [
        ([], ["0/ddp/module.weight", "0/ddp/module.bias"]),
        (None, ["0/ddp/module.weight", "0/ddp/module.bias"]),
        (
            ["**"],
            [
                "0/ddp/module.weight",
                "0/ddp/module.bias",
                "0/nonddp/weight",
                "0/nonddp/bias",
            ],
        ),
    ],
)
@run_with_pet(nproc=2)
def test_ddp_replication_glob(
    replication_globs: Optional[List[str]],
    expected_replicated_paths: List[str],
    tmp_path: Path,
) -> None:
    dist.init_process_group(backend="gloo")
    app_state: AppState = {
        "ddp": DDP(torch.nn.Linear(4, 3)),
        "nonddp": torch.nn.Linear(3, 2),
    }
    snapshot = Snapshot.take(
        path=str(tmp_path),
        app_state=app_state,
        replicated=replication_globs,
    )
    replicated_paths = [
        path
        for path, entry in snapshot.get_manifest().items()
        if is_fully_replicated_entry(entry)
    ]
    assert set(replicated_paths) == set(expected_replicated_paths)
