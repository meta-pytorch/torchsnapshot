#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torchsnapshot import Snapshot
from torchsnapshot.test_utils import check_state_dict_eq, run_with_pet


def _create_fsdp_model(
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Linear(32, 32)

    fsdp_model = FSDP(
        module=model,
        device_id=device,
    )
    FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
    return fsdp_model


@pytest.mark.skipif(
    bool(not torch.cuda.is_available()), reason="The test requires GPUs to run."
)
@pytest.mark.skipif(
    bool(torch.cuda.device_count() < 2), reason="At least two GPUs are required."
)
@run_with_pet(nproc=2)
def test_model_and_optim_fsdp(tmp_path: Path) -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    fsdp_model = _create_fsdp_model(17, device)

    snapshot = Snapshot.take(
        path=str(tmp_path),
        app_state={"fsdp_model": fsdp_model},
    )
    state_dict_from_method = snapshot.get_state_dict_for_key("fsdp_model")
    FSDP.set_state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT)

    full_state_dict = fsdp_model.state_dict()
    for k, v in full_state_dict.items():
        full_state_dict[k] = v.cpu()

    assert check_state_dict_eq(full_state_dict, state_dict_from_method)
