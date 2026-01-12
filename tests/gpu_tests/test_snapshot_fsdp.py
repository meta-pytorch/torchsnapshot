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
from torchsnapshot.tricks.fsdp import FSDPOptimizerAdapter


def _create_fsdp_model(
    seed: int,
    device: torch.device,
    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )

    fsdp_model = FSDP(
        module=model,
        device_id=device,
    )
    FSDP.set_state_dict_type(fsdp_model, state_dict_type)
    return fsdp_model


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The test requires GPUs to run."
)
# pyre-fixme[56]: Pyre was not able to infer the type of the decorator
#  `pytest.mark.gpu_only`.
@pytest.mark.gpu_only
@pytest.mark.usefixtures("toggle_batching")
# Sharded state dict will test ShardedTensors, full tests Tensors
@pytest.mark.parametrize(
    "state_dict_type", [StateDictType.FULL_STATE_DICT, StateDictType.SHARDED_STATE_DICT]
)
@run_with_pet(nproc=2)
def test_model_and_optim_fsdp(tmp_path: Path, state_dict_type: StateDictType) -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    foo_fsdp = _create_fsdp_model(
        seed=42,
        device=device,
        state_dict_type=state_dict_type,
    )
    bar_fsdp = _create_fsdp_model(
        seed=777 + dist.get_rank(),
        device=device,
        state_dict_type=state_dict_type,
    )

    assert not check_state_dict_eq(foo_fsdp.state_dict(), bar_fsdp.state_dict())

    # Need to step and zero_grad in order to initialize all the optimizer parameters
    foo_optim = torch.optim.AdamW(foo_fsdp.parameters(), lr=0.01)
    foo_optim.step(closure=None)
    foo_optim.zero_grad(set_to_none=True)

    bar_optim = torch.optim.AdamW(bar_fsdp.parameters(), lr=0.02)
    bar_optim.step(closure=None)
    bar_optim.zero_grad(set_to_none=True)

    # pyre-fixme[6]: For 1st argument expected `FullyShardedDataParallel` but got
    #  `Module`.
    foo_fsdp_optim = FSDPOptimizerAdapter(foo_fsdp, foo_optim)
    # pyre-fixme[6]: For 1st argument expected `FullyShardedDataParallel` but got
    #  `Module`.
    bar_fsdp_optim = FSDPOptimizerAdapter(bar_fsdp, bar_optim)

    assert not check_state_dict_eq(
        foo_fsdp_optim.state_dict(), bar_fsdp_optim.state_dict()
    )

    foo_app_state = {"foo": foo_fsdp, "optim": foo_fsdp_optim}

    snapshot = Snapshot.take(str(tmp_path), foo_app_state)
    snapshot.restore({"foo": bar_fsdp, "optim": bar_fsdp_optim})

    assert check_state_dict_eq(foo_fsdp_optim.state_dict(), bar_fsdp_optim.state_dict())
    assert check_state_dict_eq(foo_fsdp.state_dict(), bar_fsdp.state_dict())
