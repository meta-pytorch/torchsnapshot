#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import reduce
from operator import mul
from pathlib import Path
from typing import Generator, List

import pytest
import torch
import torch.distributed as dist
from _pytest.fixtures import SubRequest  # @manual
from torch.nn.parallel import DistributedDataParallel
from torchsnapshot import Snapshot
from torchsnapshot.knobs import override_max_chunk_size_bytes
from torchsnapshot.test_utils import check_state_dict_eq, run_with_pet
from torchsnapshot.tricks.ddp import (
    DDP_STATE_DICT_PREFIX,
    DistributedDataParallelAdapter,
)

WORLD_SIZE: int = 4


@pytest.fixture
def layer_shapes() -> List[List[int]]:
    return [[128, 64], [64, 32], [32, 16]]


@pytest.fixture(params=["chunking_on", "chunking_off"])
def toggle_chunking(
    layer_shapes: List[List[int]], request: SubRequest
) -> Generator[None, None, None]:
    if not request.param == "chunking_off":
        yield
        return
    max_chunk_size_bytes = min(reduce(mul, ls) for ls in layer_shapes) * 4
    with override_max_chunk_size_bytes(max_chunk_size_bytes):
        yield


@pytest.mark.usefixtures("toggle_batching", "toggle_chunking")
@run_with_pet(nproc=WORLD_SIZE)
def test_ddp_simple(layer_shapes: List[List[int]], tmp_path: Path) -> None:
    """
    Randomly initialize two DDP-wrapped models. Take the snapshot of one
    model and use the snapshot to restore the other model. Verify that the
    two model's state dicts are equal after restoration.
    """
    dist.init_process_group(backend="gloo")

    src = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    dst = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    src_ddp = DistributedDataParallel(src)
    dst_ddp = DistributedDataParallel(dst)
    src_optim = torch.optim.Adagrad(src_ddp.parameters(), lr=0.01)
    dst_optim = torch.optim.Adagrad(dst_ddp.parameters(), lr=0.001)
    assert not check_state_dict_eq(dst_ddp.state_dict(), src_ddp.state_dict())
    assert not check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())

    snapshot = Snapshot.take(
        path=str(tmp_path),
        app_state={"ddp": src_ddp, "optim": src_optim},
        replicated=["optim/**"],
    )
    snapshot.restore(app_state={"ddp": dst_ddp, "optim": dst_optim})
    assert check_state_dict_eq(dst_ddp.state_dict(), src_ddp.state_dict())
    assert check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())


@pytest.mark.usefixtures("toggle_batching", "toggle_chunking")
@run_with_pet(nproc=WORLD_SIZE)
def test_ddp_upscale(layer_shapes: List[List[int]], tmp_path: Path) -> None:
    dist.init_process_group(backend="gloo")
    sub_pg = dist.new_group(backend="gloo", ranks=list(range(WORLD_SIZE - 1)))

    # Initialize src with the same seed across ranks
    torch.manual_seed(42)
    src = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    src_optim = torch.optim.Adagrad(src.parameters(), lr=0.01)

    # Initialize dst with the different seeds across ranks
    torch.manual_seed(777 + dist.get_rank())
    dst = torch.nn.Sequential(
        # pyre-ignore
        *(torch.nn.Linear(*layer_shape) for layer_shape in layer_shapes),
    )
    dst_optim = torch.optim.Adagrad(dst.parameters(), lr=0.001)
    assert not check_state_dict_eq(dst.state_dict(), src.state_dict())
    assert not check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())

    if dist.get_rank() < WORLD_SIZE - 1:
        # Initialize src_ddp among sub_pg
        src_ddp = DistributedDataParallel(src, process_group=sub_pg)

        # Take a snapshot among sub_pg
        Snapshot.take(
            path=str(tmp_path),
            app_state={"ddp": src_ddp, "optim": src_optim},
            replicated=["optim/**"],
            pg=sub_pg,
        )

        dist.barrier()

        # Initialize dst_ddp among the global pg
        dst_ddp = DistributedDataParallel(dst)

        snapshot = Snapshot(path=str(tmp_path))
        snapshot.restore(app_state={"ddp": dst_ddp, "optim": dst_optim})
        assert check_state_dict_eq(dst_ddp.state_dict(), src_ddp.state_dict())
        assert check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())
    else:
        dist.barrier()

        # Initialize dst_ddp among the global pg
        dst_ddp = DistributedDataParallel(dst)

        snapshot = Snapshot(path=str(tmp_path))
        snapshot.restore(app_state={"ddp": dst_ddp, "optim": dst_optim})
        assert check_state_dict_eq(dst.state_dict(), src.state_dict())
        assert check_state_dict_eq(dst_optim.state_dict(), src_optim.state_dict())


@run_with_pet(nproc=WORLD_SIZE)
def test_ddp_save_load_non_ddp(tmp_path: Path) -> None:
    """
    Randomly initialize one DDP-wrapped model, and one non-DDP-wrapped model. Take the snapshot of one
    model and use the snapshot to restore the other model using the DDPWrapper helper class. Verify that the
    two model's state dicts are equal after restoration.
    """
    dist.init_process_group(backend="gloo")

    # Initialize src with the same seed across ranks
    torch.manual_seed(42)
    src = torch.nn.Linear(1, 1)
    src_ddp = DistributedDataParallel(src)

    # Initialize dst with the different seeds across ranks
    torch.manual_seed(777 + dist.get_rank())
    dst = torch.nn.Linear(1, 1)

    assert not check_state_dict_eq(dst.state_dict(), src_ddp.state_dict())

    snapshot = Snapshot.take(
        path=str(tmp_path),
        app_state={"ddp": src_ddp},
    )

    dst_adaptor = DistributedDataParallelAdapter(dst)
    snapshot.restore(app_state={"ddp": dst_adaptor})
    restored_state_dict = dst_adaptor.module.state_dict()

    consumed_state_dict = src_ddp.state_dict()
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        consumed_state_dict, DDP_STATE_DICT_PREFIX
    )
    # The utility consume_prefix_in_state_dict_if_present re-inserts keys into the state dict
    # which changes the order they appear in the state dict, as it is an OrderedDict.
    # to test for equality, explicitly sort the state dicts by key before comparison
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Any, typing.Any]` but
    #  got `List[str]`.
    # pyre-fixme[6]: For 2nd argument expected `Dict[typing.Any, typing.Any]` but
    #  got `List[str]`.
    assert check_state_dict_eq(sorted(restored_state_dict), sorted(consumed_state_dict))
