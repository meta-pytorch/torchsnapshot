#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torchsnapshot.dtensor_utils import is_replicated_dtensor, is_sharded

WORLD_SIZE = 4


@instantiate_parametrized_tests
class TestDTensorUtils(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(WORLD_SIZE)
    # pyre-fixme[3]: Return type must be annotated.
    def test_is_sharded_is_replicated(self):
        mesh = DeviceMesh("cuda", mesh=[[0, 1], [2, 3]])
        placements = [Replicate(), Shard(0)]
        local_tensor = torch.rand((16, 16))
        dtensor = distribute_tensor(
            tensor=local_tensor, device_mesh=mesh, placements=placements
        )
        assert is_sharded(dtensor)
        assert is_replicated_dtensor(dtensor)

        placements = [Replicate(), Replicate()]
        dtensor = distribute_tensor(
            tensor=local_tensor, device_mesh=mesh, placements=placements
        )
        assert not is_sharded(dtensor)
        assert is_replicated_dtensor(dtensor)

        # pyre-ignore
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{rank}/cuda:{rank}" for rank in range(dist.get_world_size())
            ],
        )
        stensor = sharded_tensor.empty(spec, (16, 16))
        assert is_sharded(stensor)

        placements = [Shard(0), Shard(1)]
        dtensor = distribute_tensor(
            tensor=local_tensor, device_mesh=mesh, placements=placements
        )
        assert is_sharded(dtensor)
        assert not is_replicated_dtensor(dtensor)
