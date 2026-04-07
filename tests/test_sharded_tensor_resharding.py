#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[21, 56]: ignore pytest undefine import and invalid decoration
import itertools
import uuid
from typing import cast, Generator, List

import pytest
import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
)
from torchsnapshot.io_preparer import ShardedTensorIOPreparer


@pytest.fixture
def dummy_pg() -> Generator[None, None, None]:
    dist.init_process_group(
        backend="gloo", init_method=f"file:///tmp/{uuid.uuid4()}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


def sharding_specs() -> List[ShardingSpec]:
    specs: List[ShardingSpec] = [
        ChunkShardingSpec(
            dim=dim,
            placements=[
                "rank:0/cpu",
            ]
            * num_shards,
        )
        for dim, num_shards in itertools.product([0, 1], [3, 5])
    ]
    specs.append(
        EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[64, 64],
                    placement="rank:0/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[0, 64],
                    shard_sizes=[64, 64],
                    placement="rank:0/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[64, 0],
                    shard_sizes=[64, 64],
                    placement="rank:0/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[64, 64],
                    shard_sizes=[64, 64],
                    placement="rank:0/cpu",
                ),
            ]
        )
    )
    return specs


@pytest.mark.asyncio
@pytest.mark.parametrize("src_spec", sharding_specs())
@pytest.mark.parametrize("dst_spec", sharding_specs())
async def test_sharded_tensor_resharding(
    src_spec: ShardingSpec, dst_spec: ShardingSpec, dummy_pg: None
) -> None:
    # Randomly initialize two sharded tensors
    src = sharded_tensor.empty(src_spec, 128, 128)
    dst = sharded_tensor.empty(dst_spec, 128, 128)
    for st in [src, dst]:
        for shard in st.local_shards():
            shard.tensor.random_()

    # Verify that they are not the same
    src_gathered = torch.empty(128, 128)
    dst_gathered = torch.empty(128, 128)
    src.gather(out=src_gathered)
    dst.gather(out=dst_gathered)
    assert not torch.allclose(src_gathered, dst_gathered)

    entry, write_reqs = ShardedTensorIOPreparer.prepare_write(
        storage_path="src", obj=src
    )
    read_reqs, _ = ShardedTensorIOPreparer.prepare_read(entry=entry, obj_out=dst)

    # Fulfill the dst's read requests with src's write requests
    path_to_buf = {wr.path: await wr.buffer_stager.stage_buffer() for wr in write_reqs}
    for rr in read_reqs:
        await rr.buffer_consumer.consume_buffer(buf=cast(bytes, path_to_buf[rr.path]))

    src.gather(out=src_gathered)
    dst.gather(out=dst_gathered)
    assert torch.allclose(src_gathered, dst_gathered)
