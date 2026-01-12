#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Generator, List, Set, Tuple

import pytest
import torch
import torch.distributed as dist
from _pytest.fixtures import SubRequest  # @manual
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
)
from torchsnapshot.io_preparer import (
    ShardedTensorIOPreparer,
    TensorBufferConsumer,
    TensorIOPreparer,
)
from torchsnapshot.knobs import override_max_shard_size_bytes
from torchsnapshot.test_utils import run_with_pet_async, tensor_eq

WORLD_SIZE = 4


@pytest.fixture(params=[("chunk", 0), ("chunk", 1), ("enumerate", None)])
def sharding_spec(shape: Tuple[int, int], request: SubRequest) -> ShardingSpec:
    """
    Fixture for generating different sharding specs given the global shape.
    """
    sharding_type, dim = request.param
    if sharding_type == "chunk":
        # pyre-ignore
        return ChunkShardingSpec(
            dim=dim, placements=[f"rank:{rank}/cpu" for rank in range(WORLD_SIZE)]
        )

    assert sharding_type == "enumerate"
    #     b    d
    #   +----+---+
    # a |    |   |
    #   +----+---+
    # c |    |   |
    #   +----+---+
    a = shape[0] // 2
    b = shape[1] // 2
    c = shape[0] - shape[0] // 2
    d = shape[1] - shape[1] // 2
    return EnumerableShardingSpec(
        [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[a, b],
                placement="rank:0/cpu",
            ),
            ShardMetadata(
                shard_offsets=[0, b],
                shard_sizes=[a, d],
                placement="rank:1/cpu",
            ),
            ShardMetadata(
                shard_offsets=[a, 0],
                shard_sizes=[c, b],
                placement="rank:2/cpu",
            ),
            ShardMetadata(
                shard_offsets=[a, b],
                shard_sizes=[c, d],
                placement="rank:3/cpu",
            ),
        ]
    )


@pytest.fixture
def enable_subdivision(
    shape: Tuple[int, int], request: SubRequest
) -> Generator[bool, None, None]:
    if not request.param:
        yield False
    else:
        max_shard_size_bytes = shape[0] * shape[1] * 4 // WORLD_SIZE // WORLD_SIZE
        with override_max_shard_size_bytes(max_shard_size_bytes):
            yield True


@pytest.mark.parametrize("shape", [(128, 128), (127, 129)])
@pytest.mark.parametrize("enable_subdivision", [True, False], indirect=True)
@run_with_pet_async(nproc=WORLD_SIZE)
async def test_sharded_tensor_io_preparer(
    shape: Tuple[int, int], sharding_spec: ShardingSpec, enable_subdivision: bool
) -> None:
    """
    Verify the basic behavior of ShardedTensorIOPreparer.
    """
    dist.init_process_group("gloo")
    src = sharded_tensor.empty(sharding_spec, *shape)
    dst = sharded_tensor.empty(sharding_spec, *shape)
    for st in [src, dst]:
        for shard in st.local_shards():
            shard.tensor.random_()

    entry, write_reqs = ShardedTensorIOPreparer.prepare_write(
        storage_path="/foo", obj=src
    )
    assert len(entry.shards) == len(write_reqs)

    if enable_subdivision:
        # When subdivision is enabled, we have more write requests than local
        # shards, and each write request corresponds to a subview of a local
        # shard.
        assert len(src.local_shards()) < len(write_reqs)
        for shard_entry, shard in zip(entry.shards, src.local_shards()):
            assert (
                TensorIOPreparer.get_tensor_size_from_entry(shard_entry.tensor)
                < shard.tensor.storage().size() * shard.tensor.element_size()
            )
    else:
        assert len(src.local_shards()) == len(write_reqs)
        for shard_entry, shard in zip(entry.shards, src.local_shards()):
            assert (
                TensorIOPreparer.get_tensor_size_from_entry(shard_entry.tensor)
                == shard.tensor.storage().size() * shard.tensor.element_size()
            )

    # Verify no overlapping locations among local shards
    locations = set()
    for shard, wr in zip(entry.shards, write_reqs):
        assert shard.tensor.location == wr.path
        locations.add(wr.path)

    assert len(locations) == len(write_reqs)

    # Verify no overlapping locations among global shards
    # pyre-ignore
    obj_list: List[Set[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(obj_list, locations)
    all_locations = [location for ls in obj_list for location in ls]
    assert len(set(all_locations)) == len(all_locations)

    location_to_buf = {
        wr.path: bytes(await wr.buffer_stager.stage_buffer()) for wr in write_reqs
    }

    # Verify that the size of the storage of a persisted shard matches with the
    # shape of the shard (as opposed to the size of the storage of the shard).
    for idx, buf in enumerate(location_to_buf.values()):
        deserialized = TensorBufferConsumer.deserialize_tensor(
            buf=buf, entry=entry.shards[idx].tensor
        )
        assert (
            deserialized.storage().size() * deserialized.element_size()
            == TensorIOPreparer.get_tensor_size_from_entry(entry.shards[idx].tensor)
        )

    # Consume the buffers with dst and verify that src == dst
    assert not tensor_eq(src, dst)
    read_reqs, _ = ShardedTensorIOPreparer.prepare_read(entry=entry, obj_out=dst)
    for rr in read_reqs:
        await rr.buffer_consumer.consume_buffer(buf=location_to_buf[rr.path])
    assert tensor_eq(src, dst)


class ShardedTensorIOPreparerTest(unittest.TestCase):
    def _verify_subdivided_shards(
        self,
        subdivided: List[Tuple[torch.Tensor, List[int], List[int]]],
        dim: int,
        expected_num_sub_shards: int,
        expected_combined: torch.Tensor,
        expected_offsets: List[int],
        expected_sizes: List[int],
    ) -> None:
        """
        Combine the tensor, offsets, sizes of a subdivision result and verify
        the correctness along the way.
        """
        _, offsets, sizes = copy.deepcopy(subdivided[0])
        for _, sub_offsets, sub_sizes in subdivided[1:]:
            self.assertEqual(len(offsets), len(sub_offsets))
            self.assertEqual(len(sizes), len(sub_sizes))
            for i in range(len(offsets)):
                if i != dim:
                    self.assertEqual(sub_offsets[i], offsets[i])
                    self.assertEqual(sub_sizes[i], sizes[i])
            self.assertEqual(sub_offsets[dim], offsets[dim] + sizes[dim])
            sizes[dim] += sub_sizes[dim]

        sub_views = [sub_view for sub_view, _, _ in subdivided]
        combined = torch.concat(sub_views, dim)
        self.assertEqual(len(subdivided), expected_num_sub_shards)
        self.assertTrue(torch.allclose(combined, expected_combined))
        self.assertEqual(offsets, expected_offsets)
        self.assertEqual(sizes, expected_sizes)

    def test_subdivide_shard(self) -> None:
        tensor = torch.randn(256, 256)
        # max_shard_sz_bytes is smaller than the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=77,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=256,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is between 1x and 2x the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=1999,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=256,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than 2x of the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=4001,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=86,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than 2x of the size of a slice
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[0, 512],
            sizes=[256, 256],
            dim=1,
            max_shard_sz_bytes=4001,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=1,
            expected_num_sub_shards=86,
            expected_combined=tensor,
            expected_offsets=[0, 512],
            expected_sizes=[256, 256],
        )

        # max_shard_sz_bytes is greater than the shard size
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=tensor,
            offsets=[512, 0],
            sizes=[256, 256],
            dim=0,
            max_shard_sz_bytes=300000,
        )
        self._verify_subdivided_shards(
            subdivided=subdivided,
            dim=0,
            expected_num_sub_shards=1,
            expected_combined=tensor,
            expected_offsets=[512, 0],
            expected_sizes=[256, 256],
        )
