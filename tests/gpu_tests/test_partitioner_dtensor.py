#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import uuid
from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torchsnapshot.batcher import batch_write_requests
from torchsnapshot.io_preparer import prepare_read
from torchsnapshot.io_types import ReadIO, WriteIO
from torchsnapshot.partitioner import (
    consolidate_replicated_entries_dist,
    partition_write_reqs,
)
from torchsnapshot.pg_wrapper import PGWrapper
from torchsnapshot.serialization import (
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    NCCL_SUPPORTED_DTYPES,
)
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import _dtensor_test_case, rand_tensor, tensor_eq

WORLD_SIZE: int = 4


@instantiate_parametrized_tests
class TestPartitioner(DTensorTestBase):
    @parametrize("dtype", NCCL_SUPPORTED_DTYPES)
    @parametrize("enable_batcher", [True, False])
    @skip_if_lt_x_gpu(WORLD_SIZE)
    # pyre-fixme[56]: While applying decorator
    #  `torch.testing._internal.distributed._tensor.common_dtensor.with_comms`: For 1st
    #  argument expected `(object) -> object` but got `(self: TestPartitioner, dtype:
    #  dtype, enable_batcher: bool) -> Coroutine[typing.Any, typing.Any, None]`.
    @with_comms
    async def test_partitioner(
        self,
        dtype: torch.dtype,
        enable_batcher: bool,
    ) -> None:
        """
        Verify the behavior of the partitioner by:

        - Write DTensor objects with the partitioner enabled:
        - Optionally enable the batcher
        - Read the written objects and compare with the originals
        """

        tensors = []
        entries = {}
        write_reqs = defaultdict(list)

        # Use the same seed to simulate replicated-ness
        torch.manual_seed(42)

        # DTensors
        for idx in range(10):
            logical_path = f"replicated_sharded_{idx}"
            tensor, entry, wrs = _dtensor_test_case(
                dtype=dtype,
                shape=[64, 64],
                logical_path=logical_path,
                rank=dist.get_rank(),
                replicated=False,
            )
            tensors.append(tensor)
            entries[logical_path] = entry
            write_reqs[logical_path].extend(wrs)

        # Perform partition
        partitioned_entries, partitioned_write_reqs = partition_write_reqs(
            entries=entries, write_reqs=write_reqs, pg=PGWrapper(pg=None)
        )
        partitioned_write_reqs = [
            wr for wrs in partitioned_write_reqs.values() for wr in wrs
        ]

        # The partitioner should work with or without the batcher
        if enable_batcher:
            batched_entries, batched_write_reqs = batch_write_requests(
                entries=list(partitioned_entries.values()),
                write_reqs=partitioned_write_reqs,
            )
            # Make sure that batching happened
            if dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
                assert len(batched_write_reqs) < len(partitioned_write_reqs)
            partitioned_entries = dict(zip(partitioned_entries.keys(), batched_entries))
            partitioned_write_reqs = batched_write_reqs

        partitioned_entries = consolidate_replicated_entries_dist(
            partitioned_entries, pg=PGWrapper(pg=None), dedup=False
        )

        # Verify that all logical paths are still present
        for logical_path in entries.keys():
            assert logical_path in partitioned_entries

        # Gather locations to be written by all ranks
        locations = [wr.path for wr in partitioned_write_reqs]
        # pyre-ignore
        obj_list: List[List[str]] = [None] * dist.get_world_size()
        dist.all_gather_object(obj_list, locations)
        locations = {location for locations in obj_list for location in locations}

        # Verify there are no duplicate write requests
        assert len(locations) == len(set(locations))

        # Fulfill the write requests
        plugin = FSStoragePlugin(root=f"/tmp/{uuid.uuid4()}")
        for wr in partitioned_write_reqs:
            buf = await wr.buffer_stager.stage_buffer()
            write_io = WriteIO(path=wr.path, buf=buf)
            await plugin.write(write_io)

        # Wait for all ranks to finish writing before begin reading
        dist.barrier()

        # Verify the integrity of the writes by loading the persisted tensors and
        # comparing them with the original tensors.
        dst_tensors = []
        for tensor in tensors:
            if type(tensor) == DTensor:
                dst_tensors.append(
                    DTensor.from_local(
                        local_tensor=rand_tensor(tuple(tensor.shape), dtype=dtype),
                        device_mesh=tensor.device_mesh,
                        placements=tensor.placements,
                    )
                )
            else:
                raise AssertionError(f"Unexpected tensor type {type(tensor)}")

        for logical_path, tensor, dst_tensor in zip(
            entries.keys(), tensors, dst_tensors
        ):
            assert not tensor_eq(tensor, dst_tensor)

            entry = partitioned_entries[logical_path]
            rrs, _ = prepare_read(entry, obj_out=dst_tensor)
            for rr in rrs:
                read_io = ReadIO(path=rr.path, byte_range=rr.byte_range)
                await plugin.read(read_io)
                await rr.buffer_consumer.consume_buffer(read_io.buf.getvalue())

            assert tensor_eq(tensor, dst_tensor)

        await plugin.close()
