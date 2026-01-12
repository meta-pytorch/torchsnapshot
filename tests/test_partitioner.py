#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torchsnapshot.batcher import batch_write_requests
from torchsnapshot.io_preparer import (
    ChunkedTensorIOPreparer,
    get_storage_path,
    prepare_read,
    prepare_write,
)
from torchsnapshot.io_types import ReadIO, WriteIO, WriteReq
from torchsnapshot.manifest import Entry
from torchsnapshot.partitioner import (
    consolidate_replicated_entries_dist,
    partition_write_reqs,
)
from torchsnapshot.pg_wrapper import PGWrapper
from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
)
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import (
    _tensor_test_case,
    rand_tensor,
    run_with_pet_async,
    tensor_eq,
)

WORLD_SIZE: int = 4


def _chunked_tensor_test_case(
    dtype: torch.dtype,
    shape: List[int],
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[torch.Tensor, Entry, List[WriteReq]]:
    tensor = rand_tensor(shape, dtype=dtype)
    tensor_sz_bytes = tensor.nelement() * tensor.element_size()
    chunking_instruction = ChunkedTensorIOPreparer.chunk_tensor(
        tensor=tensor, chunk_sz_bytes=tensor_sz_bytes // WORLD_SIZE
    )
    entry, wrs = ChunkedTensorIOPreparer.prepare_write(
        storage_path=get_storage_path(
            obj=tensor, logical_path=logical_path, rank=rank, replicated=replicated
        ),
        tensor=tensor,
        chunking_instruction=chunking_instruction,
    )
    return tensor, entry, wrs


def _sharded_tensor_test_case(
    dtype: torch.dtype,
    shape: List[int],
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[ShardedTensor, Entry, List[WriteReq]]:
    # pyre-ignore
    spec = ChunkShardingSpec(
        dim=0, placements=[f"rank:{rank}/cpu" for rank in range(dist.get_world_size())]
    )
    tensor = sharded_tensor.empty(spec, shape)
    for shard in tensor.local_shards():
        shard.tensor.random_()
    entry, wrs = prepare_write(
        obj=tensor, logical_path=logical_path, rank=rank, replicated=replicated
    )
    return tensor, entry, wrs


@pytest.mark.parametrize("dtype", ALL_SUPPORTED_DTYPES)
@pytest.mark.parametrize("enable_batcher", [True, False])
@run_with_pet_async(nproc=WORLD_SIZE)
async def test_partitioner(  # noqa
    dtype: torch.dtype, enable_batcher: bool, tmp_path: Path
) -> None:
    """
    Verify the behavior of the partitioner by:

    - Write various types of objects with the partitioner enabled:
        - Replicated, chunked tensor
        - Replicated, unchunked tensor
        - Non-replicated, chunked tensor
        - Non-replicated, unchunked tensor
        - Sharded tensor
    - Optionally enable the batcher
    - Read the written objects and compare with the originals
    """
    dist.init_process_group(backend="gloo")

    tensors = []
    entries = {}
    write_reqs = defaultdict(list)

    # Use the same seed to simulate replicated-ness
    torch.manual_seed(42)

    # Replicated, chunked tensor
    for idx in range(10):
        logical_path = f"replicated_chunked_{idx}"
        tensor, entry, wrs = _chunked_tensor_test_case(
            dtype=dtype,
            shape=[64, 64],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=True,
        )
        tensors.append(tensor)
        entries[logical_path] = entry
        write_reqs[logical_path].extend(wrs)

    # Replicated, unchunked tensor
    for idx in range(10):
        logical_path = f"replicated_unchunked_{idx}"
        tensor, entry, wrs = _tensor_test_case(
            dtype=dtype,
            shape=[64, 64],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=True,
        )
        tensors.append(tensor)
        entries[logical_path] = entry
        write_reqs[logical_path].extend(wrs)

    # Use different seed to initialize non-replicated test cases
    torch.manual_seed(777 + dist.get_rank())

    # Non-replicated, chunked tensor
    for idx in range(10):
        logical_path = f"nonreplicated_chunked_{idx}"
        tensor, entry, wrs = _chunked_tensor_test_case(
            dtype=dtype,
            shape=[64, 64],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=False,
        )
        tensors.append(tensor)
        entries[logical_path] = entry
        write_reqs[logical_path].extend(wrs)

    # Non-replicated, unchunked tensor
    for idx in range(10):
        logical_path = f"nonreplicated_unchunked_{idx}"
        tensor, entry, wrs = _tensor_test_case(
            dtype=dtype,
            shape=[64, 64],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=False,
        )
        tensors.append(tensor)
        entries[logical_path] = entry
        write_reqs[logical_path].extend(wrs)

    # Sharded tensors
    for idx in range(10):
        logical_path = f"sharded_{idx}"
        tensor, entry, wrs = _sharded_tensor_test_case(
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
    plugin = FSStoragePlugin(root=str(tmp_path))
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
        if type(tensor) == torch.Tensor:
            dst_tensors.append(rand_tensor(tuple(tensor.shape), dtype=dtype))
        elif type(tensor) == ShardedTensor:
            dst_tensors.append(sharded_tensor.empty(tensor.sharding_spec(), [64, 64]))
        else:
            raise AssertionError(f"Unexpected tensor type {type(tensor)}")

    for logical_path, tensor, dst_tensor in zip(entries.keys(), tensors, dst_tensors):
        assert not tensor_eq(tensor, dst_tensor)

        entry = partitioned_entries[logical_path]
        rrs, _ = prepare_read(entry, obj_out=dst_tensor)
        for rr in rrs:
            read_io = ReadIO(path=rr.path, byte_range=rr.byte_range)
            await plugin.read(read_io)
            await rr.buffer_consumer.consume_buffer(read_io.buf.getvalue())

        assert tensor_eq(tensor, dst_tensor)

    await plugin.close()
