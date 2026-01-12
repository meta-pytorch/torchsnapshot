#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[21, 56]: ignore pytest undefine import and invalid decoration
import copy
import random
import sys
import uuid
from typing import cast, Generator, List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate
from torchsnapshot.batcher import (
    batch_read_requests,
    batch_write_requests,
    BatchedBufferStager,
    GPUBatchedBufferStager,
    is_batchable,
)
from torchsnapshot.io_preparer import (
    ChunkedTensorIOPreparer,
    ObjectIOPreparer,
    prepare_read,
    ShardedTensorIOPreparer,
    TensorIOPreparer,
)
from torchsnapshot.io_preparers.dtensor import DTensorIOPreparer
from torchsnapshot.io_types import WriteReq
from torchsnapshot.manifest import Entry
from torchsnapshot.serialization import ALL_SUPPORTED_DTYPES
from torchsnapshot.test_utils import rand_tensor, tensor_eq, tensor_local_sz_bytes

NUM_TENSORS = 50
TENSOR_SHAPE = (64, 64)


@pytest.fixture
def use_gpu() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def dummy_pg(use_gpu: bool) -> Generator[None, None, None]:
    """
    Fixture for initializing a single process pg.
    """
    backend = "cpu:gloo,cuda:nccl" if use_gpu else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"file:///tmp/{uuid.uuid4()}",
        rank=0,
        world_size=1,
    )
    if use_gpu:
        torch.cuda.set_device(torch.device("cuda:0"))
    yield
    dist.destroy_process_group()


TestCase = Tuple[List[torch.Tensor], List[Entry], List[WriteReq], List[torch.Tensor]]


def _sharded_tensor_to_gpu(
    tensor: sharded_tensor.ShardedTensor,
) -> sharded_tensor.ShardedTensor:
    # TODO: this is available as ShardedTensor.to() in PyTorch 1.13.
    # Remove this once we drop PyTorch 1.12 support.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    shards: List[sharded_tensor.Shard] = []
    for shard in tensor.local_shards():
        new_tensor = shard.tensor.to(
            device=device,
        )
        metadata = copy.deepcopy(shard.metadata)
        # pyre-ignore
        metadata.placement._device = device
        shards.append(sharded_tensor.Shard(new_tensor, metadata))

    metadata = copy.deepcopy(tensor.metadata())
    for meta in metadata.shards_metadata:
        meta.placement._device = device

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        shards,
        sharded_tensor_metadata=metadata,
        process_group=tensor._process_group,
    )


@pytest.fixture
def sharded_tensor_test_cases(
    use_gpu: bool,
) -> TestCase:
    """
    Fixture for preparing sharded tensor test cases.

    Returns:
        - src tensors
        - Entries produced by ShardedTensorIOPreparer
        - Write requests produced by ShardedTensorIOPreparer
        - dst tensors whose values are different from the src tensors
    """
    spec = ChunkShardingSpec(  # pyre-ignore
        dim=0,
        placements=[
            "rank:0/cpu",
        ]
        * 4,
    )
    srcs = [sharded_tensor.empty(spec, TENSOR_SHAPE) for _ in range(NUM_TENSORS)]
    dsts = [sharded_tensor.empty(spec, TENSOR_SHAPE) for _ in range(NUM_TENSORS)]
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        for shard in src.local_shards() + dst.local_shards():
            shard.tensor.random_()
        if use_gpu and idx % 2 == 0:
            srcs[idx] = _sharded_tensor_to_gpu(src)
            dsts[idx] = _sharded_tensor_to_gpu(dst)

    entries = []
    write_reqs = []
    for idx, tensor in enumerate(srcs):
        entry, wrs = ShardedTensorIOPreparer.prepare_write(
            storage_path=f"sharded_tensor_{idx}", obj=tensor
        )
        entries.append(entry)
        write_reqs.extend(wrs)
    return (
        cast(List[torch.Tensor], srcs),
        entries,
        write_reqs,
        cast(List[torch.Tensor], dsts),
    )


@pytest.fixture
def dtensor_test_cases(
    use_gpu: bool,
) -> TestCase:
    """
    Fixture for preparing DTensor test cases.

    Returns:
        - src tensors
        - Entries produced by DTensorIOPreparer
        - Write requests produced by DTensorIOPreparer
        - dst tensors whose values are different from the src tensors
    """

    placements = [Replicate()]

    srcs = []
    dsts = []
    for idx in range(NUM_TENSORS):
        mesh = (
            DeviceMesh("cuda", mesh=[0])
            if use_gpu and idx % 2 == 0
            else DeviceMesh("cpu", mesh=[0])
        )
        srcs.append(
            distribute_tensor(
                tensor=torch.rand(*TENSOR_SHAPE),
                device_mesh=mesh,
                placements=placements,
            )
        )
        dsts.append(
            distribute_tensor(
                tensor=torch.rand(*TENSOR_SHAPE),
                device_mesh=mesh,
                placements=placements,
            )
        )

    entries = []
    write_reqs = []
    for idx, tensor in enumerate(srcs):
        entry, wrs = DTensorIOPreparer.prepare_write(
            storage_path=f"dtensor_{idx}", obj=tensor
        )
        entries.append(entry)
        write_reqs.extend(wrs)
    return (
        cast(List[torch.Tensor], srcs),
        entries,
        write_reqs,
        cast(List[torch.Tensor], dsts),
    )


@pytest.fixture
def tensor_test_cases(
    dtype: torch.dtype, enable_chunking: bool, use_gpu: bool
) -> TestCase:
    """
    Fixture for preparing tensor test cases.

    Returns:
        - src tensors
        - Entries produced by TensorIOPreparer/ChunkedTensorIOPreparer
        - Write requests produced by TensorIOPreparer/ChunkedTensorIOPreparer
        - dst tensors whose values are different from the src tensors
    """
    if dtype is not None:
        dtypes = [dtype] * NUM_TENSORS
    else:
        dtypes = [
            ALL_SUPPORTED_DTYPES[i % len(ALL_SUPPORTED_DTYPES)]
            for i in range(NUM_TENSORS)
        ]
    srcs = [rand_tensor(TENSOR_SHAPE, dtype=dtypes[i]) for i in range(NUM_TENSORS)]
    dsts = [rand_tensor(TENSOR_SHAPE, dtype=dtypes[i]) for i in range(NUM_TENSORS)]
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        if use_gpu and idx % 2 == 0:
            srcs[idx] = src.to(torch.cuda.current_device())
            dsts[idx] = dst.to(torch.cuda.current_device())

    entries = []
    write_reqs = []
    for idx, tensor in enumerate(srcs):
        if enable_chunking:
            tensor_sz_bytes = tensor.nelement() * tensor.element_size()
            chunking_instruction = ChunkedTensorIOPreparer.chunk_tensor(
                tensor=tensor, chunk_sz_bytes=tensor_sz_bytes // 4
            )
            entry, wrs = ChunkedTensorIOPreparer.prepare_write(
                storage_path=f"{idx}",
                tensor=tensor,
                chunking_instruction=chunking_instruction,
            )
        else:
            entry, wrs = TensorIOPreparer.prepare_write(
                storage_path=f"{idx}", tensor=tensor
            )
        entries.append(entry)
        write_reqs.extend(wrs)
    return srcs, entries, write_reqs, dsts


@pytest.fixture
def slab_size_bytes(
    tensor_test_cases: TestCase,
    sharded_tensor_test_cases: TestCase,
    dtensor_test_cases: TestCase,
) -> int:
    """
    Fixture for determining the slab size.

    Returns:
        Slab size inferred from tensor_test_cases and sharded_tensor_test_cases
        that makes sure multiple slabs are needed.
    """
    total_tensor_sz_bytes = 0
    for tensor in (
        tensor_test_cases[0] + sharded_tensor_test_cases[0] + dtensor_test_cases[0]
    ):
        total_tensor_sz_bytes += tensor_local_sz_bytes(tensor=tensor)
    # We want to test multiple slabs
    return total_tensor_sz_bytes // 8


@pytest.fixture
def read_chunk_size_bytes(
    tensor_test_cases: TestCase,
    sharded_tensor_test_cases: TestCase,
    dtensor_test_cases: TestCase,
) -> int:
    """
    Fixture for determining the read chunk size.

    Returns:
        Read chunk size inferred from tensor_test_cases and
        sharded_tensor_test_cases that makes sure each tensor is read as
        multiple chunks.
    """
    min_tensor_sz_bytes = sys.maxsize
    for tensor in (
        tensor_test_cases[0] + sharded_tensor_test_cases[0] + dtensor_test_cases[0]
    ):
        min_tensor_sz_bytes = min(
            min_tensor_sz_bytes, tensor_local_sz_bytes(tensor=tensor)
        )
    # We want to test multiple slabs
    return min_tensor_sz_bytes // 4


@pytest.mark.asyncio
@pytest.mark.cpu_and_gpu
@pytest.mark.usefixtures("dummy_pg")
@pytest.mark.parametrize("dtype", ALL_SUPPORTED_DTYPES)
@pytest.mark.parametrize("enable_chunking", [True, False])
@pytest.mark.parametrize("enable_batched_read", [True, False])
@pytest.mark.parametrize("enable_chunked_read", [True, False])
async def test_batcher(
    tensor_test_cases: TestCase,
    sharded_tensor_test_cases: TestCase,
    dtensor_test_cases: TestCase,
    slab_size_bytes: int,
    read_chunk_size_bytes: Optional[int],
    enable_batched_read: bool,
    enable_chunked_read: bool,
    use_gpu: bool,
) -> None:
    """
    Verify the behavior of the batcher.
    """
    src_tensors, entries, write_reqs, dst_tensors = (
        # pyre-fixme[6]: For 1st argument expected `List[Tensor]` but got
        #  `Union[List[Entry], List[WriteReq], List[Tensor]]`.
        # pyre-fixme[58]: `+` is not supported for operand types
        #  `List[torch._tensor.Tensor]` and `Union[List[Entry], List[WriteReq],
        #  List[torch._tensor.Tensor]]`.
        a + b + c
        for a, b, c in zip(
            tensor_test_cases, sharded_tensor_test_cases, dtensor_test_cases
        )
    )

    # Mix in some object write requests
    for idx in range(NUM_TENSORS):
        entry, wrs = ObjectIOPreparer.prepare_write(
            storage_path=f"object_{idx}", obj=object()
        )
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `ObjectEntry`.
        entries.append(entry)
        # pyre-fixme[6]: For 1st argument expected `Iterable[Tensor]` but got
        #  `List[WriteReq]`.
        write_reqs.extend(wrs)

    # The order of the write request shouldn't matter so we shuffle it
    random.shuffle(write_reqs)

    # The src tensors and dst tensors should have different value now
    for src, dst in zip(src_tensors, dst_tensors):
        assert not tensor_eq(src, dst)

    # Expect the batcher to fail if it did not receive all affected entries
    # NOTE: don't shuffle the entries as they have to be aligned with
    # src_tensors and dst_tensors.
    # pyre-fixme[6]: For 1st argument expected `BufferStager` but got `Tensor`.
    if is_batchable(entries[0]):
        with pytest.raises(RuntimeError):
            batch_write_requests(
                # pyre-fixme[6]: For 1st argument expected `List[Entry]` but got
                #  `List[Tensor]`.
                entries=entries[1:],
                # pyre-fixme[6]: For 2nd argument expected `List[WriteReq]` but got
                #  `List[Tensor]`.
                write_reqs=write_reqs,
            )

    # Batch the write requests
    entries, batched_write_reqs = batch_write_requests(
        # pyre-fixme[6]: For 1st argument expected `List[Entry]` but got `List[Tensor]`.
        entries=entries,
        # pyre-fixme[6]: For 2nd argument expected `List[WriteReq]` but got
        #  `List[Tensor]`.
        write_reqs=copy.deepcopy(write_reqs),
    )
    assert len(batched_write_reqs) < len(write_reqs)
    write_reqs = batched_write_reqs

    buffer_stager_types = {type(wr.buffer_stager) for wr in write_reqs}

    # In this test setup, sharded tensor shard are alway batchable
    assert BatchedBufferStager in buffer_stager_types

    # When use_gpu == True, verify that BatchedBufferStager is used
    if use_gpu:
        assert GPUBatchedBufferStager in buffer_stager_types

    # Prepare read requests for the dst tensors
    read_reqs = []
    for entry, obj_out in zip(entries, dst_tensors + [None] * NUM_TENSORS):
        rrs, _ = prepare_read(
            entry=entry,
            obj_out=obj_out,
            buffer_size_limit_bytes=(
                read_chunk_size_bytes if enable_chunked_read else None
            ),
        )
        read_reqs.extend(rrs)

    # Read should work regardless of whether write was batched or not
    if enable_batched_read:
        batched_read_reqs = batch_read_requests(read_reqs=read_reqs)
        assert len(batched_read_reqs) < len(read_reqs)
        read_reqs = batched_read_reqs

    # Fulfill read requests with write requests
    location_to_buf = {
        wr.path: await wr.buffer_stager.stage_buffer(executor=None) for wr in write_reqs
    }
    for rr in read_reqs:
        buf = location_to_buf[rr.path]
        byte_range = rr.byte_range
        if byte_range is not None:
            buf = buf[byte_range[0] : byte_range[1]]
        else:
            buf = buf
        await rr.buffer_consumer.consume_buffer(buf, executor=None)

    # The src tensors and dst tensors should have the same values now
    for src, dst in zip(src_tensors, dst_tensors):
        assert tensor_eq(src, dst)
