#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

# pyre-strict

import asyncio
import copy
import itertools
import logging
import math
from collections import defaultdict
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.distributed._shard.sharded_tensor import (
    Shard as ShardedTensorShard,
    ShardedTensor,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)
from torchsnapshot.io_preparers.tensor import (
    tensor_copy,
    TensorBufferConsumer,
    TensorIOPreparer,
)
from torchsnapshot.io_types import BufferConsumer, Future, ReadReq, WriteReq
from torchsnapshot.knobs import get_max_shard_size_bytes
from torchsnapshot.manifest import Shard, ShardedTensorEntry, TensorEntry
from torchsnapshot.serialization import Serializer, string_to_dtype

logger: logging.Logger = logging.getLogger(__name__)


class ShardedTensorIOPreparer:
    @staticmethod
    def subdivide_shard(
        shard: torch.Tensor,
        offsets: List[int],
        sizes: List[int],
        dim: int,
        max_shard_sz_bytes: int,
    ) -> List[Tuple[torch.Tensor, List[int], List[int]]]:
        """
        Subdivide the shard along the sharding dim.
        """
        if max_shard_sz_bytes <= 0:
            raise ValueError(
                f"max_shard_sz_bytes must be a positive integer (got {max_shard_sz_bytes})."
            )
        slice_sz = reduce(mul, sizes) // sizes[dim] * shard.element_size()
        chunk_length = max(math.floor(max_shard_sz_bytes / slice_sz), 1)
        n_chunks = math.ceil(sizes[dim] / chunk_length)

        subdivided = []
        for i in range(n_chunks):
            start = i * chunk_length
            length = min((i + 1) * chunk_length, sizes[dim]) - i * chunk_length

            sub_offsets = copy.deepcopy(offsets)
            sub_offsets[dim] += start
            sub_sizes = copy.deepcopy(sizes)
            sub_sizes[dim] = length
            sub_view = torch.narrow(shard, dim, start, length)
            subdivided.append((sub_view, sub_offsets, sub_sizes))
        return subdivided

    @staticmethod
    def _shards_get_overlap_region_wrt_saved_tensor(
        saved_shard: ShardMetadata, current_shard: ShardMetadata
    ) -> List[Tuple[int, int, int, int]]:
        """
        Return the overlapping region between saved_shard and current_shard.
        There returned list has the same number of elements as the tensor's dimension.
        For each element, we produce a tuple with the following contents:
            (dimension, `saved_shard` offset, `current_shard` offset, length)

        Offsets are relative to each shard.
        """
        # TODO: This is copied from
        # torch.distributed._shard.checkpoint.resharding._shards_get_overlap_region_wrt_saved_tensor
        # which is not in PyTorch 1.11 yet. Remove this method and import it
        # from PyTorch directly once we drop support for 1.11.
        narrows = []
        for dim, (
            saved_shard_offset,
            current_shard_offset,
            saved_shard_size,
            current_shard_size,
        ) in enumerate(
            zip(
                saved_shard.shard_offsets,
                current_shard.shard_offsets,
                saved_shard.shard_sizes,
                current_shard.shard_sizes,
            )
        ):
            min_range_end = min(
                saved_shard_offset + saved_shard_size,
                current_shard_offset + current_shard_size,
            )

            length = min_range_end - max(current_shard_offset, saved_shard_offset)

            if saved_shard_offset > current_shard_offset:
                offset_for_saved_tensor = 0
                offset_for_current_tensor = saved_shard_offset - current_shard_offset
            else:
                offset_for_saved_tensor = current_shard_offset - saved_shard_offset
                offset_for_current_tensor = 0

            narrows.append(
                (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
            )
        return narrows

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        obj: ShardedTensor,
        is_async_snapshot: bool = False,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[ShardedTensorEntry, List[WriteReq]]:
        shards = []
        write_reqs = []
        for shard in obj.local_shards():
            sharding_spec = obj.sharding_spec()
            if isinstance(sharding_spec, ChunkShardingSpec):
                sharding_dim = sharding_spec.dim
            else:
                sharding_dim = 0
            subdivided = cls.subdivide_shard(
                shard=shard.tensor,
                offsets=shard.metadata.shard_offsets,
                sizes=shard.metadata.shard_sizes,
                dim=sharding_dim,
                max_shard_sz_bytes=get_max_shard_size_bytes(),
            )

            for tensor, offsets, sizes in subdivided:
                suffix = "_".join(str(i) for i in offsets)
                entry, tensor_write_reqs = TensorIOPreparer.prepare_write(
                    storage_path=f"{storage_path}_{suffix}",
                    tensor=tensor,
                    is_async_snapshot=is_async_snapshot,
                    _tensor_prepare_func=_tensor_prepare_func,
                )
                write_reqs += tensor_write_reqs

                shards.append(
                    Shard(
                        offsets=offsets,
                        sizes=sizes,
                        tensor=entry,
                    )
                )
        return ShardedTensorEntry(shards=shards), write_reqs

    @staticmethod
    def _get_global_shape(entry: ShardedTensorEntry) -> List[int]:
        # TODO: the global dtype and shape should be tracked in the metadata
        global_shape = [0] * len(entry.shards[0].sizes)
        for shard in entry.shards:
            for dim in range(len(shard.offsets)):
                if shard.offsets[dim] + shard.sizes[dim] > global_shape[dim]:
                    global_shape[dim] = shard.offsets[dim] + shard.sizes[dim]
        return global_shape

    @staticmethod
    def _validate_shape(global_shape: List[int], obj_out: torch.Tensor) -> None:
        if isinstance(obj_out, ShardedTensor):
            out_shape = list(obj_out.metadata().size)
        else:
            out_shape = list(obj_out.shape)
        if out_shape != global_shape:
            logger.warning(
                f"The shape of obj_out ({out_shape}) is different from the "
                f"shape of the persisted sharded tensor ({global_shape}). "
                "Only the overlapping part will be loaded. "
            )

    @classmethod
    def prepare_read(
        cls,
        entry: ShardedTensorEntry,
        obj_out: Optional[ShardedTensor] = None,
    ) -> Tuple[List[ReadReq], Future[Union[ShardedTensor, torch.Tensor]]]:
        # Note: in case obj_out is None, a Future[Tensor] will be returned
        if obj_out is None:
            obj_out = ShardedTensorIOPreparer.empty_tensor_from_sharded_tensor_entry(
                entry
            )

        global_shape = cls._get_global_shape(entry=entry)
        cls._validate_shape(global_shape=global_shape, obj_out=obj_out)

        if type(obj_out) == ShardedTensor:
            local_shards = obj_out.local_shards()
        elif type(obj_out) == torch.Tensor:
            local_shards = [
                ShardedTensorShard(
                    tensor=obj_out,
                    metadata=ShardMetadata(
                        shard_offsets=[0] * len(obj_out.shape),
                        shard_sizes=list(obj_out.shape),
                        placement=str(obj_out.device),
                    ),
                )
            ]
        else:
            raise RuntimeError(
                f"obj_out must either be a Tensor or ShardedTensor (got {type(obj_out)})"
            )

        # For each persisted shard, find all its overlapping regions with the
        # local shards
        path_byte_range_to_overlapping_regions = defaultdict(list)
        for local_shard, shard in itertools.product(local_shards, entry.shards):
            shard_md = ShardMetadata(
                shard_offsets=shard.offsets,
                shard_sizes=shard.sizes,
                placement="cpu",
            )
            if not _check_shard_metadata_pair_overlap(local_shard.metadata, shard_md):
                continue
            path_byte_range = (shard.tensor.location, shard.tensor.byte_range_tuple)
            path_byte_range_to_overlapping_regions[path_byte_range].append(
                _OverlappingRegion(
                    dst_tensor=local_shard.tensor,
                    overlap_region=cls._shards_get_overlap_region_wrt_saved_tensor(
                        saved_shard=shard_md,
                        current_shard=local_shard.metadata,
                    ),
                )
            )

        # Read each persisted shard once and use it to load all its overlapping
        # regions with the local shards
        read_reqs = []
        for shard in entry.shards:
            path_byte_range = (shard.tensor.location, shard.tensor.byte_range_tuple)
            if path_byte_range not in path_byte_range_to_overlapping_regions:
                continue
            read_reqs.append(
                ReadReq(
                    path=shard.tensor.location,
                    buffer_consumer=ShardedTensorBufferConsumer(
                        overlapping_regions=path_byte_range_to_overlapping_regions[
                            path_byte_range
                        ],
                        entry=shard.tensor,
                    ),
                    byte_range=shard.tensor.byte_range_tuple,
                )
            )
        return read_reqs, Future(obj=obj_out)

    @staticmethod
    def empty_tensor_from_sharded_tensor_entry(
        entry: ShardedTensorEntry,
    ) -> torch.Tensor:
        # construct tensor for to fill in-place
        # by reading shard metadata
        shape = entry.get_tensor_shape()
        dtype = entry.shards[0].tensor.dtype
        tensor = torch.empty(shape, dtype=string_to_dtype(dtype))
        return tensor


@dataclass
class _OverlappingRegion:
    dst_tensor: torch.Tensor
    overlap_region: List[
        Tuple[int, int, int, int]
    ]  # (dim, src_offset, dst_offset, length)

    def get_views(self, src_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_view = src_tensor
        dst_view = self.dst_tensor
        for dim, src_offset, dst_offset, length in self.overlap_region:
            src_view = torch.narrow(src_view, dim, src_offset, length)
            dst_view = torch.narrow(dst_view, dim, dst_offset, length)
        return src_view, dst_view


class ShardedTensorBufferConsumer(BufferConsumer):
    def __init__(
        self,
        overlapping_regions: List[_OverlappingRegion],
        entry: TensorEntry,
    ) -> None:
        self.overlapping_regions = overlapping_regions
        self.entry = entry

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        deserialized = TensorBufferConsumer.deserialize_tensor(
            buf=buf, entry=self.entry
        )
        for overlapping_region in self.overlapping_regions:
            src_view, dst_view = overlapping_region.get_views(src_tensor=deserialized)
            if executor is not None:
                await asyncio.get_running_loop().run_in_executor(
                    executor, tensor_copy, dst_view, src_view
                )
            else:
                tensor_copy(dst_view, src_view)

    def get_consuming_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")
