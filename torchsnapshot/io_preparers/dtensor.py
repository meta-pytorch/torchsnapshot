#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch._prims_common import ShapeType
from torch.distributed._shard.sharded_tensor import (
    Shard as ShardedTensorShard,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Placement,
    Replicate,
    Shard as ShardPlacement,
)

try:
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

except ImportError:

    def compute_local_shape_and_global_offset(
        global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        raise RuntimeError(
            "Please use the latest nightly pytorch release to use this feature."
        )


from torchsnapshot.io_preparers.sharded_tensor import (
    _OverlappingRegion,
    ShardedTensorBufferConsumer,
    ShardedTensorIOPreparer,
)
from torchsnapshot.io_preparers.tensor import TensorIOPreparer
from torchsnapshot.io_types import Future, ReadReq, WriteReq
from torchsnapshot.knobs import get_max_shard_size_bytes
from torchsnapshot.manifest import DTensorEntry, Shard as ShardEntry

logger: logging.Logger = logging.getLogger(__name__)


class DTensorIOPreparer:
    @staticmethod
    def _get_largest_shard_dim(
        local_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
    ) -> int:
        """
        Given a local shard, identify which dims have been sharded and chunk across the largest one.
        If it is not sharded, chunk over the largest dim.

        Args:
            shard (Tensor): local tensor on this rank
            mesh (DeviceMesh): device mesh associated with the DTensor the shard is from
            placements (Tuple[Placement,...]): placements associated with the DTensor the shard is from

        Returns:
            largest_shard_dim (int): index of largest dim that is sharded
        """
        # Find which dims of device mesh have Shard placements,
        # then get the tensor dim we are sharding across
        shard_dims = [
            placement.dim
            for placement in placements
            if isinstance(placement, ShardPlacement)
        ]

        # Find the largest sharding dim for the local tensor, so we can chunk
        # for parallelized saving
        if len(shard_dims) > 0:
            largest_shard_dim = max(shard_dims, key=lambda dim: local_shape[dim])
        else:
            # If tensor is not sharded then it is replicated across the device mesh.
            # Just chunk over largest dimension
            largest_shard_dim = torch.argmax(torch.tensor(local_shape)).item()

        # pyre-fixme[7]: Expected `int` but got `Union[bool, float, int]`.
        return largest_shard_dim

    @staticmethod
    def _get_dim_map(obj: DTensor) -> List[List[int]]:
        dim_map = [[] for _ in range(len(obj.size()))]
        for mesh_dim, pm in enumerate(obj.placements):
            if isinstance(pm, ShardPlacement):
                dim_map[pm.dim].append(mesh_dim)
            elif not isinstance(pm, Replicate):
                raise ValueError("Unsupported placement type")
        # Add [-1] for mesh dims that aren't sharded
        dim_map = [dims if len(dims) != 0 else [-1] for dims in dim_map]
        return dim_map

    @staticmethod
    def _get_global_shape(entry: DTensorEntry) -> List[int]:
        # Note: this assumes you've already merged DTensorEntries across ranks
        # (if the write loads were partitioned) with a function like
        # get_manifest_for_rank(). This will not compute the correct shape otherwise.
        global_shape = [0] * len(entry.shards[0].sizes)
        for shard in entry.shards:
            for dim in range(len(shard.offsets)):
                if shard.offsets[dim] + shard.sizes[dim] > global_shape[dim]:
                    global_shape[dim] = shard.offsets[dim] + shard.sizes[dim]
        return global_shape

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        obj: DTensor,
        is_async_snapshot: bool = False,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[DTensorEntry, List[WriteReq]]:
        """
        Compile list of write requests and the corresponding DTensorEntry given the provided
        DTensor object.

        Args:
            storage_path (str): device location of the DTensor. Because the DTensor is sharded and/or replicated
                the path will usually start with "sharded/" or "replicated/" instead of the rank number.
            obj (DTensor): DTensor object to be saved.
            is_async_snapshot (bool): whether or not the write request is from async_take
            _tensor_prepare_func (Optional[Callable[[torch.Tensor, bool], torch.Tensor]]): custom transform to apply
                to tensor before staging it in buffer.

        Returns:
            (DTensorEntry, List[WriteReq]): list of write requests and corresponding DTensorEntry
        """
        mesh = obj.device_mesh
        placements = obj.placements

        # Get shape of local shard on this rank and the offsets w.r.t. the global tensor
        local_shape, shard_offsets = compute_local_shape_and_global_offset(
            obj.size(), mesh, placements
        )

        # Determine which sharding dim to chunk
        largest_shard_dim = cls._get_largest_shard_dim(local_shape, mesh, placements)

        # Chunk the shard across the largest dim to improve save performance
        subdivided = ShardedTensorIOPreparer.subdivide_shard(
            shard=obj.to_local(),
            offsets=list(shard_offsets),
            sizes=list(local_shape),
            dim=largest_shard_dim,
            max_shard_sz_bytes=get_max_shard_size_bytes(),
        )

        # Create write reqs for each chunk of the local shard for this rank
        # Each chunk is represented as a Shard in the manifest, although
        # the actual shard is the tensor pre-chunking. Chunking is simply
        # done for optimization purposes.
        write_reqs = []
        shards = []
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
                ShardEntry(
                    offsets=offsets,
                    sizes=sizes,
                    tensor=entry,
                )
            )

        dtensor_entry = DTensorEntry(
            shards=shards,
            mesh=mesh.mesh.cpu().numpy().tolist(),
            dim_map=cls._get_dim_map(obj),
        )

        return dtensor_entry, write_reqs

    @classmethod
    def prepare_read(
        cls,
        entry: DTensorEntry,
        obj_out: Optional[DTensor] = None,
    ) -> Tuple[List[ReadReq], Future[DTensor]]:
        if obj_out is None:
            raise RuntimeError(
                "No output DTensor object found. Cannot read a DTensorEntry without a runtime object."
            )

        entry_global_shape = cls._get_global_shape(entry=entry)
        out_global_shape = obj_out.shape
        if out_global_shape != entry_global_shape:
            logger.warning(
                f"The shape of obj_out ({out_global_shape}) is different from the "
                f"shape of the persisted sharded tensor ({entry_global_shape}). "
                "Only the overlapping part will be loaded. "
            )

        out_local_shape, out_global_offset = compute_local_shape_and_global_offset(
            out_global_shape, obj_out.device_mesh, obj_out.placements
        )
        out_local_tensor = obj_out.to_local()
        assert out_local_tensor.shape == out_local_shape

        local_shards = [
            ShardedTensorShard(
                tensor=out_local_tensor,
                metadata=ShardMetadata(
                    shard_offsets=list(out_global_offset),
                    shard_sizes=list(out_local_shape),
                    placement=str(out_local_tensor.device),
                ),
            )
        ]

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
                    overlap_region=ShardedTensorIOPreparer._shards_get_overlap_region_wrt_saved_tensor(
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
