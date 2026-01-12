#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import cast, Dict, List, Tuple

import numpy as np
from torchsnapshot.manifest_utils import is_fully_replicated_entry

from .io_preparer import ObjectBufferStager, TensorBufferStager, TensorIOPreparer
from .io_types import WriteReq
from .manifest import ChunkedTensorEntry, DTensorEntry, Entry
from .manifest_utils import (
    _get_replicated_ranks,
    is_partially_replicated_entry,
    is_replicated_entry,
)
from .pg_wrapper import PGWrapper


@dataclass(frozen=True)
class _WriteLoad:
    logical_path: str
    write_req_idx: int
    size: int


def _is_subpartitionable(
    logical_path: str,
    rank_to_entries: List[Dict[str, Entry]],
) -> bool:
    entries = [entries[logical_path] for entries in rank_to_entries]
    return isinstance(entries[0], ChunkedTensorEntry) and all(
        entry == entries[0] for entry in entries
    )


def _assign_rank_write_loads(
    rank_to_write_loads: List[Dict[str, List[_WriteLoad]]],
    rank_to_size: List[int],
    ranks_to_choose: List[int],
    logical_path: str,
    size: int,
    partition_result: List[List[_WriteLoad]],
) -> None:
    """
    Given a list of write loads for each rank, assign new write load to rank
    with smallest load.
    """
    chosen_rank = min(ranks_to_choose, key=lambda rank: rank_to_size[rank])
    partition_result[chosen_rank].extend(rank_to_write_loads[chosen_rank][logical_path])
    rank_to_size[chosen_rank] += size


def _partition_write_loads(
    rank_to_entries: List[Dict[str, Entry]],
    rank_to_write_loads: List[Dict[str, List[_WriteLoad]]],
    rank_to_size: List[int],
    world_size: int,
) -> List[List[_WriteLoad]]:
    partition_result: List[List[_WriteLoad]] = [[] for _ in range(world_size)]
    partitionables = set()

    for logical_path in rank_to_entries[0].keys():
        # A logical path may associate with multiple write requests spread. We
        # say the logical path is subpartitionable if the different write
        # requests can be fulfilled by different ranks.
        if not _is_subpartitionable(
            logical_path=logical_path,
            rank_to_entries=rank_to_entries,
        ):
            # If the logical path is not subpartitionable, all associated write
            # requests need to be fulfilled by a single rank.
            size = sum(wl.size for wl in rank_to_write_loads[0][logical_path])
            # If the entry is partially replicated (i.e., sharded across one mesh dim
            # and replicated across another), then only choose from replicated ranks
            # Only applicable to DTensorEntry.
            if is_partially_replicated_entry(rank_to_entries[0][logical_path]):
                replicated_ranks = _get_replicated_ranks(
                    entry=cast(DTensorEntry, rank_to_entries[0][logical_path]),
                )
                # For each set of replicated ranks, assign write load to rank
                # with smallest current load
                for ranks_to_choose in replicated_ranks:
                    _assign_rank_write_loads(
                        rank_to_write_loads=rank_to_write_loads,
                        rank_to_size=rank_to_size,
                        ranks_to_choose=list(ranks_to_choose),
                        logical_path=logical_path,
                        size=size,
                        partition_result=partition_result,
                    )
            # Fully replicated case
            else:
                _assign_rank_write_loads(
                    rank_to_write_loads=rank_to_write_loads,
                    rank_to_size=rank_to_size,
                    ranks_to_choose=list(range(world_size)),
                    logical_path=logical_path,
                    size=size,
                    partition_result=partition_result,
                )
        else:
            # If the logical path is subpartitionable, all associated write
            # loads are considered a unit of partitioning.
            partitionables.update(rank_to_write_loads[0][logical_path])

    # Greedily assign replicated chunks among ranks, based on current sizes of ranks
    for partitionable in partitionables:
        chosen_rank = np.argmin(rank_to_size)
        partition_result[chosen_rank].append(partitionable)
        rank_to_size[chosen_rank] += partitionable.size

    return partition_result


def _estimate_write_req_storage_size(write_req: WriteReq) -> int:
    buffer_stager = write_req.buffer_stager
    if isinstance(buffer_stager, TensorBufferStager):
        size = TensorIOPreparer.get_tensor_size_from_entry(entry=buffer_stager.entry)
    elif isinstance(buffer_stager, ObjectBufferStager):
        size = buffer_stager.get_staging_cost_bytes()
    else:
        raise AssertionError(f"Unrecognized buffer stager type {type(buffer_stager)}")
    return size


def _partition_replicated_write_reqs(
    entries: Dict[str, Entry],
    write_reqs: Dict[str, List[WriteReq]],
    non_replicated_size: int,
    pg: PGWrapper,
) -> Tuple[Dict[str, Entry], Dict[str, List[WriteReq]]]:
    """
    Partition replicated write requests across all ranks.

    Args:
        entries: The replicated entries to be produced by the current rank.
        write_reqs: The replicated write requests to be fulfilled by the current rank.
        non_replicated_size: The total size of the non-replicated write
            requests on the current rank.
        pg: The process group used for partitioning.

    Returns:
        Partitioned entries and write requests.
    """
    # Invariant: replicated logical paths are already verified
    write_loads = defaultdict(list)
    for logical_path, wrs in write_reqs.items():
        for idx, wr in enumerate(wrs):
            size = _estimate_write_req_storage_size(write_req=wr)
            write_load = _WriteLoad(
                logical_path=logical_path, write_req_idx=idx, size=size
            )
            write_loads[logical_path].append(write_load)

    # pyre-ignore
    object_list: List[Tuple[Dict[str, Entry], Dict[str, List[_WriteLoad]], int]] = [
        None
    ] * pg.get_world_size()
    pg.all_gather_object(
        obj_list=object_list, obj=(entries, write_loads, non_replicated_size)
    )
    rank_to_entries, rank_to_write_loads, rank_to_size = list(zip(*object_list))

    if pg.get_rank() == 0:
        # Rank 0 performs the partitioning
        partition_result = _partition_write_loads(
            rank_to_entries=rank_to_entries,
            rank_to_write_loads=rank_to_write_loads,
            rank_to_size=list(rank_to_size),
            world_size=pg.get_world_size(),
        )
        obj_list = [partition_result]
    else:
        # pyre-ignore
        obj_list: List[List[List[_WriteLoad]]] = [None]

    pg.broadcast_object_list(obj_list=obj_list, src=0)
    partition_result = obj_list[0]

    write_loads = sorted(
        (write_load.logical_path, write_load.write_req_idx)
        for write_load in partition_result[pg.get_rank()]
    )
    new_entries = {}
    new_write_reqs = defaultdict(list)
    for logical_path, write_req_idx in write_loads:
        entry = entries[logical_path]
        if isinstance(entry, ChunkedTensorEntry):
            chunk = entry.chunks[write_req_idx]
            if logical_path not in new_entries:
                new_entries[logical_path] = copy.deepcopy(entry)
                new_entries[logical_path].chunks = [chunk]
            else:
                new_entries[logical_path].chunks.append(chunk)
        else:
            new_entries[logical_path] = entry
        new_write_reqs[logical_path].append(write_reqs[logical_path][write_req_idx])

    return new_entries, new_write_reqs


def partition_write_reqs(
    entries: Dict[str, Entry], write_reqs: Dict[str, List[WriteReq]], pg: PGWrapper
) -> Tuple[Dict[str, Entry], Dict[str, List[WriteReq]]]:
    """
    Partition replicated write requests across all ranks.

    After partitioning, each replicated write request will only be in the
    returned write requests on a single rank.

    NOTE: after partitioning, the returned entries only covers objects/parts of
    objects whose write requests are to be fulfilled by the current rank, which
    may not contain all entries for the current rank. After partitioning, the
    entries needs to be consolidated with func::`consolidate_replicated_entries`
    before being written to the manifest.

    Args:
        entries: The entries to be produced by the current rank.
        write_reqs: The write requests to be fulfilled by the current rank.
        pg: The process group used for partitioning.

    Returns:
        Partitioned entries and write requests.
    """
    # Verify that all entries associated with the write reqs are passed in
    if not set(write_reqs.keys()).issubset(set(entries.keys())):
        raise RuntimeError(
            "Not all entries associated with the write reqs are passed in. "
            f"Missing: {set(write_reqs.keys()) - set(entries.keys())}."
        )

    if os.environ.get("TORCH_SNAPSHOT_DISABLE_PARTITIONER") is not None:
        raise NotImplementedError(
            "TORCH_SNAPSHOT_DISABLE_PARTITIONER is not implemented."
        )

    # Split replicated and non-replicated entries and write requests
    replicated_entries = {k: v for k, v in entries.items() if is_replicated_entry(v)}
    replicated_write_reqs = {
        k: v for k, v in write_reqs.items() if k in replicated_entries
    }
    non_replicated_entries = {
        k: v for k, v in entries.items() if not is_replicated_entry(v)
    }
    non_replicated_write_reqs = {
        k: v for k, v in write_reqs.items() if k in non_replicated_entries
    }

    # Calculate the total size of non-replicated write requests on the current
    # rank. This will be taken into account when partitioning the replicated
    # write requests among ranks.
    non_replicated_size = sum(
        _estimate_write_req_storage_size(wr)
        for wrs in non_replicated_write_reqs.values()
        for wr in wrs
    )

    # Partition replicated write requests among ranks
    replicated_entries, replicated_write_reqs = _partition_replicated_write_reqs(
        entries=replicated_entries,
        write_reqs=replicated_write_reqs,
        non_replicated_size=non_replicated_size,
        pg=pg,
    )
    new_entries = {**replicated_entries, **non_replicated_entries}
    new_write_reqs = {**replicated_write_reqs, **non_replicated_write_reqs}

    return new_entries, new_write_reqs


def _consolidate_replicated_chunked_tensor_entries(
    rank_to_entries: List[Dict[str, Entry]],
) -> List[Dict[str, Entry]]:
    groups: Dict[str, List[ChunkedTensorEntry]] = defaultdict(list)

    for entries in rank_to_entries:
        for logical_path, entry in entries.items():
            if is_replicated_entry(entry) and isinstance(entry, ChunkedTensorEntry):
                groups[logical_path].append(entry)

    for logical_path, group in groups.items():
        merged = ChunkedTensorEntry(
            dtype=group[0].dtype,
            shape=group[0].shape,
            chunks=sorted(
                (chunk for entry in group for chunk in entry.chunks),
                key=lambda chunk: chunk.offsets,
            ),
            replicated=True,
        )
        for entries in rank_to_entries:
            entries[logical_path] = merged

    return rank_to_entries


def consolidate_replicated_entries(
    rank_to_entries: List[Dict[str, Entry]], dedup: bool = True
) -> List[Dict[str, Entry]]:
    """
    Consolidate replicated entries across ranks.

    After using func::`partition_write_reqs` to partition replicated write
    requests, the entries must be consolidated with this function before being
    written to the manifest.

    Args:
        rank_to_entries: The entries from all ranks.
        dedup: Whether to place replicated entries only in rank 0's manifest.

    Returns:
        Consolidated entries for all ranks.
    """
    rank_to_entries = _consolidate_replicated_chunked_tensor_entries(
        rank_to_entries=rank_to_entries
    )

    # Collect all replicated entries and remove them from the manifests
    replicated_entries = {}
    for entries in rank_to_entries:
        for logical_path in list(entries.keys()):
            entry = entries[logical_path]
            if not is_fully_replicated_entry(entry):
                continue
            if logical_path in replicated_entries:
                if replicated_entries[logical_path] != entry:
                    raise ValueError(
                        f"Paths for replicated entry for {logical_path} do not match: replicated entries={replicated_entries[logical_path]} vs. entry={entry}"
                    )
            else:
                replicated_entries[logical_path] = entry
            del entries[logical_path]

    # Add the replicated entries to the manifests
    for rank, entries in enumerate(rank_to_entries):
        if dedup and rank != 0:
            continue
        for logical_path, entry in replicated_entries.items():
            entries[logical_path] = entry

    return rank_to_entries


def consolidate_replicated_entries_dist(
    entries: Dict[str, Entry], pg: PGWrapper, dedup: bool = True
) -> Dict[str, Entry]:
    # pyre-ignore
    obj_list: List[Dict[str, Entry]] = [None] * pg.get_world_size()
    pg.all_gather_object(obj_list=obj_list, obj=entries)
    rank_to_entries = consolidate_replicated_entries(
        rank_to_entries=obj_list,
        dedup=dedup,
    )
    return rank_to_entries[pg.get_rank()]
