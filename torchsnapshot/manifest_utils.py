#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from typing import List, Set

import numpy as np
from torchsnapshot.manifest import (
    DictEntry,
    DTensorEntry,
    Entry,
    ListEntry,
    OrderedDictEntry,
    ShardedTensorEntry,
)


def is_dict_entry(entry: Entry) -> bool:
    return isinstance(entry, (DictEntry, OrderedDictEntry))


def is_replicated_entry(entry: Entry) -> bool:
    """
    Returns true if entry is partially or fully replicated.
    """
    return is_fully_replicated_entry(entry) or is_partially_replicated_entry(entry)


def is_container_entry(entry: Entry) -> bool:
    return isinstance(entry, (ListEntry, DictEntry, OrderedDictEntry))


def is_sharded_entry(entry: Entry) -> bool:
    if isinstance(entry, DTensorEntry):
        return any(dims[0] != -1 for dims in entry.dim_map)
    return isinstance(entry, ShardedTensorEntry)


def is_fully_replicated_entry(entry: Entry) -> bool:
    """
    Return True for an entry that is fully replicated on all ranks
    """
    if isinstance(entry, DTensorEntry):
        return all(dims[0] == -1 for dims in entry.dim_map)
    if not hasattr(entry, "replicated"):
        return False
    # pyre-ignore
    return entry.replicated


def is_partially_replicated_entry(entry: Entry) -> bool:
    """
    Return True for an entry that is both sharded and replicated, which only applies
    to DTensorEntries
    """
    if isinstance(entry, DTensorEntry):
        return (
            0 < sum(1 for dims in entry.dim_map if dims[0] == -1) < len(entry.dim_map)
        )
    return False


def _get_replicated_ranks(
    entry: DTensorEntry,
) -> List[Set[int]]:
    """
    Given a DTensorEntry across ranks, return a list of rank sets
    where each set denotes a replicated shard.
    """

    mesh = entry.mesh
    mesh_shape = np.array(entry.mesh).shape
    dim_map = entry.dim_map
    shard_dims = []
    for dims in dim_map:
        if dims[0] != -1:
            shard_dims.extend(dims)
    replicate_dims = set(range(len(mesh_shape))) - set(shard_dims)

    # Programmatically generate slices of the device mesh that represent
    # sets of replicated ranks. Iterate across sharded dims, taking the
    # whole slice of the replicated dim each time.
    #
    # Example:
    # 3D mesh = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], replicate on dim 0, shard on dims 1, 2
    # The sets of replicated ranks returned is [[0, 4], [1, 5], [2, 6], [3, 7]]
    slices_for_dims = []
    mesh_shape = np.array(mesh).shape
    for dim, size in enumerate(mesh_shape):
        if dim in replicate_dims:
            # Take entire dimension
            slices_for_dims.append([slice(None)])
        elif dim in shard_dims:
            # Take one element at a time
            slices_for_dims.append([slice(i, i + 1) for i in range(size)])

    slice_combinations = list(itertools.product(*slices_for_dims))
    # Gymnastics to take advantage of numpy's multidimensional slicing and squeeze
    return [set(np.array(mesh)[s].flatten()) for s in slice_combinations]
