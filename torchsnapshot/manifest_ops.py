#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
from collections import defaultdict
from typing import Dict, List, Tuple

from torchsnapshot.dtensor_utils import _ReplicatedShards
from torchsnapshot.manifest_utils import (
    _get_replicated_ranks,
    is_container_entry,
    is_dict_entry,
    is_fully_replicated_entry,
)

from .knobs import is_sharded_tensor_elasticity_enabled_at_root_only
from .manifest import (
    DTensorEntry,
    Entry,
    Manifest,
    ShardedTensorEntry,
    SnapshotMetadata,
)


def get_manifest_for_rank(
    metadata: SnapshotMetadata, rank: int
) -> Tuple[Manifest, Dict[str, Entry]]:
    """
    Get the local manifest for the rank from the snapshot metadata.

    Args:
        metadata: The snapshot metadata.
        rank: The target rank.

    Returns:
        The local manifest for the rank and merged sharded tensor entries.
    """
    rank_to_manifest = _get_rank_to_manifest(metadata=metadata)
    merged_sd_entries: Dict[str, Entry] = {}
    merged_sd_entries = _get_merged_sharded_tensor_entries(rank_to_manifest)
    merged_sd_entries.update(_get_merged_dtensor_entries(rank_to_manifest))

    if rank < metadata.world_size:
        return (
            _get_manifest_for_existing_rank(
                rank_to_manifest=rank_to_manifest,
                merged_sd_entries=merged_sd_entries,
                rank=rank,
            ),
            merged_sd_entries,
        )
    else:
        return (
            _get_manifest_for_new_rank(rank_to_manifest=rank_to_manifest),
            merged_sd_entries,
        )


def _get_manifest_for_existing_rank(
    rank_to_manifest: List[Dict[str, Entry]],
    merged_sd_entries: Dict[str, Entry],
    rank: int,
) -> Manifest:
    local_manifest = rank_to_manifest[rank].copy()

    # Replicated entries are removed from the global manifest
    for logical_path, entry in rank_to_manifest[0].items():
        if is_fully_replicated_entry(entry):
            local_manifest[logical_path] = entry

    for logical_path, entry in local_manifest.items():
        if isinstance(entry, ShardedTensorEntry) or isinstance(entry, DTensorEntry):
            local_manifest[logical_path] = merged_sd_entries[logical_path]

    return local_manifest


def _get_manifest_for_new_rank(rank_to_manifest: List[Dict[str, Entry]]) -> Manifest:
    # Use rank 0's manifest as the base
    local_manifest = rank_to_manifest[0].copy()

    # Remove non-replicated entries
    for logical_path in list(local_manifest.keys()):
        entry = local_manifest[logical_path]
        if is_container_entry(entry) or is_fully_replicated_entry(entry):
            continue
        _remove_entry(manifest=local_manifest, logical_path=logical_path)
    return local_manifest


def _get_rank_to_manifest(metadata: SnapshotMetadata) -> List[Dict[str, Entry]]:
    rank_to_manifest: List[Dict[str, Entry]] = [{} for _ in range(metadata.world_size)]
    for path, entry in metadata.manifest.items():
        tokens = path.split("/")
        rnk = int(tokens.pop(0))
        logical_path = "/".join(tokens)
        rank_to_manifest[rnk][logical_path] = entry
    return copy.deepcopy(rank_to_manifest)


def _get_merged_sharded_tensor_entries(
    rank_to_manifest: List[Dict[str, Entry]],
) -> Dict[str, Entry]:
    groups = defaultdict(list)
    for manifest in rank_to_manifest:
        for logical_path, entry in manifest.items():
            if isinstance(entry, ShardedTensorEntry):
                groups[logical_path].append(entry)

    sd_entries = {}
    for logical_path, group in groups.items():
        shards = sorted(
            (shard for entry in group for shard in entry.shards),
            key=lambda shard: shard.offsets,
        )
        sd_entries[logical_path] = ShardedTensorEntry(
            shards=shards,
        )
    return sd_entries


def _get_merged_dtensor_entries(
    rank_to_manifest: List[Dict[str, Entry]],
) -> Dict[str, Entry]:
    """
    Merge all DTensor entries across ranks if sharded
    """
    # Collects all entries across ranks for a given logical path
    groups = defaultdict(list)
    # Ranks whose shards have been covered so far by another replicated rank, keyed by logical path
    path_to_processed_ranks = defaultdict(set)
    # Contains mapping from a rank to all other ranks that have a replicate of its shard, keyed by logical path
    path_to_replicated_ranks: Dict[str, _ReplicatedShards] = {}

    for rank, manifest in enumerate(rank_to_manifest):
        for logical_path, entry in manifest.items():
            if isinstance(entry, DTensorEntry):
                # If the DTensor is not sharded or we already covered this shard, skip
                if (
                    is_fully_replicated_entry(entry)
                    or rank in path_to_processed_ranks[logical_path]
                ):
                    continue
                if logical_path not in path_to_replicated_ranks:
                    path_to_replicated_ranks[logical_path] = _ReplicatedShards(
                        replicated_ranks_for_shards=_get_replicated_ranks(
                            entry=entry,
                        )
                    )
                ranks_with_this_shard = path_to_replicated_ranks[
                    logical_path
                ].get_all_replicated_ranks(rank)
                path_to_processed_ranks[logical_path].update(ranks_with_this_shard)
                groups[logical_path].append(entry)

    sd_entries = {}
    for logical_path, group in groups.items():
        shards = sorted(
            (shard for entry in group for shard in entry.shards),
            key=lambda shard: shard.offsets,
        )
        sd_entries[logical_path] = DTensorEntry(
            mesh=group[0].mesh,
            dim_map=group[0].dim_map,
            shards=shards,
        )
    return sd_entries


def handle_sharded_tensor_elasticity(
    manifest: Manifest,
    merged_sd_entries: Dict[str, Entry],
    tensor_requests: List[str],
) -> None:
    """
    Handles the elastic behavior of :class:`ShardedTensor` and :class:`DTensor`.

    Both can be elastic in several ways:

    - A rank loads a portion of a sharded tensor different from what it saved
    - A rank loads a sharded tensor that it did not participate in saving
    - A rank doesn't load a sharded tensor that it participated in saving

    The first scenario is taken care of by :func:`get_manifest_for_rank`, which
    makes all shards available to all instances of :class:`ShardedTensorEntry`
    and :class:`DTensorEntry`

    The second and the third scenarios require manipulating the presence of a
    sharded tensor in the loaded state dict, which is handled by this function:

    - If the sharded tensor entry is missing from the local manifest (i.e. the
      rank did not participate in saving it), this function adds the entry to
      the local manifest.
    - If the sharded tensor is missing from the model's state dict, the
      function removes the corresponding entry from the local manifest.

    This function works best effort to support elastic behavior for models and optimizers.

    NOTE: By default, this assumes that all ranks contain a corresponding sharded tensor entry for the particular stateful.
    In case not all ranks contain the sharded tensor entry for a particular stateful object, setting `TORCHSNAPSHOT_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ONLY=1`
    will restrict manipulating the presence of sharded tensors only if all sharded tensors are at the root of the state dict.

    Args:
        manifest: The local manifest for the rank.
        merged_sd_entries: The merged sharded tensor entries.
        tensor_requests: The logical paths of tensors in the target stateful
            object's state dict.
    """

    # Some state dicts might be irregular, in that they have `foo/bar/sharded_tensor` on rank A but not have `foo` on rank B.
    # To load a sharded tensor, every rank has to have `foo/bar/sharded_tensor`.
    # This means that in order to support automatic resharding, we have to create foo/bar/sharded_tensor on ranks that don't have it.
    # As a simplifying assumption, torchsnapshot exposes this knob to manipulate the presence of sharded tensors only if all sharded tensors
    # are at the root of the state dict.
    if is_sharded_tensor_elasticity_enabled_at_root_only() and not all(
        len(logical_path.split("/")) == 2 for logical_path in merged_sd_entries
    ):
        return

    # Filter out tensor requests that will not be fulfilled by ShardedTensorEntry/DTensorEntry
    tensor_requests = [tr for tr in tensor_requests if tr in merged_sd_entries]

    # Add missing sharded tensor entries that are requested to the manifest
    for logical_path in tensor_requests:
        if logical_path not in manifest:
            manifest[logical_path] = merged_sd_entries[logical_path]
            tokens = logical_path.split("/")
            key = tokens.pop()
            manifest["/".join(tokens)].keys.append(key)

    # Remove sharded tensor entries that are not requested from the manifest
    for logical_path in list(manifest.keys()):
        if (
            isinstance(manifest[logical_path], ShardedTensorEntry)
            or isinstance(manifest[logical_path], DTensorEntry)
        ) and logical_path not in tensor_requests:
            del manifest[logical_path]


def _remove_entry(manifest: Manifest, logical_path: str) -> None:
    """
    Remove an entry from a manifest.

    Example:

        manifest (before):
        {
            "foo": DictEntry(keys=["bar", "baz"]),
            "foo/bar": ...,
            "foo/baz": ...,
        }

        logical_path: "foo/bar",

        manifest (after):
        {
            "foo": DictEntry(keys=["baz"]),
            "foo/baz": ...,
        }
    """
    if logical_path not in manifest:
        return

    del manifest[logical_path]

    tokens = logical_path.split("/")
    key = tokens.pop()
    parent_path = "/".join(tokens)
    if len(parent_path) == 0:
        return

    parent = manifest[parent_path]
    if is_dict_entry(parent):
        if key in parent.keys:
            parent.keys.remove(key)
        else:
            parent.keys.remove(int(key))
