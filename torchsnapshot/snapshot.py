#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import copy
import fnmatch
import functools
import itertools
import logging
import os
import random
import sys
import traceback
from asyncio import AbstractEventLoop
from collections import defaultdict
from datetime import timedelta
from threading import Thread
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot.dtensor_utils import is_sharded

from .asyncio_utils import maybe_nested_loop
from .batcher import batch_read_requests, batch_write_requests
from .dist_store import get_or_create_store, LinearBarrier
from .event import Event
from .event_handlers import log_event
from .flatten import flatten, inflate
from .io_preparer import prepare_read, prepare_write
from .io_types import ReadIO, ReadReq, StoragePlugin, WriteIO, WriteReq
from .knobs import is_batching_disabled
from .manifest import Entry, Manifest, PrimitiveEntry, SnapshotMetadata
from .manifest_ops import get_manifest_for_rank, handle_sharded_tensor_elasticity
from .manifest_utils import is_container_entry
from .partitioner import consolidate_replicated_entries, partition_write_reqs
from .pg_wrapper import PGWrapper
from .rng_state import RNGState
from .scheduler import (
    _MAX_PER_RANK_MEMORY_BUDGET_BYTES,
    get_process_memory_budget_bytes,
    PendingIOWork,
    sync_execute_read_reqs,
    sync_execute_write_reqs,
)
from .stateful import AppState, Stateful
from .storage_plugin import url_to_storage_plugin_in_event_loop
from .version import __version__ as torchsnapshot_version

logger: logging.Logger = logging.getLogger(__name__)

SNAPSHOT_METADATA_FNAME = ".snapshot_metadata"
T = TypeVar("T")


class Snapshot:
    """
    Create a reference to an existing snapshot.

    Args:
        path (str): The path to the snapshot. This should be the same as the
            ``path`` argument used for :func:`Snapshot.take` when the snapshot
            was taken.

        pg (ProcessGroup, optional): The process group for the participants of
            :meth:`Snapshot.restore`. If none, the default process group will be
            used.

        storage_options (Dict[str, Any], optional): Additional keyword options
            for the storage plugin to use. See each storage plugin's documentation
            for customizations.
    """

    def __init__(
        self,
        path: str,
        pg: Optional[dist.ProcessGroup] = None,
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path: str = path
        self.pg: Optional[dist.ProcessGroup] = pg
        self._metadata: Optional[SnapshotMetadata] = None
        self._storage_options = storage_options

    @property
    def metadata(self) -> SnapshotMetadata:
        if self._metadata is None:
            event_loop = maybe_nested_loop()
            storage = url_to_storage_plugin_in_event_loop(
                url_path=self.path,
                event_loop=event_loop,
                storage_options=self._storage_options,
            )
            self._metadata = self._read_snapshot_metadata(
                storage=storage, event_loop=event_loop
            )
            storage.sync_close(event_loop=event_loop)
            event_loop.close()
        return cast(SnapshotMetadata, self._metadata)

    @classmethod
    def take(
        cls,
        path: str,
        app_state: AppState,
        pg: Optional[dist.ProcessGroup] = None,
        replicated: Optional[List[str]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        _custom_tensor_prepare_func: Optional[
            Callable[[str, torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> "Snapshot":
        """
        Takes a snapshot of the application state.

        Args:
            app_state (Dict[str, Stateful]): The application state to persist.
                It takes the form of a dictionary, with the keys being
                user-defined strings and the values being stateful objects.
                Stateful objects are objects that exposes ``.state_dict()`` and
                ``.load_state_dict()`` methods. Common PyTorch objects such as
                :class:`torch.nn.Module`, :class:`torch.optim.Optimizer`, and
                LR schedulers all qualify as stateful objects.

            path (str): The location to save the snapshot. ``path`` can have a
                URI prefix (e.g. ``s3://``) that specifies a storage backend.
                If no URI prefix is supplied, ``path`` is assumed to be a file
                system location. For distributed snapshot, if ``path`` is
                inconsistent across participating ranks, the value specified by
                rank 0 will be used. For multi-host snapshot, ``path`` needs to
                be a location accessible by all hosts.

                .. note:: ``path`` must **not** point to an existing snapshot.

            pg (ProcessGroup, optional): The process group for the participants
                of :meth:`Snapshot.take`. If none, the default process group will
                be used.

            replicated (List[str], optional): Glob patterns for marking
                checkpoint content as replicated. Matching objects will be deduped
                and load-balanced across ranks.

                .. note:: The replication property is automatically inferred
                    for ``DistributedDataParallel``. Only specify this argument
                    if your model has fully replicated states but does not use
                    ``DistributedDataParallel``.

            storage_options (Dict[str, Any], optional): Additional keyword
                options for the storage plugin to use. See each storage plugin's
                documentation for customizations.

        Returns:
            The newly taken snapshot.
        """
        torch._C._log_api_usage_once("torchsnapshot.Snapshot.take")
        cls._validate_app_state(app_state)

        event_loop = maybe_nested_loop()
        pg_wrapper = PGWrapper(pg=pg)

        unique_id = _generate_random_int64()
        rank = pg_wrapper.get_rank()
        log_event(
            Event(
                name="take",
                metadata={"action": "start", "unique_id": unique_id, "rank": rank},
            )
        )

        path, coalesced_replicated = cls._coalesce_path_and_replicated(
            path=path,
            pg_wrapper=pg_wrapper,
            app_state=app_state,
            replicated=replicated or [],
        )
        storage = url_to_storage_plugin_in_event_loop(
            url_path=path, event_loop=event_loop, storage_options=storage_options
        )
        pending_io_work, metadata = cls._take_impl(
            path=path,
            app_state=app_state,
            replicated=coalesced_replicated,
            pg_wrapper=PGWrapper(pg),
            storage=storage,
            event_loop=event_loop,
            is_async_snapshot=False,
            _custom_tensor_prepare_func=_custom_tensor_prepare_func,
        )
        pending_io_work.sync_complete(event_loop=event_loop)

        # IMPORTANT: commit snapshot metadata only after all ranks complete writing
        pg_wrapper.barrier()
        if pg_wrapper.get_rank() == 0:
            cls._write_snapshot_metadata(
                snapshot_metadata=metadata,
                storage=storage,
                event_loop=event_loop,
            )

        storage.sync_close(event_loop=event_loop)
        event_loop.close()
        snapshot = cls(path=path, pg=pg, storage_options=storage_options)
        snapshot._metadata = metadata

        log_event(
            Event(
                name="take",
                metadata={
                    "action": "end",
                    "unique_id": unique_id,
                    "is_success": True,
                    "rank": rank,
                },
            )
        )
        return snapshot

    @classmethod
    def async_take(
        cls,
        path: str,
        app_state: AppState,
        pg: Optional[dist.ProcessGroup] = None,
        replicated: Optional[List[str]] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        _custom_tensor_prepare_func: Optional[
            Callable[[str, torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> "PendingSnapshot":
        """
        Asynchronously takes a snapshot from the application state.

        This function is identical to :func:`Snapshot.take`, except that it
        returns early and performs as much I/O operations in the background as
        possible, allowing training to resume early.

        Args:
            app_state (Dict[str, Stateful]): Same as the ``app_state`` argument of :func:`Snapshot.take`.
            path (str): Same as the ``path`` argument of :func:`Snapshot.take`.
            pg (ProcessGroup, optional): Same as the ``pg`` argument of :func:`Snapshot.take`.
            replicated (List[str], optional): Same as the ``replicated`` argument of :func:`Snapshot.take`.
            storage_options (Dict[str, Any], optional): Same as the ``storage_options`` argument of :func:`Snapshot.take`.

        Returns:
            A handle to the pending snapshot. The handle has exposes a
            ``.done()`` method for querying the progress and a ``.wait()``
            method for waiting for the snapshot's completion.
        """
        torch._C._log_api_usage_once("torchsnapshot.Snapshot.async_take")
        cls._validate_app_state(app_state)

        event_loop = maybe_nested_loop()
        pg_wrapper = PGWrapper(pg=pg)

        unique_id = _generate_random_int64()
        rank = pg_wrapper.get_rank()
        log_event(
            Event(
                name="async_take",
                metadata={"action": "start", "unique_id": unique_id, "rank": rank},
            )
        )

        path, coalesced_replicated = cls._coalesce_path_and_replicated(
            path=path,
            pg_wrapper=pg_wrapper,
            app_state=app_state,
            replicated=replicated or [],
        )
        storage = url_to_storage_plugin_in_event_loop(
            url_path=path, event_loop=event_loop, storage_options=storage_options
        )

        pending_io_work, metadata = cls._take_impl(
            path=path,
            app_state=app_state,
            replicated=coalesced_replicated,
            pg_wrapper=PGWrapper(pg),
            storage=storage,
            event_loop=event_loop,
            is_async_snapshot=True,
            _custom_tensor_prepare_func=_custom_tensor_prepare_func,
        )

        log_event(
            Event(
                name="async_take",
                metadata={
                    "action": "end_collection",
                    "unique_id": unique_id,
                    "rank": rank,
                },
            )
        )

        # PendingSnapshot is responsible for closing `storage` and `event_loop`
        return PendingSnapshot(
            path=path,
            pending_io_work=pending_io_work,
            pg_wrapper=pg_wrapper,
            metadata=metadata,
            storage=storage,
            event_loop=event_loop,
            storage_options=storage_options,
            unique_id=unique_id,
        )

    def restore(self, app_state: AppState, strict: bool = True) -> None:
        """
        Restores the application state from the snapshot.

        Args:
            app_state (Dict[str, Stateful]): The application state to restore.
                ``app_state`` needs to be either identical to or a subset of the
                ``app_state`` used for :func:`Snapshot.take` when the snapshot was
                taken.
            strict (bool, optional): If ``True``, raises an error if the expected
                state_dict keys in the snapshot do not match the actual keys in
                the :class:`torch.nn.Module`. This only applies to :class:`torch.nn.Module`
                and not other objects being restored in ``app_state``.
        """
        torch._C._log_api_usage_once("torchsnapshot.Snapshot.restore")
        self._validate_app_state(app_state)

        event_loop = maybe_nested_loop()
        pg_wrapper = PGWrapper(self.pg)

        unique_id = _generate_random_int64()
        rank = pg_wrapper.get_rank()
        log_event(
            Event(
                name="restore",
                metadata={"action": "start", "unique_id": unique_id, "rank": rank},
            )
        )

        storage = url_to_storage_plugin_in_event_loop(
            url_path=self.path,
            event_loop=event_loop,
            storage_options=self._storage_options,
        )

        app_state = app_state.copy()
        rng_state_item = self._pop_rng_state(app_state=app_state)

        global_keys = self._gather_keys(
            keys=list(app_state.keys()), pg_wrapper=pg_wrapper
        )
        for key in global_keys:
            self._load_stateful(
                stateful_key=key,
                stateful=app_state.get(key),
                strict=strict,
                storage=storage,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
            pg_wrapper.barrier()

        # Restore the RNG state last to avoid potential side effects.
        if rng_state_item is not None:
            key, stateful = rng_state_item
            self._load_stateful(
                stateful_key=key,
                stateful=stateful,
                strict=strict,
                storage=storage,
                pg=pg_wrapper,
                event_loop=event_loop,
            )
        storage.sync_close(event_loop=event_loop)
        event_loop.close()

        log_event(
            Event(
                name="restore",
                metadata={
                    "action": "end",
                    "unique_id": unique_id,
                    "is_success": True,
                    "rank": rank,
                },
            )
        )

    def read_object(
        self,
        path: str,
        obj_out: Optional[T] = None,
        memory_budget_bytes: Optional[int] = None,
    ) -> T:
        """
        Reads an object from the snapshot's content.

        Args:
            path (str): The path to the target object within the snapshot.
                ``path`` is equivalent to the target object's key in the
                snapshot manifest and can be obtained via
                :meth:`Snapshot.get_manifest`.

            obj_out (Any, optional): When specified, load the object in-place
                into ``obj_out`` if in-place load is supported for the object's
                type. Otherwise, ``obj_out`` is ignored.

                .. note::
                    When the target object is a ``ShardedTensor``, and ``obj_out``
                    is None, will return cpu, full tensor version of the sharded
                    tensor.

            memory_budget_bytes (int, optional): When specified, the read
                operation will keep the temporary memory buffer size below this
                threshold.

        Returns:
            The object read from the snapshot's content.
        """
        torch._C._log_api_usage_once("torchsnapshot.Snapshot.read_object")
        unique_id = _generate_random_int64()
        log_event(
            Event(
                name="read_object", metadata={"action": "start", "unique_id": unique_id}
            )
        )

        # TODO: better message for malformatted path
        rank_str, unranked_path = path.split("/", 1)
        rank = int(rank_str)
        # Transform the manifest such that (1) replicated entries are made
        # available to the rank (2) sharded tensor shards saved by all ranks
        # are made available to the rank. The availability of the entries is
        # determined from the perspective of the rank specified in the path.
        manifest, merged_sd_entries = get_manifest_for_rank(
            metadata=self.metadata, rank=rank
        )

        if unranked_path not in merged_sd_entries and unranked_path not in manifest:
            # TODO: show candidates based on edit distance
            raise RuntimeError(
                f'The supplied path "{path}" does not exist in the snapshot\'s manifest. '
                "Please verify the available paths within the snapshot via `snapshot.get_manifest()`."
            )
        if not isinstance(obj_out, (torch.Tensor, ShardedTensor, DTensor)):
            logger.warning(
                f"`obj_out` is of type {type(obj_out)}, which does not support in-place load. "
                "Its state won't be changed after load. The loaded object will be returned."
            )

        event_loop = maybe_nested_loop()
        pg_wrapper = PGWrapper(self.pg)
        storage = url_to_storage_plugin_in_event_loop(
            url_path=self.path,
            event_loop=event_loop,
            storage_options=self._storage_options,
        )
        entry = merged_sd_entries.get(unranked_path) or manifest[unranked_path]
        if isinstance(entry, PrimitiveEntry):
            return cast(T, entry.get_value())

        read_reqs, fut = prepare_read(
            entry=entry,
            obj_out=obj_out,
            # TODO: find a suitable buffer_size_limit_bytes to enable chunked
            # read even when memory_budget_bytes is not specified, as chunked
            # tensor read allows for pipelining HtoD copy and storage I/O when
            # reading a single tensor.
            buffer_size_limit_bytes=memory_budget_bytes,
        )

        if not is_batching_disabled():
            read_reqs = batch_read_requests(read_reqs=read_reqs)

        sync_execute_read_reqs(
            read_reqs=read_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes
            or _MAX_PER_RANK_MEMORY_BUDGET_BYTES,
            rank=pg_wrapper.get_rank(),
            event_loop=event_loop,
        )
        storage.sync_close(event_loop=event_loop)
        event_loop.close()

        log_event(
            Event(
                name="read_object",
                metadata={"action": "end", "unique_id": unique_id, "is_success": True},
            )
        )

        return fut.obj

    def get_manifest(self) -> Dict[str, Entry]:
        """
        Returns the snapshot manifest.

        Each entry in the dictionary corresponds to an object in the snapshot,
        with the keys being the logical paths to the objects and the values
        being the metadata describing the object. For distributed snapshots,
        the manifest contain entries for objects saved by all ranks.

        Returns:
            The snapshot manifest.
        """
        return copy.deepcopy(self.metadata.manifest)

    @classmethod
    def _take_impl(
        cls,
        path: str,
        app_state: AppState,
        replicated: Set[str],
        pg_wrapper: PGWrapper,
        storage: StoragePlugin,
        event_loop: asyncio.AbstractEventLoop,
        is_async_snapshot: bool,
        _custom_tensor_prepare_func: Optional[
            Callable[[str, torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[PendingIOWork, SnapshotMetadata]:
        app_state = app_state.copy()
        rng_state_item = cls._pop_rng_state(app_state=app_state)
        rng_state_dict = None

        manifest: Manifest = {}
        flattened: Dict[str, Any] = {}

        # Invariant: for the same snapshot, the RNG state is the same after
        # .take() and .restore().
        # This can be achieved by ensuring .take() has no side effect on the
        # RNG state. Since we can't guarantee that the .state_dict() method on
        # stateful objects has no side effect on the RNG state, we retrieve the
        # RNG state before saving other stateful objects, and restore the RNG
        # state after saving other stateful objects.
        if rng_state_item is not None:
            key, stateful = rng_state_item
            rng_state_dict = stateful.state_dict()
            mnfst, fltnd = flatten(rng_state_dict, prefix=key)
            manifest.update(mnfst)
            flattened.update(fltnd)

        # Different ranks can register different sets of stateful objects,
        # whose .state_dict() methods may invoke collectives. To avoid
        # potential interleaving of different collectives, we first gather the
        # global key list, then invoke .state_dict() on stateful objects in
        # order with synchronization.
        # TODO: merge this with coalesce path to save an all_gather call
        global_keys = cls._gather_keys(
            keys=list(app_state.keys()), pg_wrapper=pg_wrapper
        )

        for key in global_keys:
            if key in app_state:
                state_dict = app_state[key].state_dict()
                mnfst, fltnd = flatten(state_dict, prefix=key)
                manifest.update(mnfst)
                flattened.update(fltnd)
            pg_wrapper.barrier()

        # Undo any potential side effects to the RNG state. The rest of this
        # function won't affect the RNG state or execute application code.
        if rng_state_item is not None:
            _, stateful = rng_state_item
            stateful.load_state_dict(cast(Dict[str, torch.Tensor], rng_state_dict))

        replicated_paths = cls._calculate_replicated_entries(
            flattened, replicated, pg_wrapper
        )

        object_entries: Dict[str, Entry] = {}
        logical_path_to_write_reqs: Dict[str, List[WriteReq]] = {}
        primitive_entries: Dict[str, PrimitiveEntry] = {}

        for logical_path, obj in flattened.items():
            entry, wrs = prepare_write(
                obj=obj,
                logical_path=logical_path,
                rank=pg_wrapper.get_rank(),
                replicated=logical_path in replicated_paths,
                is_async_snapshot=is_async_snapshot,
                _tensor_prepare_func=(
                    functools.partial(_custom_tensor_prepare_func, logical_path)
                    if _custom_tensor_prepare_func is not None
                    else None
                ),
            )
            # Primitive entries don't have write requests
            # and don't need to be partitioned
            if isinstance(entry, PrimitiveEntry):
                primitive_entries[logical_path] = entry
            else:
                object_entries[logical_path] = entry
                logical_path_to_write_reqs[logical_path] = wrs

        object_entries, logical_path_to_write_reqs = partition_write_reqs(
            entries=object_entries, write_reqs=logical_path_to_write_reqs, pg=pg_wrapper
        )
        write_reqs: List[WriteReq] = [
            wr for wrs in logical_path_to_write_reqs.values() for wr in wrs
        ]

        if not is_batching_disabled():
            _, write_reqs = batch_write_requests(
                entries=list(object_entries.values()), write_reqs=write_reqs
            )

        all_entries = dict(**primitive_entries, **object_entries)

        manifest.update(all_entries)
        manifest = cls._gather_manifest(manifest=manifest, pg=pg_wrapper)

        memory_budget_bytes = get_process_memory_budget_bytes(pg=pg_wrapper)
        pending_io_work = sync_execute_write_reqs(
            write_reqs=write_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=pg_wrapper.get_rank(),
            event_loop=event_loop,
        )
        metadata = SnapshotMetadata(
            version=torchsnapshot_version,
            world_size=pg_wrapper.get_world_size(),
            manifest=manifest,
        )
        return pending_io_work, metadata

    @staticmethod
    def _calculate_replicated_entries(
        flattened: Dict[str, Any], replicated: Set[str], pg: PGWrapper
    ) -> Set[str]:
        rank = pg.get_rank()
        world_size = pg.get_world_size()
        replicated_paths = []
        for path, val in flattened.items():
            if any(fnmatch.fnmatch(path, p) for p in replicated) and not is_sharded(
                val
            ):
                replicated_paths.append(path)
        # pyre-ignore
        obj_list: List[List[str]] = [None] * world_size
        pg.all_gather_object(obj_list, replicated_paths)

        if rank == 0:
            # A path is only treated as replicated if:
            # (1) The path matches one of the patterns specified in `replicated`
            # (2) The path exists on all ranks
            # (3) The value is not sharded
            path_count = defaultdict(int)
            for paths in obj_list:
                for path in paths:
                    path_count[path] += 1
            replicated_paths = list(
                filter(lambda p: path_count[p] == world_size, replicated_paths)
            )
            replicated_paths_list = [replicated_paths]
        else:
            replicated_paths_list = [[]]
        pg.broadcast_object_list(replicated_paths_list, src=0)
        replicated_paths = replicated_paths_list[0]
        return set(replicated_paths)

    @staticmethod
    def _validate_app_state(app_state: AppState) -> None:
        # performs runtime typechecking that all values are Stateful
        for key, value in app_state.items():
            if not isinstance(value, Stateful):
                value_type = type(value)
                raise TypeError(
                    f"Expected Stateful in app_state for key {key}, got {value_type}."
                )

    # pyre-fixme: inflate returns Dict[Any,Any]
    # Missing return annotation [3]: Return type must be specified as type that does not contain `Any`
    def get_state_dict_for_key(
        self, key: str, replicate_from_rank0: bool = False
    ) -> Dict[Any, Any]:
        """
        Gets the state dict for a selected key in the snapshot.
        This is useful in case you want to get the state dict without loading it to the stateful.

        Args:
            key (str): The key to get the state dict for. Assumes the key was stored as a topline
                key in the snapshot.

            replicate_from_rank0 (bool): Whether to replicate the state dict from rank 0 to all ranks.
                Useful when loading state dict from a checkpoint where world size was smaller, as this
                function would return empty state dict for the new ranks.

        Returns:
            The state dict associated with the key.

        Below is a usage example

        .. code-block:: python

            snapshot = Snapshot.take(path=..., app_state={"stateful_key": module})
            module_state_dict = snapshot.get_state_dict_for_key("stateful_key")
        """
        event_loop = maybe_nested_loop()
        pg = PGWrapper(self.pg)
        rank = pg.get_rank() if not replicate_from_rank0 else 0
        manifest, _ = get_manifest_for_rank(metadata=self.metadata, rank=rank)

        # filter out irrelevant entries from the manifest
        manifest = {k: v for k, v in manifest.items() if k.split("/")[0] == key}

        storage = url_to_storage_plugin_in_event_loop(
            url_path=self.path,
            event_loop=event_loop,
            storage_options=self._storage_options,
        )

        return self._get_state_dict_for_manifest(
            key, manifest, {}, pg, storage, event_loop, replicate_from_rank0
        )

    def _load_stateful(  # noqa
        self,
        stateful_key: str,
        stateful: Optional[Stateful],
        strict: bool,
        storage: StoragePlugin,
        pg: PGWrapper,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        if stateful is None:
            return

        manifest, merged_sd_entries = get_manifest_for_rank(
            metadata=self.metadata, rank=pg.get_rank()
        )

        # In most cases (e.g. when the stateful is an nn.Module), the stateful
        # has already allocated memory for its tensors. Materializing the
        # persisted state dict and invoking .load_state_dict() would result in
        # a memory footprint that is 2x the size of the stateful. We can reduce
        # the memory footprint by exploiting the fact that most .state_dict()
        # implementations return references to the internal tensors. By loading
        # directly into the already allocated tensors and use them to construct
        # a state dict for .load_state_dict(), we can eliminate an extra
        # intermediate copy of the state. Even if the tensors in the state dict
        # are copies of the internal tensors, this approach would not use more
        # memory compared to the baseline.
        _, flattened = flatten(stateful.state_dict(), prefix=stateful_key)
        flattened = {
            k: v
            for k, v in flattened.items()
            # ShardedTensor became a subclass of torch.Tensor since PyTorch
            # 1.13. We can drop the check for ShardedTensor once PyTorch 1.12.1
            # is no longer supported.
            if isinstance(v, (torch.Tensor, ShardedTensor, DTensor))
        }

        handle_sharded_tensor_elasticity(
            manifest=manifest,
            merged_sd_entries=merged_sd_entries,
            tensor_requests=list(flattened.keys()),
        )

        # Build the originally saved state dict and use it to restore the stateful
        state_dict = self._get_state_dict_for_manifest(
            stateful_key, manifest, flattened, pg, storage, event_loop
        )

        if isinstance(stateful, torch.nn.Module):
            stateful.load_state_dict(state_dict, strict=strict)
        else:
            stateful.load_state_dict(state_dict)

    @staticmethod
    # pyre-fixme: inflate returns Dict[Any,Any]
    # Missing return annotation [3]: Return type must be specified as type that does not contain `Any`
    def _get_state_dict_for_manifest(
        stateful_key: str,
        manifest: Manifest,
        flattened: Dict[str, Union[torch.Tensor, ShardedTensor, DTensor]],
        pg: PGWrapper,
        storage: StoragePlugin,
        event_loop: AbstractEventLoop,
        replicate_from_rank0: bool = False,
    ) -> Dict[Any, Any]:
        container_entries = {}
        read_reqs: List[ReadReq] = []
        futs = {}
        for logical_path, entry in manifest.items():
            if is_container_entry(entry):
                container_entries[logical_path] = entry
                continue

            rrs, fut = prepare_read(
                entry=entry,
                obj_out=flattened.get(logical_path),
            )
            read_reqs += rrs
            futs[logical_path] = fut

            # Free memory in case the items is a copy
            if logical_path in flattened:
                del flattened[logical_path]

        if not is_batching_disabled():
            read_reqs = batch_read_requests(read_reqs=read_reqs)

        memory_budget_bytes = get_process_memory_budget_bytes(pg=pg)
        sync_execute_read_reqs(
            read_reqs=read_reqs,
            storage=storage,
            memory_budget_bytes=memory_budget_bytes,
            rank=0 if replicate_from_rank0 else pg.get_rank(),
            event_loop=event_loop,
        )

        # Build the originally saved state dict and use it to restore the stateful
        return inflate(
            manifest=container_entries,
            flattened={k: fut.obj for k, fut in futs.items()},
            prefix=stateful_key,
        )

    @staticmethod
    def _write_snapshot_metadata(
        snapshot_metadata: SnapshotMetadata,
        storage: StoragePlugin,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        write_io = WriteIO(
            path=SNAPSHOT_METADATA_FNAME,
            buf=snapshot_metadata.to_yaml().encode("utf-8"),
        )
        storage.sync_write(write_io=write_io, event_loop=event_loop)

    @staticmethod
    def _read_snapshot_metadata(
        storage: StoragePlugin, event_loop: asyncio.AbstractEventLoop
    ) -> SnapshotMetadata:
        read_io = ReadIO(path=SNAPSHOT_METADATA_FNAME)
        try:
            storage.sync_read(read_io=read_io, event_loop=event_loop)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read {SNAPSHOT_METADATA_FNAME}. "
                "Ensure path to snapshot is correct, "
                "otherwise snapshot is likely incomplete or corrupted."
            ) from e
        yaml_str = read_io.buf.getvalue().decode("utf-8")
        return SnapshotMetadata.from_yaml(yaml_str)

    @classmethod
    def _coalesce_path_and_replicated(
        cls,
        path: str,
        pg_wrapper: PGWrapper,
        app_state: AppState,
        replicated: List[str],
    ) -> Tuple[str, Set[str]]:
        rank = pg_wrapper.get_rank()

        # coalesce path
        # TODO: use a single all_gather for both path and replicated.
        # Only emit a single message for path inconsistency.
        obj_list = [path]
        pg_wrapper.broadcast_object_list(obj_list, src=0)
        if obj_list[0] != path:
            logger.warning(
                f"Rank {rank} specified a path ({path}) "
                f"different from rank 0 ({obj_list[0]}). Using path specified by rank 0."
            )

        # TODO: this should be folded into _calculate_replicated_entries
        # coalesce replicated
        replicated = cls._infer_replicated(replicated, app_state)
        # pyre-ignore[9]
        global_replicated: List[List[str]] = [None] * pg_wrapper.get_world_size()
        pg_wrapper.all_gather_object(global_replicated, replicated)

        coalesced_replicated = cls._coalesce_replicated(
            global_replicated=global_replicated
        )
        if set(replicated) != coalesced_replicated:
            logger.warning(
                f"Rank {rank} specified replicated paths: {set(global_replicated[rank])} "
                f"different from replicated paths verified across all ranks: {set(replicated)}"
            )
        return obj_list[0], coalesced_replicated

    @staticmethod
    def _infer_replicated(replicated: List[str], app_state: AppState) -> List[str]:
        new_replicated = replicated.copy()
        if "**" in new_replicated:
            return new_replicated
        for key, val in app_state.items():
            if isinstance(val, DDP):
                ignored = set(cast(List[str], val.parameters_to_ignore))
                if not ignored:
                    new_replicated.append(os.path.join(key, "**"))
                    continue
                for name, _ in itertools.chain(
                    val.named_parameters(), val.named_buffers()
                ):
                    if name not in ignored:
                        new_replicated.append(os.path.join(key, name))
        return new_replicated

    @staticmethod
    def _coalesce_replicated(global_replicated: List[List[str]]) -> Set[str]:
        # pyre-fixme[6]: For 1st argument expected `(_T1) -> _S` but got `Type[set]`.
        verified_replicated = set.intersection(*map(set, global_replicated))
        return verified_replicated

    @staticmethod
    def _gather_keys(keys: List[str], pg_wrapper: PGWrapper) -> List[str]:
        # pyre-ignore
        gathered_keys: List[List[str]] = [None] * pg_wrapper.get_world_size()
        pg_wrapper.all_gather_object(gathered_keys, keys)
        return sorted(set(itertools.chain.from_iterable(gathered_keys)))

    @staticmethod
    def _pop_rng_state(
        app_state: AppState,
    ) -> Optional[Tuple[str, RNGState]]:
        rng_state_items = {
            key: stateful
            for key, stateful in app_state.items()
            if isinstance(stateful, RNGState)
        }
        if len(rng_state_items) > 1:
            raise RuntimeError(
                "Multiple RNGState objects in app state: "
                f"{list(rng_state_items.keys())}"
            )
        elif len(rng_state_items) == 1:
            key, stateful = list(rng_state_items.items())[0]
            del app_state[key]
            return key, stateful
        else:
            return None

    @staticmethod
    def _gather_manifest(manifest: Dict[str, Entry], pg: PGWrapper) -> Dict[str, Any]:
        # pyre-ignore
        manifests: List[Dict[str, Entry]] = [None] * pg.get_world_size()
        pg.all_gather_object(manifests, manifest)
        manifests = consolidate_replicated_entries(rank_to_entries=manifests)

        global_manifest = {}
        for rank, manifest in enumerate(manifests):
            for logical_path, entry in manifest.items():
                global_manifest[os.path.join(str(rank), logical_path)] = entry
        return global_manifest


class PendingSnapshot:
    DEFAULT_BARRIER_TIMEOUT = timedelta(seconds=1800)

    def __init__(
        self,
        path: str,
        pending_io_work: PendingIOWork,
        pg_wrapper: PGWrapper,
        metadata: SnapshotMetadata,
        storage: StoragePlugin,
        event_loop: asyncio.AbstractEventLoop,
        unique_id: Optional[int],
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = path
        self.pg: Optional[dist.ProcessGroup] = pg_wrapper.pg
        # pyre-ignore
        self.exc_info: Optional[Any] = None
        self._done = False
        self._storage_options = storage_options
        self._unique_id = unique_id

        self.thread = Thread(
            target=self._complete_snapshot,
            kwargs={
                "path": path,
                "rank": pg_wrapper.get_rank(),
                "world_size": pg_wrapper.get_world_size(),
                "pending_io_work": pending_io_work,
                "metadata": metadata,
                "storage": storage,
                "event_loop": event_loop,
                "store": get_or_create_store(pg_wrapper=pg_wrapper),
            },
        )
        self.thread.start()

    def _complete_snapshot(
        self,
        path: str,
        rank: int,
        world_size: int,
        pending_io_work: PendingIOWork,
        metadata: SnapshotMetadata,
        storage: StoragePlugin,
        event_loop: asyncio.AbstractEventLoop,
        store: dist.TCPStore,
    ) -> None:
        # WARNING: do not use any collectives in this method

        succeeded = False
        # Use a dist.Store-based barrier for synchronization so that the
        # snapshot can be committed in the background thread.
        barrier = LinearBarrier(
            prefix=f"torchsnapshot_{path}",
            store=store,
            rank=rank,
            world_size=world_size,
            leader_rank=0,
        )
        try:
            pending_io_work.sync_complete(event_loop)
            barrier.arrive(timeout=self.DEFAULT_BARRIER_TIMEOUT)

            if rank == 0:
                Snapshot._write_snapshot_metadata(
                    snapshot_metadata=metadata,
                    storage=storage,
                    event_loop=event_loop,
                )
            barrier.depart(timeout=self.DEFAULT_BARRIER_TIMEOUT)
            succeeded = True
        except Exception as e:
            barrier.report_error(str(e))
            self.exc_info = sys.exc_info()
            logger.warning(
                f"Encountered exception while taking snapshot asynchronously:\n{e}"
            )
        finally:
            storage.sync_close(event_loop=event_loop)
            event_loop.close()
        self._done = True
        log_event(
            Event(
                name="async_take",
                metadata={
                    "action": "end",
                    "unique_id": self._unique_id,
                    "is_success": succeeded,
                    "rank": rank,
                },
            )
        )

    def wait(self) -> Snapshot:
        self.thread.join()
        if self.exc_info is not None:
            formatted = "".join(traceback.format_exception(*self.exc_info))
            raise RuntimeError(
                f"Encountered exception while taking snapshot asynchronously:\n{formatted}"
            )
        return Snapshot(
            path=self.path, pg=self.pg, storage_options=self._storage_options
        )

    def done(self) -> bool:
        return self._done


def _generate_random_int64() -> int:
    return random.randint(0, 2**63 - 1)
