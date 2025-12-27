#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import timedelta
from typing import Dict, Optional

import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_socket_with_port

from .pg_wrapper import PGWrapper


_DEFAULT_TCP_STORE_TIMEOUT = timedelta(seconds=600)

_pg_to_store: Dict[Optional[dist.ProcessGroup], dist.Store] = {}


def get_or_create_store(pg_wrapper: PGWrapper) -> dist.Store:
    """
    Get or create a dist.Store.

    If a default store is present, return the store. Otherwise, bootstrap a
    store with the input process group.

    Args:
        pg_wrapper: The pg with which to bootstrap a store if a default store
            is not present.

    Returns:
        A dist.Store instance.
    """
    store = None
    if dist.is_initialized():
        store = dist.distributed_c10d._get_default_store()

    if store is not None:
        return store
    else:
        # The default store is only absent when the global process group is
        # initialized with the MPI backend. In this case, we bootstrap a store
        # with the input process group.
        if pg_wrapper.pg in _pg_to_store:
            return _pg_to_store[pg_wrapper.pg]
        store = create_store(pg_wrapper=pg_wrapper)
        _pg_to_store[pg_wrapper.pg] = store
        return store


def create_store(pg_wrapper: PGWrapper) -> dist.Store:
    """
    Bootstrap a dist.Store with a process group.

    Args:
        pg_wrapper: The pg with which to bootstrap a store if a default store
            is not present.

    Returns:
        The bootstrapped dist.Store instance.
    """
    if pg_wrapper.get_rank() == 0:
        # Find a free port
        sock = get_socket_with_port()
        address = sock.getsockname()
        master_addr, master_port = address[0], address[1]
        sock.close()
        # Broadcast master address/port to peers
        obj_list = [master_addr, master_port]
    else:
        # Receive master address/port from the leader rank
        obj_list = [None, None]
    pg_wrapper.broadcast_object_list(obj_list=obj_list, src=0)
    master_addr, master_port = obj_list[0], obj_list[1]

    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=pg_wrapper.get_world_size(),
        is_master=pg_wrapper.get_rank() == 0,
        timeout=_DEFAULT_TCP_STORE_TIMEOUT,
        wait_for_workers=True,
    )
    _pg_to_store[pg_wrapper.pg] = store
    return store


class LinearBarrier:
    """
    A dist.Store-based linear barrier implementation.

    The barrier is performed in two stages:

    arrive - Non-leader ranks notify the leader rank that they've arrived at
        the barrier.

    depart - The leader rank notifies non-leader ranks that it has arrived at
        the barrier.

    The barrier is separated into two stages because this allows the leader
    rank to perform some actions in-between the two stages, with the knowledge
    that all ranks have arrived at the barrier, while holding other ranks in
    the barrier.
    """

    def __init__(
        self,
        prefix: str,
        store: dist.Store,
        rank: int,
        world_size: int,
        leader_rank: int,
    ) -> None:
        self.prefix = prefix
        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.leader_rank = leader_rank
        self.arrived = False
        self.departed = False

    def arrive(self, timeout: timedelta) -> None:
        """
        The first stage of the barrier.

        Args:
            timeout: The timeout for the "arrive" stage.
        """
        if self.arrived:
            raise RuntimeError("Can't call .arrive() multiple times on a barrier.")
        if self.departed:
            raise RuntimeError("Can't call .arrive() on a completed barrier.")
        self.arrived = True

        if self.rank == self.leader_rank:
            peer_keys = [
                self._key(rank=rank)
                for rank in range(self.world_size)
                if rank != self.leader_rank
            ]
            self.store.wait(peer_keys, timeout)
            for key in peer_keys:
                err = self.store.get(key)
                if len(err) != 0:
                    self.report_error(err=str(err))
                    raise RuntimeError(str(err))
        else:
            self.store.set(self._key(rank=self.rank), "")

    def depart(self, timeout: timedelta) -> None:
        """
        The second stage of the barrier.

        Args:
            timeout: The timeout for the "depart" stage.
        """
        if not self.arrived:
            raise RuntimeError(
                "Can't call .depart() before calling .arrive() on a barrier."
            )
        if self.departed:
            raise RuntimeError("Can't call .depart() on a completed barrier.")
        self.arrived = True

        if self.rank == self.leader_rank:
            self.store.set(self._key(self.leader_rank), "")
        else:
            leader_key = self._key(rank=self.leader_rank)
            self.store.wait([leader_key], timeout)
            err = self.store.get(leader_key)
            if len(err) != 0:
                raise RuntimeError(str(err))

    def report_error(self, err: str) -> None:
        """
        Report the error that prevents the current rank from completing the barrier.

        Leader rank - can report error before calling .depart(). The error will
            be received by non-leader ranks in .depart().

        Non-leader rank - can report error before calling .arrive(). The error
            will be received by the leader rank in .arrive() and non-leader ranks
            in .depart().

        Args:
            err: The error to be propagated to peer ranks.
        """
        self.store.set(
            self._key(self.rank), f"Rank {self.rank} encountered error: {err}"
        )

    def _key(self, rank: int) -> str:
        return f"{self.prefix}_{rank}"
