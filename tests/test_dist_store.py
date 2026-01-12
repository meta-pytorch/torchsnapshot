#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from datetime import timedelta

import torch.distributed as dist
import torch.distributed.launcher as pet
from torchsnapshot.dist_store import create_store, get_or_create_store, LinearBarrier
from torchsnapshot.pg_wrapper import PGWrapper
from torchsnapshot.test_utils import get_pet_launch_config


class DistStoreTest(unittest.TestCase):
    @staticmethod
    def _test_create_store(init_pg: bool) -> None:
        if init_pg:
            dist.init_process_group(backend="gloo")
        pg_wrapper = PGWrapper(pg=dist.group.WORLD)
        store = create_store(pg_wrapper=pg_wrapper)

        if pg_wrapper.get_rank() == 0:
            store.set("foo", "bar")
        else:
            unittest.TestCase().assertEqual(store.get("foo"), b"bar")

    def test_create_store(self) -> None:
        for nproc in [1, 2, 4]:
            for init_pg in [True, False]:
                lc = get_pet_launch_config(nproc=nproc)
                pet.elastic_launch(lc, entrypoint=self._test_create_store)(init_pg)

    @staticmethod
    def _test_linear_barrier() -> None:
        pg_wrapper = PGWrapper(pg=dist.group.WORLD)
        store = get_or_create_store(pg_wrapper=pg_wrapper)

        barrier = LinearBarrier(
            prefix="foo",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        barrier.arrive(timeout=timedelta(seconds=5))
        barrier.depart(timeout=timedelta(seconds=5))

    def test_linear_barrier(self) -> None:
        for nproc in [1, 2, 4]:
            lc = get_pet_launch_config(nproc=nproc)
            pet.elastic_launch(lc, entrypoint=self._test_linear_barrier)()

    @staticmethod
    def _test_linear_barrier_timeout() -> None:
        dist.init_process_group(backend="gloo")
        pg_wrapper = PGWrapper(pg=dist.group.WORLD)
        rank = pg_wrapper.get_rank()
        store = get_or_create_store(pg_wrapper=pg_wrapper)
        tc = unittest.TestCase()

        dist.barrier()

        barrier = LinearBarrier(
            prefix="foo",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # If a non-leader rank doesn't arrive at the barrier, the leader rank
        # should timeout in .arrive() and other non-leader ranks should timeout
        # in .depart().
        if rank == 0:
            with tc.assertRaisesRegex(RuntimeError, "timeout"):
                barrier.arrive(timeout=timedelta(seconds=5))
        elif rank == 1:
            pass
        else:
            barrier.arrive(timeout=timedelta(seconds=5))
            with tc.assertRaisesRegex(RuntimeError, "timeout"):
                barrier.depart(timeout=timedelta(seconds=5))

        dist.barrier()

        barrier = LinearBarrier(
            prefix="bar",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # If the leader rank doesn't arrive at the barrier, non-leader ranks
        # should timeout in .depart().
        if rank == 0:
            pass
        else:
            barrier.arrive(timeout=timedelta(seconds=5))
            with tc.assertRaisesRegex(RuntimeError, "timeout"):
                barrier.depart(timeout=timedelta(seconds=5))

        dist.barrier()

        barrier = LinearBarrier(
            prefix="baz",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # Sanity check: the store should still be healthy
        barrier.arrive(timeout=timedelta(seconds=5))
        barrier.depart(timeout=timedelta(seconds=5))

    def test_linear_barrier_timeout(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_linear_barrier_timeout)()

    @staticmethod
    def _test_linear_barrier_error() -> None:
        dist.init_process_group(backend="gloo")
        pg_wrapper = PGWrapper(pg=dist.group.WORLD)
        rank = pg_wrapper.get_rank()
        store = get_or_create_store(pg_wrapper=pg_wrapper)
        tc = unittest.TestCase()

        dist.barrier()

        barrier = LinearBarrier(
            prefix="foo",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # If the leader rank reports error before arriving at the barrier,
        # non-leader ranks should receive the error in .depart().
        if rank == 0:
            barrier.report_error("sorry")
        else:
            barrier.arrive(timeout=timedelta(seconds=5))
            with tc.assertRaisesRegex(RuntimeError, "sorry"):
                barrier.depart(timeout=timedelta(seconds=5))

        barrier = LinearBarrier(
            prefix="bar",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # If the leader rank reports error before departing the barrier,
        # non-leader ranks should receive the error in .depart().
        if rank == 0:
            barrier.arrive(timeout=timedelta(seconds=5))
            barrier.report_error("sorry")
        else:
            barrier.arrive(timeout=timedelta(seconds=5))
            with tc.assertRaisesRegex(RuntimeError, "sorry"):
                barrier.depart(timeout=timedelta(seconds=5))

        barrier = LinearBarrier(
            prefix="baz",
            store=store,
            rank=pg_wrapper.get_rank(),
            world_size=pg_wrapper.get_world_size(),
            leader_rank=0,
        )
        # If a non-leader rank reports error before arriving at the barrier,
        # the leader rank should received the error in .arrive(), other
        # non-leader ranks should receive the error in .depart().
        if rank == 0:
            with tc.assertRaisesRegex(RuntimeError, "sorry"):
                barrier.arrive(timeout=timedelta(seconds=5))
        elif rank == 1:
            barrier.report_error("sorry")
        else:
            barrier.arrive(timeout=timedelta(seconds=5))
            with tc.assertRaisesRegex(RuntimeError, "sorry"):
                barrier.depart(timeout=timedelta(seconds=5))

    def test_linear_barrier_error(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_linear_barrier_error)()
