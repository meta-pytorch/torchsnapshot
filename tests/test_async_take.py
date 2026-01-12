#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot
from torchsnapshot.io_types import WriteIO
from torchsnapshot.snapshot import SNAPSHOT_METADATA_FNAME
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import get_pet_launch_config


class SlowFSStoragePlugin(FSStoragePlugin):
    async def write(self, write_io: WriteIO) -> None:
        await asyncio.sleep(5)
        await super().write(write_io=write_io)


class FaultyFSStoragePlugin(FSStoragePlugin):
    async def write(self, write_io: WriteIO) -> None:
        await asyncio.sleep(5)
        if dist.get_world_size() == 1 or dist.get_rank() == 1:
            raise Exception("sorry")
        else:
            await super().write(write_io=write_io)


class AsyncTakeTest(unittest.TestCase):
    @staticmethod
    def _test_async_take_with_error(path: str) -> None:
        tc = unittest.TestCase()

        dist.init_process_group(backend="gloo")
        with patch(
            "torchsnapshot.storage_plugin.FSStoragePlugin", FaultyFSStoragePlugin
        ):
            future = torchsnapshot.Snapshot.async_take(
                path, {"foo": torch.nn.Linear(128, 64)}
            )
        tc.assertFalse(future.done())
        with tc.assertRaisesRegex(RuntimeError, "sorry"):
            future.wait()

    def test_async_take_with_error(self) -> None:
        for nproc in [2, 4]:
            with tempfile.TemporaryDirectory() as path:
                lc = get_pet_launch_config(nproc=nproc)
                pet.elastic_launch(lc, entrypoint=self._test_async_take_with_error)(
                    path
                )
                metadata_path = os.path.join(path, SNAPSHOT_METADATA_FNAME)
                self.assertFalse(os.path.isfile(metadata_path))

    @staticmethod
    def _test_unwaited_async_take(path: str) -> None:
        tc = unittest.TestCase()

        dist.init_process_group(backend="gloo")
        with patch("torchsnapshot.storage_plugin.FSStoragePlugin", SlowFSStoragePlugin):
            future = torchsnapshot.Snapshot.async_take(
                path, {"foo": torch.nn.Linear(128, 64)}
            )
        tc.assertFalse(future.done())

    # In Python3.8, an unwaited async snapshot can complete during interpreter shutdown.
    # In Python3.9, the follow exception would occur:
    #     RuntimeError: cannot schedule new futures after interpreter shutdown
    #
    # TODO: if it's not possible to allow unwaited async snapshot in Python3.9,
    # we may need to require users to always explicitly wait for async snapshots.
    @unittest.skip(
        "Skipping due to inconsistent behavior between Python3.8 and Python3.9"
    )
    def test_unwaited_async_take(self) -> None:
        for nproc in [1, 2, 4]:
            with tempfile.TemporaryDirectory() as path:
                lc = get_pet_launch_config(nproc=nproc)
                pet.elastic_launch(lc, entrypoint=self._test_unwaited_async_take)(path)
                metadata_path = os.path.join(path, SNAPSHOT_METADATA_FNAME)
                self.assertTrue(os.path.isfile(metadata_path))

    @staticmethod
    def _test_unwaited_async_take_with_error(path: str) -> None:
        tc = unittest.TestCase()

        dist.init_process_group(backend="gloo")
        with patch(
            "torchsnapshot.storage_plugin.FSStoragePlugin", FaultyFSStoragePlugin
        ):
            future = torchsnapshot.Snapshot.async_take(
                path, {"foo": torch.nn.Linear(128, 64)}
            )
        tc.assertFalse(future.done())

    def test_unwaited_async_take_with_error(self) -> None:
        for nproc in [1, 2, 4]:
            with tempfile.TemporaryDirectory() as path:
                lc = get_pet_launch_config(nproc=nproc)
                pet.elastic_launch(
                    lc, entrypoint=self._test_unwaited_async_take_with_error
                )(path)
                metadata_path = os.path.join(path, SNAPSHOT_METADATA_FNAME)
                self.assertFalse(os.path.isfile(metadata_path))
