#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import io
import logging
import random
from pathlib import Path

import pytest
import torch
from torchsnapshot import Snapshot, StateDict
from torchsnapshot.io_types import ReadIO, WriteIO
from torchsnapshot.storage_plugins.fs import FSStoragePlugin

logger: logging.Logger = logging.getLogger(__name__)

_TENSOR_SZ = int(1_000_000 / 4)


def test_fs_read_write_via_snapshot(tmp_path: Path) -> None:
    tensor = torch.rand((_TENSOR_SZ,))
    app_state = {"state": StateDict(tensor=tensor)}
    snapshot = Snapshot.take(path=str(tmp_path), app_state=app_state)

    app_state["state"]["tensor"] = torch.rand((_TENSOR_SZ,))
    assert not torch.allclose(tensor, app_state["state"]["tensor"])

    snapshot.restore(app_state)
    assert torch.allclose(tensor, app_state["state"]["tensor"])


@pytest.mark.asyncio
async def test_fs_write_read_delete(tmp_path: Path) -> None:
    plugin = FSStoragePlugin(root=str(tmp_path))

    tensor = torch.rand((_TENSOR_SZ,))
    buf = io.BytesIO()
    torch.save(tensor, buf)
    write_io = WriteIO(path="tensor", buf=buf.getbuffer())

    await plugin.write(write_io=write_io)

    read_io = ReadIO(path="tensor")
    await plugin.read(read_io=read_io)
    loaded = torch.load(read_io.buf)
    assert torch.allclose(tensor, loaded)

    await plugin.delete(path="tensor")
    await plugin.close()


@pytest.mark.asyncio
async def test_fs_ranged_read(tmp_path: Path) -> None:
    plugin = FSStoragePlugin(root=str(tmp_path))

    buf = bytes(random.getrandbits(8) for _ in range(2000))
    write_io = WriteIO(path="rand_bytes", buf=memoryview(buf))

    await plugin.write(write_io=write_io)

    read_io = ReadIO(path="rand_bytes", byte_range=(100, 200))
    await plugin.read(read_io=read_io)
    assert len(read_io.buf.getvalue()) == 100
    assert read_io.buf.getvalue(), buf[100:200]

    await plugin.close()
