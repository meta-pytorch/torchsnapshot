#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import os
import pathlib
from typing import Any, Dict, Optional, Set

import aiofiles
import aiofiles.os
from torchsnapshot.io_types import ReadIO, StoragePlugin, WriteIO


class FSStoragePlugin(StoragePlugin):
    def __init__(
        self, root: str, storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        self.root = root
        self._dir_cache: Set[pathlib.Path] = set()

    async def write(self, write_io: WriteIO) -> None:
        path = os.path.join(self.root, write_io.path)

        dir_path = pathlib.Path(path).parent
        if dir_path not in self._dir_cache:
            dir_path.mkdir(parents=True, exist_ok=True)
            self._dir_cache.add(dir_path)

        async with aiofiles.open(path, "wb+") as f:
            # pyre-ignore: memoryview is actually supported
            await f.write(write_io.buf)

    async def read(self, read_io: ReadIO) -> None:
        path = os.path.join(self.root, read_io.path)
        byte_range = read_io.byte_range

        async with aiofiles.open(path, "rb") as f:
            if byte_range is None:
                read_io.buf = io.BytesIO(await f.read())
            else:
                offset = byte_range[0]
                size = byte_range[1] - byte_range[0]
                await f.seek(offset)
                read_io.buf = io.BytesIO(await f.read(size))

    async def delete(self, path: str) -> None:
        path = os.path.join(self.root, path)
        await aiofiles.os.remove(path)

    async def delete_dir(self, path: str) -> None:
        path = os.path.join(self.root, path)
        await aiofiles.os.rmdir(path)

    async def close(self) -> None:
        pass
