#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import asyncio
import io
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Generic, Optional, Tuple, TypeVar, Union

from .asyncio_utils import maybe_nested_loop


BufferType = Union[bytes, memoryview]


class BufferStager:
    @abc.abstractmethod
    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        pass

    @abc.abstractmethod
    def get_staging_cost_bytes(self) -> int:
        pass


@dataclass
class WriteReq:
    path: str
    buffer_stager: BufferStager


class BufferConsumer:
    @abc.abstractmethod
    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def get_consuming_cost_bytes(self) -> int:
        pass


@dataclass
class ReadReq:
    path: str
    buffer_consumer: BufferConsumer
    byte_range: Optional[Tuple[int, int]] = None


T = TypeVar("T")


@dataclass
class Future(Generic[T]):
    obj: Optional[T] = None


@dataclass
class WriteIO:
    path: str
    buf: BufferType


@dataclass
class ReadIO:
    path: str
    buf: io.BytesIO = field(default_factory=io.BytesIO)
    byte_range: Optional[Tuple[int, int]] = None


class StoragePlugin(abc.ABC):
    @abc.abstractmethod
    async def write(self, write_io: WriteIO) -> None:
        pass

    @abc.abstractmethod
    async def read(self, read_io: ReadIO) -> None:
        pass

    @abc.abstractmethod
    async def delete(self, path: str) -> None:
        pass

    @abc.abstractmethod
    async def delete_dir(self, path: str) -> None:
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        pass

    def sync_write(
        self, write_io: WriteIO, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = maybe_nested_loop()
        event_loop.run_until_complete(self.write(write_io=write_io))

    def sync_read(
        self, read_io: ReadIO, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = maybe_nested_loop()
        event_loop.run_until_complete(self.read(read_io=read_io))

    def sync_close(
        self, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if event_loop is None:
            event_loop = maybe_nested_loop()
        event_loop.run_until_complete(self.close())
