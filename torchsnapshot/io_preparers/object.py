#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import io
import logging
import sys
from concurrent.futures import Executor
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import torch
from torchsnapshot.io_types import (
    BufferConsumer,
    BufferStager,
    BufferType,
    Future,
    ReadReq,
    WriteReq,
)
from torchsnapshot.manifest import ObjectEntry
from torchsnapshot.serialization import Serializer

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class ObjectIOPreparer(Generic[T]):
    @staticmethod
    def prepare_write(
        storage_path: str,
        obj: T,
    ) -> Tuple[ObjectEntry, List[WriteReq]]:
        buffer_stager = ObjectBufferStager(obj=obj)
        return (
            ObjectEntry(
                location=storage_path,
                serializer=Serializer.TORCH_SAVE.value,
                obj_type=type(obj).__module__ + "." + type(obj).__name__,
                replicated=False,
            ),
            [WriteReq(path=storage_path, buffer_stager=buffer_stager)],
        )

    @classmethod
    def prepare_read(
        cls, entry: ObjectEntry, obj_out: Optional[Any]
    ) -> Tuple[List[ReadReq], Future[T]]:
        # obj_out is only used for memory estimation
        fut = Future(obj=obj_out)
        buffer_consumer = ObjectBufferConsumer(fut=fut)
        return [
            ReadReq(
                path=entry.location,
                buffer_consumer=buffer_consumer,
            )
        ], fut


class ObjectBufferStager(BufferStager):
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        buf = io.BytesIO()
        torch.save(self.obj, buf)
        return buf.getvalue()

    def get_staging_cost_bytes(self) -> int:
        # TODO: this is not accurate
        return sys.getsizeof(self.obj)


class ObjectBufferConsumer(BufferConsumer, Generic[T]):
    def __init__(self, fut: Future[T]) -> None:
        self.fut = fut

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        obj: T = torch.load(io.BytesIO(buf), map_location=map_location)
        self.fut.obj = obj

    def get_consuming_cost_bytes(self) -> int:
        return sys.getsizeof(self.fut.obj)
