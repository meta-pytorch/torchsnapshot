#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2, 3]: Allow `Any` in type annotations

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torchsnapshot.dtensor_utils import is_sharded
from torchsnapshot.io_preparers.dtensor import DTensorIOPreparer

from .io_preparers.chunked_tensor import Chunk, ChunkedTensorIOPreparer
from .io_preparers.object import (
    ObjectBufferConsumer,
    ObjectBufferStager,
    ObjectIOPreparer,
)
from .io_preparers.sharded_tensor import ShardedTensorIOPreparer
from .io_preparers.tensor import (
    tensor_copy,
    TensorBufferConsumer,
    TensorBufferStager,
    TensorIOPreparer,
)
from .io_types import Future, ReadReq, WriteReq
from .knobs import get_max_chunk_size_bytes
from .manifest import (
    ChunkedTensorEntry,
    DTensorEntry,
    Entry,
    ObjectEntry,
    PrimitiveEntry,
    ShardedTensorEntry,
    TensorEntry,
)

logger: logging.Logger = logging.getLogger(__name__)


def get_storage_path(obj: Any, logical_path: str, rank: int, replicated: bool) -> str:
    sharded = is_sharded(obj)
    if sharded and replicated:
        return os.path.join("replicated_sharded", logical_path)
    elif sharded and not replicated:
        return os.path.join("sharded", logical_path)
    elif not sharded and replicated:
        return os.path.join("replicated", logical_path)
    else:
        return os.path.join(str(rank), logical_path)


class PrimitivePreparer:
    @staticmethod
    def should_inline(obj: Any) -> bool:
        type_name = type(obj).__name__
        if type_name not in PrimitiveEntry.supported_types:
            return False
        # TODO for long str/bytes, return False to fall back to ObjectEntry
        return True

    @staticmethod
    def prepare_write(obj: Any) -> PrimitiveEntry:
        return PrimitiveEntry.from_object(obj)

    @staticmethod
    def prepare_read(entry: PrimitiveEntry) -> Tuple[List[ReadReq], Future[Any]]:
        return [], Future(obj=entry.get_value())


def prepare_write(
    obj: Any,
    logical_path: str,
    rank: int,
    replicated: bool,
    is_async_snapshot: bool = False,
    _tensor_prepare_func: Optional[Callable[[torch.Tensor, bool], torch.Tensor]] = None,
) -> Tuple[Entry, List[WriteReq]]:
    """
    Prepare write for an object.

    Args:
        obj: The object to save.
        logical_path: The logical path of the object.
        rank: The rank of the current process.
        replicated: Whether the object is replicated.
        is_async_snapshot (bool): whether or not the write request is from async_take
        _tensor_prepare_func (Optional[Callable[[torch.Tensor, bool], torch.Tensor]]): custom transform to apply
            to tensor before staging it in buffer.

    Returns:
        The class::`Entry` describing the object, and a list of
        class::`WriteReq` for persisting the object.
    """
    if PrimitivePreparer.should_inline(obj):
        entry = PrimitivePreparer.prepare_write(obj)
        entry.replicated = replicated
        return entry, []

    storage_path = get_storage_path(obj, logical_path, rank, replicated)
    if isinstance(obj, ShardedTensor):
        return ShardedTensorIOPreparer.prepare_write(
            storage_path=storage_path,
            obj=obj,
            is_async_snapshot=is_async_snapshot,
            _tensor_prepare_func=_tensor_prepare_func,
        )
    elif isinstance(obj, DTensor):
        return DTensorIOPreparer.prepare_write(
            storage_path=storage_path,
            obj=obj,
            is_async_snapshot=is_async_snapshot,
            _tensor_prepare_func=_tensor_prepare_func,
        )
    elif isinstance(obj, torch.Tensor):
        if obj.nelement() * obj.element_size() > get_max_chunk_size_bytes():
            chunking_instruction = ChunkedTensorIOPreparer.chunk_tensor(obj)
            entry, obj_write_req = ChunkedTensorIOPreparer.prepare_write(
                storage_path=storage_path,
                tensor=obj,
                chunking_instruction=chunking_instruction,
                is_async_snapshot=is_async_snapshot,
                _tensor_prepare_func=_tensor_prepare_func,
            )
        else:
            entry, obj_write_req = TensorIOPreparer.prepare_write(
                storage_path=storage_path,
                tensor=obj,
                is_async_snapshot=is_async_snapshot,
                _tensor_prepare_func=_tensor_prepare_func,
            )
    else:
        entry, obj_write_req = ObjectIOPreparer.prepare_write(storage_path, obj)

    entry.replicated = replicated
    return entry, obj_write_req


def prepare_read(
    entry: Entry,
    obj_out: Optional[Any] = None,
    buffer_size_limit_bytes: Optional[int] = None,
) -> Tuple[List[ReadReq], Future[Any]]:
    """
    Prepare read for an object.

    Args:
        entry: The entry describing the object.
        obj: The object to load.

    Returns:
        A list of class::`ReadReq` for loading the object.
    """
    if isinstance(entry, ShardedTensorEntry):
        return ShardedTensorIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, ChunkedTensorEntry):
        return ChunkedTensorIOPreparer.prepare_read(
            entry, obj_out, buffer_size_limit_bytes=buffer_size_limit_bytes
        )
    elif isinstance(entry, DTensorEntry):
        return DTensorIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, TensorEntry):
        return TensorIOPreparer.prepare_read(
            entry, obj_out, buffer_size_limit_bytes=buffer_size_limit_bytes
        )
    elif isinstance(entry, ObjectEntry):
        return ObjectIOPreparer.prepare_read(entry, obj_out)
    elif isinstance(entry, PrimitiveEntry):
        return PrimitivePreparer.prepare_read(entry)
    else:
        raise Exception(f"Unsupported entry type: {entry} ({entry.type}).")


__all__ = [
    "Chunk",
    "ObjectBufferConsumer",
    "ObjectBufferStager",
    "TensorBufferConsumer",
    "TensorBufferStager",
    "tensor_copy",
]
