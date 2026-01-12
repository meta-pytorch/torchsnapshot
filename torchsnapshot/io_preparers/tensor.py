#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import asyncio
import logging
import math
from concurrent.futures import Executor
from functools import reduce
from operator import mul
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import torch
from torchsnapshot.io_types import (
    BufferConsumer,
    BufferStager,
    BufferType,
    Future,
    ReadReq,
    WriteReq,
)
from torchsnapshot.manifest import ChunkedTensorEntry, TensorEntry
from torchsnapshot.serialization import (
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_element_size,
    dtype_to_string,
    Serializer,
    string_to_dtype,
    SUPPORTED_QUANTIZED_DTYPES,
    tensor_as_memoryview,
    tensor_from_memoryview,
    torch_load_from_bytes,
    torch_save_as_bytes,
)
from torchsnapshot.uvm_tensor import is_uvm_tensor, uvm_to_cpu

logger: logging.Logger = logging.getLogger(__name__)


class TensorIOPreparer:
    @staticmethod
    def prepare_write(
        storage_path: str,
        tensor: torch.Tensor,
        is_async_snapshot: bool = False,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[TensorEntry, List[WriteReq]]:
        if not _tensor_prepare_func:
            proc_tensor = tensor
        else:
            proc_tensor = _tensor_prepare_func(tensor, True)  # tracing=True

        if proc_tensor.shape != tensor.shape:
            raise RuntimeError(
                "_tensor_prepare_func shouldn't change the tensor's shape "
                f"(changed from {tensor.shape} to {proc_tensor.shape})."
            )

        if proc_tensor.dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
            serializer = Serializer.BUFFER_PROTOCOL.value
        else:
            serializer = Serializer.TORCH_SAVE.value

        entry = TensorEntry(
            location=storage_path,
            serializer=serializer,
            dtype=dtype_to_string(proc_tensor.dtype),
            shape=list(proc_tensor.shape),
            replicated=False,
        )
        # stage the actual tensor, not processed tensor
        buffer_stager = TensorBufferStager(
            tensor=tensor,
            entry=entry,
            is_async_snapshot=is_async_snapshot,
            _tensor_prepare_func=_tensor_prepare_func,
        )
        return entry, [WriteReq(path=storage_path, buffer_stager=buffer_stager)]

    @classmethod
    def prepare_read(
        cls,
        entry: TensorEntry,
        tensor_out: Optional[torch.Tensor] = None,
        buffer_size_limit_bytes: Optional[int] = None,
    ) -> Tuple[List[ReadReq], Future[torch.Tensor]]:
        # TODO: When the output tensor is a CPU tensor, we should directly load
        # into its storage buffer. This is an important optimization because:
        # - We eliminate an allocation and a copy
        # - With the extra allocation, the I/O concurrency will be severely
        # reduced by the scheduler in a memory constrained environment
        if tensor_out is None or not cls.can_load_inplace(entry=entry, obj=tensor_out):
            tensor_out = cls.empty_tensor_from_entry(entry)

        if (
            buffer_size_limit_bytes is not None
            and entry.serializer == Serializer.BUFFER_PROTOCOL.value
        ):
            return cls.prepare_read_tiled(
                entry=entry,
                tensor_out=tensor_out,
                buffer_size_limit_bytes=buffer_size_limit_bytes,
            )

        buffer_consumer = TensorBufferConsumer(
            tensor=tensor_out,
            entry=entry,
        )
        return [
            ReadReq(
                path=entry.location,
                byte_range=entry.byte_range_tuple,
                buffer_consumer=buffer_consumer,
            )
        ], Future(obj=tensor_out)

    @classmethod
    def prepare_read_tiled(
        cls,
        entry: TensorEntry,
        tensor_out: torch.Tensor,
        buffer_size_limit_bytes: int,
    ) -> Tuple[List[ReadReq], Future[torch.Tensor]]:
        num_chunks = math.ceil(
            cls.get_tensor_size_from_entry(entry) / buffer_size_limit_bytes
        )
        # Try to flatten the tensor without copying to achieve better chunking granularity.
        # This is only possible if the tensor satisfies the contiguity condition described in:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view
        try:
            tensor_out = tensor_out.view(-1)
        except RuntimeError:
            pass
        chunks = torch.chunk(tensor_out, chunks=num_chunks, dim=0)
        element_size = dtype_to_element_size(string_to_dtype(entry.dtype))

        read_reqs = []
        offset = 0
        for chunk in chunks:
            chunk_sz_bytes = chunk.nelement() * element_size
            buffer_consumer = TensorBufferConsumer(
                tensor=chunk,
                entry=TensorEntry(
                    location=entry.location,
                    serializer=entry.serializer,
                    dtype=entry.dtype,
                    shape=list(chunk.shape),
                    replicated=entry.replicated,
                ),
            )
            byte_range = entry.byte_range
            if byte_range is None:
                byte_range = (
                    offset,
                    offset + chunk_sz_bytes,
                )
            else:
                byte_range = (
                    byte_range[0] + offset,
                    byte_range[0] + offset + chunk_sz_bytes,
                )
            read_reqs.append(
                ReadReq(
                    path=entry.location,
                    byte_range=byte_range,
                    buffer_consumer=buffer_consumer,
                )
            )
            offset += chunk_sz_bytes
        return read_reqs, Future(obj=tensor_out)

    @staticmethod
    def get_tensor_size_from_entry(entry: TensorEntry) -> int:
        dtype = string_to_dtype(entry.dtype)
        element_size = dtype_to_element_size(dtype)
        n_element = reduce(mul, entry.shape, 1)
        return element_size * n_element

    @staticmethod
    def can_load_inplace(
        entry: Union[TensorEntry, ChunkedTensorEntry], obj: Any
    ) -> bool:
        if obj is None or not isinstance(obj, torch.Tensor):
            return False
        if string_to_dtype(entry.dtype) == obj.dtype and entry.shape == list(obj.shape):
            return True
        return False

    @staticmethod
    def empty_tensor_from_entry(
        entry: Union[TensorEntry, ChunkedTensorEntry],
    ) -> torch.Tensor:
        if entry.dtype in SUPPORTED_QUANTIZED_DTYPES:
            # TODO: we can't allocate empty quantized tensors because we don't
            # know the scale(s) and zero point(s) before loading the tensor.
            raise RuntimeError(
                "Allocating an empty quantized tensor is not supported yet."
            )
        dtype = string_to_dtype(entry.dtype)
        return torch.empty(entry.shape, dtype=dtype)


_Arg = TypeVar("_Arg")
_Ret = TypeVar("_Ret")


async def _run_in_executor(
    executor: Optional[Executor], func: Callable[[_Arg], _Ret], *args: _Arg
) -> _Ret:
    if executor is not None:
        return await asyncio.get_running_loop().run_in_executor(executor, func, *args)
    else:
        return func(*args)


class TensorBufferStager(BufferStager):
    def __init__(
        self,
        tensor: torch.Tensor,
        entry: TensorEntry,
        is_async_snapshot: bool,
        _tensor_prepare_func: Optional[Callable[[torch.Tensor, bool], torch.Tensor]],
    ) -> None:
        self.tensor = tensor
        self.entry = entry
        self.is_async_snapshot = is_async_snapshot
        self._tensor_prepare_func = _tensor_prepare_func

    async def stage_buffer(self, executor: Optional[Executor] = None) -> BufferType:
        is_tensor_custom_prepared = False
        if self._tensor_prepare_func is not None:
            tensor = self._tensor_prepare_func(self.tensor, False)  # tracing=False
            # If the custom prepared tensor is different from the original
            # tensor and is a CPU tensor, don't copy it.
            if tensor.storage() != self.tensor.storage():
                is_tensor_custom_prepared = True

        if self.tensor.is_cuda:
            # It would be nice to copy from GPU via DMA. However, it is very
            # difficult to figure out the safe amount of page-locked memory
            # that we can use. For now, we'll resort to a thread pool for
            # concurrent DtoH copy (with GIL released).
            cpu_tensor = await _run_in_executor(
                executor, tensor_to_cpu, self.tensor.detach()
            )
        else:
            cpu_tensor = self.tensor
            if is_uvm_tensor(cpu_tensor):
                cpu_tensor = await _run_in_executor(
                    executor, uvm_to_cpu, cpu_tensor.detach()
                )
            if not is_tensor_custom_prepared and self._should_copy_cpu_tensor():
                cpu_tensor = cpu_tensor.clone()

        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            return torch_save_as_bytes(cpu_tensor)
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_as_memoryview(cpu_tensor)
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")

    def get_staging_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")

    def _should_copy_cpu_tensor(self) -> bool:
        if self.entry.serializer == Serializer.BUFFER_PROTOCOL and (
            self.is_async_snapshot or not self.tensor.is_contiguous()
        ):
            # During async snapshot, it's not safe to use
            # tensor_as_memoryview() on the original CPU tensor, as its data
            # may change before the snapshot complete. Thus we make a copy.
            #
            # If the CPU tensor is non-contiguous, tensor_as_memoryview() will
            # make a copy. We might as well do it here.
            return True
        if (
            self.entry.serializer == Serializer.TORCH_SAVE
            and self.tensor.nelement() != self.tensor.storage().size()
        ):
            # When saving a tensor view, torch.save() saves the underlying
            # storage backing the view. When the underlying storage is larger
            # than the view, it makes sense to make a copy of the view before
            # saving it.
            #
            # TODO: when the view maps to a contiguous region within its
            # storage, we can create a new tensor from the slicing the untyped
            # storage to avoid the copy.
            return True
        return False


class TensorBufferConsumer(BufferConsumer):
    def __init__(
        self,
        tensor: torch.Tensor,
        entry: TensorEntry,
    ) -> None:
        self.tensor = tensor
        self.entry = entry

    @staticmethod
    def deserialize_tensor(buf: bytes, entry: TensorEntry) -> torch.Tensor:
        if entry.serializer == Serializer.TORCH_SAVE.value:
            return torch_load_from_bytes(buf)
        elif entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            dtype = string_to_dtype(entry.dtype)
            return tensor_from_memoryview(
                memoryview(buf), dtype=dtype, shape=entry.shape
            )
        else:
            raise ValueError(f"Unrecognized serializer: {entry.serializer}.")

    async def consume_buffer(
        self, buf: bytes, executor: Optional[Executor] = None
    ) -> None:
        loaded = self.deserialize_tensor(buf=buf, entry=self.entry)
        if executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                executor, tensor_copy, self.tensor, loaded
            )
        else:
            tensor_copy(self.tensor, loaded)

    def get_consuming_cost_bytes(self) -> int:
        tensor_sz_bytes = TensorIOPreparer.get_tensor_size_from_entry(self.entry)
        if self.entry.serializer == Serializer.TORCH_SAVE.value:
            # The peak memory footprint of torch.load is 2x the tensor size
            return tensor_sz_bytes * 2
        elif self.entry.serializer == Serializer.BUFFER_PROTOCOL.value:
            return tensor_sz_bytes
        else:
            raise ValueError(f"Unrecognized serializer: {self.entry.serializer}.")


@torch.jit.script
def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cpu")  # pragma: no cover


@torch.jit.script
def _tensor_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.detach().copy_(src)  # pragma: no cover


def _q_params_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.qscheme() != rhs.qscheme():
        return False
    if lhs.qscheme() == torch.per_tensor_affine:
        return (
            lhs.q_scale() == rhs.q_scale() and lhs.q_zero_point() == rhs.q_zero_point()
        )
    elif lhs.qscheme() == torch.per_channel_affine:
        return torch.allclose(
            lhs.q_per_channel_scales(), lhs.q_per_channel_scales()
        ) and torch.allclose(
            lhs.q_per_channel_zero_points(), lhs.q_per_channel_zero_points()
        )
    else:
        raise RuntimeError(f"Unrecognized qscheme {lhs.qscheme()}")


@torch.jit.script
def _tensor_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.dequantize()  # pragma: no cover


def tensor_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    # When both dst and src are quantized tensors, .copy_() works as follows:
    # - src storage -> dst storage
    # - src qscheme -> dst qscheme
    #
    # We need to take special care when dst is a view of a larger tensor and
    # the quantization schemes of src and dst are different, because the new
    # qscheme will only be reflected on dest, which is a view object, but not
    # on the original tensor. Directly calling .copy_() in this case will cause
    # a region of the larger tensor's storage contain data that does not match
    # the larger tensor's qscheme.

    if (
        src.is_quantized
        and (
            not dst.is_quantized  # Copying from quantized Tensor to non-quantized Tensor is not allowed
            or dst.qscheme()
            != src.qscheme()  # Quantized copy only works with same qscheme
            or dst.dtype != src.dtype  # Quantized copy requires matching dtypes
            or (dst._is_view() and not _q_params_equal(dst, src))  # See the top comment
        )
    ):
        # TODO: tile the dequantize -> copy to reduce memory footprint
        src = _tensor_dequantize(src)
    _tensor_copy(dst, src)
