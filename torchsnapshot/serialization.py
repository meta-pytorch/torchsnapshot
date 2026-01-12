#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import io
import logging
import operator
import struct
import warnings
from enum import Enum
from typing import Dict, List

import torch

try:
    # TODO: drop this once PyTorch 1.12 is no longer supported
    # https://github.com/pytorch/pytorch/pull/82438
    # pyre-ignore
    from torch import _UntypedStorage as UntypedStorage  # @manual
except ImportError:
    from torch import UntypedStorage  # @manual


logger: logging.Logger = logging.getLogger(__name__)


# https://pytorch.org/docs/stable/tensors.html#data-types
ALL_SUPPORTED_DTYPES: List[torch.dtype] = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.complex128,
    torch.complex64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.qint32,
    torch.qint8,
    torch.quint8,
]

SUPPORTED_QUANTIZED_DTYPES: List[torch.dtype] = [
    torch.qint32,
    torch.qint8,
    torch.quint8,
]

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#nccldatatype-t
NCCL_SUPPORTED_DTYPES: List[torch.dtype] = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int8,
    torch.uint8,
]

# The approach is dumb. But we want to be 100% certain we can recognize the
# dtype strings we persist.
_DTYPE_TO_STRING: Dict[torch.dtype, str] = {
    torch.float64: "torch.float64",
    torch.float32: "torch.float32",
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.complex128: "torch.complex128",
    torch.complex64: "torch.complex64",
    torch.int64: "torch.int64",
    torch.int32: "torch.int32",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
    torch.qint32: "torch.qint32",
    torch.qint8: "torch.qint8",
    torch.quint8: "torch.quint8",
}

# This could be figured out by creating an empty tensor with the dtype and
# checking its .element_size(). However, the approach may not work for
# quantized dtypes if the appropriate backend is not available. Since special
# casing is inevitable, we might as well enumerate all dtypes.
_DTYPE_TO_ELEMENT_SIZE: Dict[torch.dtype, int] = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.complex128: 16,
    torch.complex64: 8,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
    torch.qint32: 4,
    torch.qint8: 1,
    torch.quint8: 1,
}


# The approach is dumb. But we want to be 100% certain we can recognize the
# dtype strings we persist.
_STRING_TO_DTYPE: Dict[str, torch.dtype] = {
    val: key for key, val in _DTYPE_TO_STRING.items()
}


def dtype_to_string(dtype: torch.dtype) -> str:
    """
    Converty a class::`torch.dtype` to class::`str`.
    """
    if dtype in _DTYPE_TO_STRING:
        return _DTYPE_TO_STRING[dtype]
    else:
        raise ValueError(
            f"Unsupported dtype {dtype}. (Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


def dtype_to_element_size(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_TO_ELEMENT_SIZE:
        return _DTYPE_TO_ELEMENT_SIZE[dtype]
    else:
        raise ValueError(
            f"Unsupported dtype {dtype}. (Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


def string_to_dtype(s: str) -> torch.dtype:
    """
    Converty a class::`torch.dtype` to class::`str`.
    """
    if s in _STRING_TO_DTYPE:
        return _STRING_TO_DTYPE[s]
    else:
        raise ValueError(
            f"Unsupported dtype {s}. (Supported dtypes are: {ALL_SUPPORTED_DTYPES})"
        )


class Serializer(Enum):
    TORCH_SAVE = "torch_save"
    BUFFER_PROTOCOL = "buffer_protocol"
    PER_TENSOR_QTENSOR = "per_tensor_qtensor"
    PER_CHANNEL_QTENSOR = "per_channel_qtensor"


BUFFER_PROTOCOL_SUPPORTED_DTYPES: List[torch.dtype] = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


# pyre-fixme[24]: Generic type `memoryview` expects 1 type parameter.
def tensor_as_memoryview(tensor: torch.Tensor) -> memoryview:
    """
    Obtain the class::`memoryview` of a class::`torch.Tensor`.

    It is the caller's responsibility to ensure that the input is a CPU tensor
    and its dtype is defined in `BUFFER_PROTOCOL_SUPPORTED_DTYPES`. The
    function will raise an exception if the requirements are not met.

    Args:
        tensor: The tensor from which to obtain memoryview.

    Returns:
        The class::`memoryview` of the input tensor.
    """
    if tensor.dtype not in BUFFER_PROTOCOL_SUPPORTED_DTYPES:
        raise ValueError(
            f"tensor_as_memoryview() doesn't support the dtype {tensor.dtype}."
        )
    if tensor.device != torch.device("cpu"):
        raise ValueError("tensor_as_memoryview() only accepts CPU tensors.")
    if not tensor.is_contiguous():
        # This is only needed if the caller didn't need to copy the tensor from
        # device to CPU. This is still more efficient than torch.save().
        tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        return _tensor_as_memoryview_via_untyped_storage(tensor)
    # pyre-fixme[6]: For 1st argument expected `Buffer` but got `ndarray[Any, Any]`.
    return memoryview(tensor.numpy()).cast("b")


# pyre-fixme[24]: Generic type `memoryview` expects 1 type parameter.
def _tensor_as_memoryview_via_untyped_storage(tensor: torch.Tensor) -> memoryview:
    """
    Obtain the class::`memoryview` of a class::`torch.Tensor` via untyped storage.

    This function can be used to obtain the memoryview of a tensor whose dtype
    does not have an counterpart in numpy.

    Args:
        tensor: The tensor from which to obtain memoryview.

    Returns:
        The class::`memoryview` of the input tensor.
    """
    if not tensor.is_contiguous():
        raise AssertionError(
            "_tensor_as_memoryview_via_untyped_storage can be only used "
            "with contiguous tensors"
        )
    untyped_storage = contiguous_view_as_untyped_storage(tensor)
    tensor = torch.empty((0))
    tensor.set_(untyped_storage)
    # pyre-fixme[6]: For 1st argument expected `Buffer` but got `ndarray[Any, Any]`.
    return memoryview(tensor.numpy()).cast("b")


def contiguous_view_as_untyped_storage(tensor: torch.Tensor) -> UntypedStorage:
    if not tensor.is_contiguous():
        raise AssertionError(
            "contiguous_view_as_untyped_storage can be "
            "only used with contiguous tensors."
        )
    if hasattr(tensor, "untyped_storage"):
        untyped_storage = tensor.untyped_storage()
    elif hasattr(tensor.storage(), "_untyped"):
        # TODO: drop this once PyTorch 1.12 is no longer supported
        # https://github.com/pytorch/pytorch/pull/82438
        untyped_storage = tensor.storage()._untyped()
    else:
        untyped_storage = tensor.storage().untyped()
    return untyped_storage[
        tensor.storage_offset() * tensor.element_size() : tensor.storage_offset()
        * tensor.element_size()
        + tensor.nelement() * tensor.element_size()
    ]


def tensor_from_memoryview(
    # pyre-fixme[24]: Generic type `memoryview` expects 1 type parameter.
    mv: memoryview,
    dtype: torch.dtype,
    shape: List[int],
) -> torch.Tensor:
    # PyTorch issues a warning if the given memoryview is non-writable. This is
    # not a concern for torchsnapshot, as tensors created from non-writable
    # buffers are all read-only, intermediate tensors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.reshape(torch.frombuffer(mv, dtype=dtype), shape)


def torch_save_as_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def torch_load_from_bytes(buf: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(buf))


def per_tensor_qtensor_as_bytes(tensor: torch.Tensor) -> bytes:
    """
    Serialize a per-tensor quantized tensor.

    Binary format:

    +----------------------------------+ 0 bytes
    |          tensor storage          |
    +----------------------------------+ nelement * element_size bytes
    |    q_scale packed as C double    |
    +----------------------------------+ nelement * element_size + 8 bytes
    |q_zero_point packed as C long long|
    +----------------------------------+ nelement * element_size + 16 bytes

    On deserialization, nelement and element_size can be inferred from dtype
    and shape which are stored separately.

    Args:
        tensor: The per-tensor quantized tensor to serialize.

    Returns:
        The serialized tensor.
    """
    if not tensor.is_quantized or tensor.qscheme == torch.per_tensor_affine:
        raise RuntimeError(
            "per_tensor_qtensor_as_bytes() only supports "
            "per_tensor_affine quantized tensor."
        )
    buf = io.BytesIO()
    buf.write(_tensor_as_memoryview_via_untyped_storage(tensor))
    buf.write(struct.pack("d", tensor.q_scale()))
    buf.write(struct.pack("q", tensor.q_zero_point()))
    return buf.getvalue()


def per_tensor_qtensor_from_bytes(
    buf: bytes, dtype: torch.dtype, shape: List[int]
) -> torch.Tensor:
    """
    Deserialize a per-channel quantized tensor.

    NOTE: this is a zero-copy deserialization, meaning that the deserialized
    tensor directly uses the input buffer as its storage. The deserialized
    tensor can only be used in read-only fashion.

    Args:
        buf: The serialized tensor.
        dtype: The dtype of the serialized tensor.
        shape: The shape of the serialized tensor.

    Returns:
        The deserialized tensor.
    """
    nelements = functools.reduce(operator.mul, shape, 1)
    curr_stride = nelements
    strides = []
    for dim_sz in shape:
        curr_stride //= dim_sz
        strides.append(curr_stride)

    data_sz_bytes = nelements * dtype_to_element_size(dtype)

    expected_buf_len = data_sz_bytes + 16
    if len(buf) != expected_buf_len:
        raise RuntimeError(
            "The expected buffer size for the per-tensor quantized tensor "
            f"(dtype: {dtype}, shape: {shape}) is {expected_buf_len}. "
            f"The size of the input buffer is {len(buf)}."
        )

    data = buf[:data_sz_bytes]
    scale = struct.unpack("d", buf[data_sz_bytes : data_sz_bytes + 8])[0]
    zero_point = struct.unpack("q", buf[data_sz_bytes + 8 : data_sz_bytes + 16])[0]

    # Assemble the deserialized tensor
    qtensor = torch._empty_affine_quantized(
        (0), scale=scale, zero_point=zero_point, dtype=dtype
    )
    storage = torch.FloatStorage.from_buffer(memoryview(data), byte_order="native")
    if hasattr(storage, "_untyped"):
        # TODO: drop this once PyTorch 1.12 is no longer supported
        # https://github.com/pytorch/pytorch/pull/82438
        qtensor.set_(storage._untyped(), 0, shape, strides)
    else:
        qtensor.set_(storage.untyped(), 0, shape, strides)
    return qtensor


def per_channel_qtensor_as_bytes(tensor: torch.Tensor) -> bytes:
    """
    Serialize a per-channel quantized tensor.

    Binary format:

    +------------------------------------------+ 0 bytes
    |                  axis                    |
    +------------------------------------------+ 8 bytes
    |             tensor storage               |
    +------------------------------------------+ 8 + nelement * element_size bytes
    |   q_per_channel_scales tensor storage    |
    +------------------------------------------+ 8 + nelement * element_size + 8 * shape[axis] bytes
    | q_per_channel_zero_points tensor storage |
    +------------------------------------------+ 8 + nelement * element_size + 16 * shape[axis] bytes

    On deserialization, nelement and element_size can be inferred from dtype
    and shape which are stored separately.

    Args:
        tensor: The per-channel quantized tensor to serialize.

    Returns:
        The serialized tensor.
    """
    if not tensor.is_quantized or tensor.qscheme == torch.per_channel_affine:
        raise RuntimeError(
            "per_channel_qtensor_as_bytes() only supports "
            "per_channel_affine quantized tensor."
        )
    buf = io.BytesIO()
    buf.write(struct.pack("q", tensor.q_per_channel_axis()))
    buf.write(_tensor_as_memoryview_via_untyped_storage(tensor))
    buf.write(
        tensor_as_memoryview(tensor.q_per_channel_scales().to(dtype=torch.float64))
    )
    buf.write(
        tensor_as_memoryview(tensor.q_per_channel_zero_points().to(dtype=torch.int64))
    )
    return buf.getvalue()


def per_channel_qtensor_from_bytes(
    buf: bytes, dtype: torch.dtype, shape: List[int]
) -> torch.Tensor:
    """
    Deserialize a per-channel quantized tensor.

    NOTE: this is a zero-copy deserialization, meaning that the deserialized
    tensor directly uses the input buffer as its storage. The deserialized
    tensor can only be used in read-only fashion.

    Args:
        buf: The serialized tensor.
        dtype: The dtype of the serialized tensor.
        shape: The shape of the serialized tensor.

    Returns:
        The deserialized tensor.
    """
    nelements = functools.reduce(operator.mul, shape, 1)
    curr_stride = nelements
    strides = []
    for dim_sz in shape:
        curr_stride //= dim_sz
        strides.append(curr_stride)

    data_sz_bytes = nelements * dtype_to_element_size(dtype)
    axis = struct.unpack("q", buf[:8])[0]

    if axis < 0 or axis >= len(shape):
        raise RuntimeError(
            f"Read invalid axis ({axis}) from the input buffer when deserializing "
            f"the per-channel quantized tensor (dtype: {dtype}, shape: {shape})."
        )

    expected_buf_len = data_sz_bytes + 8 + 16 * shape[axis]
    if len(buf) != expected_buf_len:
        raise RuntimeError(
            "The expected buffer size for the per-channel quantized tensor "
            f"(dtype: {dtype}, shape: {shape}, axis: {axis}) is {expected_buf_len}. "
            f"The size of the input buffer is {len(buf)}."
        )

    data = buf[8 : 8 + data_sz_bytes]
    scales_data = buf[8 + data_sz_bytes : data_sz_bytes + 8 + 8 * shape[axis]]
    scales_tensor = tensor_from_memoryview(
        mv=memoryview(scales_data), shape=[shape[axis]], dtype=torch.float64
    )
    zero_points_data = buf[
        8 + data_sz_bytes + 8 * shape[axis] : 8 + data_sz_bytes + 16 * shape[axis]
    ]
    zero_points_tensor = tensor_from_memoryview(
        mv=memoryview(zero_points_data), shape=[shape[axis]], dtype=torch.int64
    )

    # Assemble the deserialized tensor
    qtensor = torch._empty_per_channel_affine_quantized(
        (0),
        scales=scales_tensor,
        zero_points=zero_points_tensor,
        axis=axis,
        dtype=dtype,
    )
    storage = torch.FloatStorage.from_buffer(memoryview(data), byte_order="native")
    if hasattr(storage, "_untyped"):
        # TODO: drop this once PyTorch 1.12 is no longer supported
        # https://github.com/pytorch/pytorch/pull/82438
        qtensor.set_(storage._untyped(), 0, shape, strides)
    else:
        qtensor.set_(storage.untyped(), 0, shape, strides)
    return qtensor
