#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Tuple

import pytest
import torch
from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    dtype_to_string,
    per_channel_qtensor_as_bytes,
    per_channel_qtensor_from_bytes,
    per_tensor_qtensor_as_bytes,
    per_tensor_qtensor_from_bytes,
    string_to_dtype,
    SUPPORTED_QUANTIZED_DTYPES,
    tensor_as_memoryview,
    tensor_from_memoryview,
)
from torchsnapshot.test_utils import rand_tensor, tensor_eq


@pytest.mark.parametrize("dtype", BUFFER_PROTOCOL_SUPPORTED_DTYPES)
def test_buffer_protocol(dtype: torch.dtype) -> None:
    foo = rand_tensor(shape=(1000, 1000), dtype=dtype)

    serialized = tensor_as_memoryview(foo).tobytes()
    dtype_str = dtype_to_string(foo.dtype)
    shape = list(foo.shape)

    bar = tensor_from_memoryview(
        memoryview(serialized),
        dtype=string_to_dtype(dtype_str),
        shape=shape,
    )
    assert torch.allclose(foo, bar)


@pytest.mark.parametrize("dtype", ALL_SUPPORTED_DTYPES)
def test_string_dtype_conversion(dtype: torch.dtype) -> None:
    dtype_str = dtype_to_string(dtype)
    restored = string_to_dtype(dtype_str)
    assert restored == dtype


@pytest.mark.parametrize("dtype", SUPPORTED_QUANTIZED_DTYPES)
@pytest.mark.parametrize("shape", [(100, 100), (10, 11, 12)])
def test_per_tensor_qtensor(dtype: torch.dtype, shape: Tuple[int, ...]) -> None:
    qtensor = rand_tensor(shape=shape, dtype=dtype)
    buf = per_tensor_qtensor_as_bytes(qtensor)
    deserialized = per_tensor_qtensor_from_bytes(buf, dtype=dtype, shape=list(shape))
    assert qtensor.dtype == deserialized.dtype
    assert qtensor.is_quantized
    assert deserialized.is_quantized
    assert qtensor.qscheme() == deserialized.qscheme()
    assert qtensor.q_scale() == deserialized.q_scale()
    assert qtensor.q_zero_point() == deserialized.q_zero_point()
    assert qtensor.stride() == deserialized.stride()
    assert torch.allclose(qtensor.dequantize(), deserialized.dequantize())


@pytest.mark.parametrize("dtype", SUPPORTED_QUANTIZED_DTYPES)
@pytest.mark.parametrize("shape", [(100, 100), (10, 11, 12)])
def test_per_channel_qtensor(dtype: torch.dtype, shape: Tuple[int, ...]) -> None:
    for axis in range(len(shape)):
        qtensor = rand_tensor(
            shape=shape,
            dtype=dtype,
            qscheme=torch.per_channel_affine,
            channel_axis=axis,
        )
        buf = per_channel_qtensor_as_bytes(qtensor)
        deserialized = per_channel_qtensor_from_bytes(
            buf, dtype=dtype, shape=list(shape)
        )
        assert qtensor.dtype == deserialized.dtype
        assert qtensor.is_quantized
        assert deserialized.is_quantized
        assert qtensor.qscheme(), deserialized.qscheme()
        assert torch.allclose(
            qtensor.q_per_channel_scales(),
            deserialized.q_per_channel_scales(),
        )
        assert torch.allclose(
            qtensor.q_per_channel_zero_points(),
            deserialized.q_per_channel_zero_points(),
        )
        assert qtensor.stride() == deserialized.stride()
        assert torch.allclose(qtensor.dequantize(), deserialized.dequantize())


@pytest.mark.parametrize("dtype", BUFFER_PROTOCOL_SUPPORTED_DTYPES)
def test_tensor_as_memoryview_for_continuous_view(dtype: torch.dtype) -> None:
    """
    Verify that tensor_as_memoryview() behaves correctly for continuous views.
    """
    tensor = rand_tensor((64, 64), dtype=dtype)
    cont_view = tensor[32:, :]
    assert cont_view.is_contiguous()

    mv = tensor_as_memoryview(cont_view)
    assert len(mv) == cont_view.nelement() * cont_view.element_size()

    deserialized_view = tensor_from_memoryview(mv=mv, dtype=dtype, shape=[32, 64])
    assert tensor_eq(deserialized_view, cont_view)
