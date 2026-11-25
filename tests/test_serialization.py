#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import struct
from pathlib import Path
from typing import Tuple

import pytest
import torch
from torchsnapshot import Snapshot, StateDict
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


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("view", ["full", "offset", "strided"])
def test_bfloat16_memoryview_preserves_all_bytes(
    default_dtype: torch.dtype, device: str, view: str
) -> None:
    values = torch.tensor([0.5, 1.0, -2.5, 3.75, 8.0], dtype=torch.bfloat16)
    if view == "full":
        values = values[:3]
    elif view == "offset":
        values = values[1:4]
    else:
        values = values[::2]
    # These values are exactly representable as bfloat16. The upper 16 bits of
    # their IEEE float32 representation independently define the stored bytes.
    bits = [struct.unpack("=I", struct.pack("=f", x))[0] >> 16 for x in values]
    expected = struct.pack("=" + "H" * len(bits), *bits)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        with torch.device(device):
            actual = tensor_as_memoryview(values).tobytes()
    finally:
        torch.set_default_dtype(previous_dtype)
    assert actual == expected


@pytest.mark.parametrize("dtype", SUPPORTED_QUANTIZED_DTYPES)
@pytest.mark.parametrize("size", [1, 3, 5])
@pytest.mark.parametrize("per_channel", [False, True])
def test_small_quantized_tensor_bytes_and_roundtrip(
    dtype: torch.dtype, size: int, per_channel: bool
) -> None:
    values = torch.arange(size, dtype=torch.float32) - 0.5
    if per_channel:
        scales = torch.full((size,), 0.5, dtype=torch.float64)
        zeros = torch.full((size,), 4, dtype=torch.int64)
        tensor = torch.quantize_per_channel(values, scales, zeros, 0, dtype)
        serialize = per_channel_qtensor_as_bytes
        deserialize = per_channel_qtensor_from_bytes
    else:
        tensor = torch.quantize_per_tensor(values, 0.5, 4, dtype)
        serialize = per_tensor_qtensor_as_bytes
        deserialize = per_tensor_qtensor_from_bytes
    # Quantization is exact for this input: round(x /0.5) +4.
    integers = [2 * i + 3 for i in range(size)]
    int_format = {torch.qint8: "b", torch.quint8: "B", torch.qint32: "i"}[dtype]
    data = struct.pack("=" + int_format * size, *integers)
    if per_channel:
        expected = (
            struct.pack("=q", 0)
            + data
            + struct.pack("=" + "d" * size, *([0.5] * size))
            + struct.pack("=" + "q" * size, *([4] * size))
        )
    else:
        expected = data + struct.pack("=dq", 0.5, 4)
    actual = serialize(tensor)
    assert actual == expected
    restored = deserialize(actual, dtype=dtype, shape=[size])
    assert restored.dtype == dtype
    assert torch.equal(restored.int_repr(), tensor.int_repr())
    assert torch.equal(restored.dequantize(), tensor.dequantize())


@pytest.mark.parametrize("dtype", [torch.bfloat16] + SUPPORTED_QUANTIZED_DTYPES)
def test_snapshot_small_tensor_roundtrip(tmp_path: Path, dtype: torch.dtype) -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    if dtype == torch.bfloat16:
        value = values.to(dtype)
        destination = torch.zeros_like(value)
    else:
        value = torch.quantize_per_tensor(values, 0.5, 4, dtype)
        destination = torch.quantize_per_tensor(torch.zeros_like(values), 0.5, 4, dtype)
    saved = Snapshot.take(str(tmp_path), {"state": StateDict(value=value)})
    output = StateDict(value=destination)
    saved.restore({"state": output})
    assert output["value"].dtype == dtype
    assert torch.equal(output["value"], value)
