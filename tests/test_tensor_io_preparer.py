#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import asyncio
import functools
import itertools
import tempfile
import unittest
from typing import Callable, cast, List

import torch
from torchsnapshot.io_preparers.tensor import tensor_copy, TensorIOPreparer
from torchsnapshot.io_types import ReadReq, WriteReq
from torchsnapshot.scheduler import execute_read_reqs, execute_write_reqs
from torchsnapshot.serialization import (
    ALL_SUPPORTED_DTYPES,
    BUFFER_PROTOCOL_SUPPORTED_DTYPES,
    Serializer,
)
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import async_test


class TensorIOPreparerTest(unittest.TestCase):
    TENSOR_DIM = (1000, 1000)

    @staticmethod
    def _fulfill_read_reqs_with_write_reqs(
        read_reqs: List[ReadReq], write_reqs: List[WriteReq]
    ) -> None:
        """
        Fulfill read requests with write requests. This allows us to test I/O
        preparers without depending on the scheduler and storage plugins.
        """
        if len(read_reqs) != len(write_reqs):
            raise AssertionError(
                "The size of read_reqs and write reqs must be the same."
            )
        path_to_buffer_stager = {
            write_req.path: write_req.buffer_stager for write_req in write_reqs
        }
        event_loop = asyncio.new_event_loop()
        for read_req in read_reqs:
            buffer_stager = path_to_buffer_stager[read_req.path]
            buffer_stager.get_staging_cost_bytes()
            read_req.buffer_consumer.get_consuming_cost_bytes()
            buf = event_loop.run_until_complete(buffer_stager.stage_buffer())
            event_loop.run_until_complete(
                read_req.buffer_consumer.consume_buffer(buf=cast(bytes, buf))
            )
        event_loop.close()

    @classmethod
    def _rand(cls, dtype: torch.dtype) -> torch.Tensor:
        if dtype in (torch.qint32, torch.qint8, torch.quint8):
            return torch.quantize_per_tensor(
                torch.rand(cls.TENSOR_DIM), 0.1, 10, dtype=dtype
            )
        elif dtype.is_floating_point or dtype.is_complex:
            return torch.randn(cls.TENSOR_DIM, dtype=dtype)
        elif dtype == torch.bool:
            return torch.randint(2, cls.TENSOR_DIM, dtype=dtype)
        else:
            return torch.randint(torch.iinfo(dtype).max, cls.TENSOR_DIM, dtype=dtype)

    def _test_tensor_io_preparer_helper(self, dtype: torch.dtype) -> None:
        """
        Use TensorIOPreparer to transfer the data of one tensor to a different
        tensor and verify that their data are the same.
        """
        foo = self._rand(dtype=dtype)
        bar = self._rand(dtype=dtype)

        # TODO: test memory estimation
        if foo.is_quantized:
            self.assertFalse(torch.allclose(foo.dequantize(), bar.dequantize()))
        else:
            self.assertFalse(torch.allclose(foo, bar))

        entry, write_reqs = TensorIOPreparer.prepare_write(
            storage_path="/foo", tensor=foo
        )
        # IMPORTANT: we should only serialize a tensor with BUFFER_PROTOCOL if
        # the tensor's dtype is in BUFFER_PROTOCOL_SUPPORTED_DTYPES.
        self.assertEqual(
            entry.serializer == Serializer.BUFFER_PROTOCOL.value,
            dtype in BUFFER_PROTOCOL_SUPPORTED_DTYPES,
        )

        read_reqs, _ = TensorIOPreparer.prepare_read(entry=entry, tensor_out=bar)
        self._fulfill_read_reqs_with_write_reqs(
            read_reqs=read_reqs, write_reqs=write_reqs
        )
        if foo.is_quantized:
            self.assertTrue(torch.allclose(foo.dequantize(), bar.dequantize()))
        else:
            self.assertTrue(torch.allclose(foo, bar))

    def test_tensor_io_preparer(self) -> None:
        for dtype in ALL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                self._test_tensor_io_preparer_helper(dtype=dtype)

    def test_get_tensor_size_from_entry(self) -> None:
        for dtype in ALL_SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                tensor = self._rand(dtype=dtype)
                entry, _ = TensorIOPreparer.prepare_write("/foo", tensor)
                self.assertTrue(
                    TensorIOPreparer.get_tensor_size_from_entry(entry),
                    tensor.element_size() * tensor.nelement(),
                )

    async def _test_chunked_read_helper(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> None:
        """
        First save src tensor. Then load the persisted src tensor into dst
        tensor using a buffer_size_limit_bytes that would lead to chunked read.
        Finally verify that src tensor equals to dst tensor.
        """
        self.assertFalse(torch.allclose(src, dst))

        with tempfile.TemporaryDirectory() as path:
            storage = FSStoragePlugin(root=path)
            entry, write_reqs = TensorIOPreparer.prepare_write(
                storage_path="src", tensor=src
            )
            pending_io_work = await execute_write_reqs(
                write_reqs=write_reqs,
                storage=storage,
                memory_budget_bytes=32 * 1024**3,
                rank=0,
            )
            await pending_io_work.complete()

            buffer_size_limit_bytes = src.nelement() * src.element_size() // 4
            read_reqs, _ = TensorIOPreparer.prepare_read(
                entry=entry,
                tensor_out=dst,
                buffer_size_limit_bytes=buffer_size_limit_bytes,
            )
            self.assertEqual(len(read_reqs), 4)
            await execute_read_reqs(
                read_reqs=read_reqs,
                storage=storage,
                memory_budget_bytes=32 * 1024**3,
                rank=0,
            )
        self.assertTrue(torch.allclose(src, dst))

    @async_test
    async def test_chunked_read(self) -> None:
        foo = torch.rand(2000, 2000)
        bar = torch.rand(2000, 2000)
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # strided
        foo = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2))
        bar = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2))
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # strided with storage_offset
        foo = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2), 1009)
        bar = torch.rand(5000, 5000).as_strided((2000, 2000), (2, 2), 1009)
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # non-contiguous view
        foo = torch.rand(4000, 4000)[1000:3000, 1000:3000]
        bar = torch.rand(4000, 4000)[1000:3000, 1000:3000]
        await self._test_chunked_read_helper(src=foo, dst=bar)

        # 1009 is a prime number
        foo = torch.rand(1009, 1009)
        bar = torch.rand(1009, 1009)
        await self._test_chunked_read_helper(src=foo, dst=bar)

    @async_test
    async def test_custom_tensor_prepare_func(self) -> None:
        foo = torch.rand(2000, 2000)
        entry, write_reqs = TensorIOPreparer.prepare_write(
            storage_path="src", tensor=foo
        )
        self.assertEqual(entry.dtype, "torch.float32")
        self.assertEqual(entry.shape, [2000, 2000])

        def quantize(tensor: torch.Tensor, tracing: bool) -> torch.Tensor:
            return torch.quantize_per_tensor(tensor, 0.1, 10, torch.qint8)

        bar = torch.rand(2000, 2000)
        entry, write_reqs = TensorIOPreparer.prepare_write(
            storage_path="src",
            tensor=bar,
            _tensor_prepare_func=quantize,
        )
        self.assertEqual(entry.dtype, "torch.qint8")
        self.assertEqual(entry.shape, [2000, 2000])

        # Expect prepare_write to fail if _tensor_prepare_func changed the tensor size
        def view(tensor: torch.Tensor, tracing: bool) -> torch.Tensor:
            return tensor[:1000, :1000]

        bar = torch.rand(2000, 2000)
        with self.assertRaises(RuntimeError):
            entry, write_reqs = TensorIOPreparer.prepare_write(
                storage_path="src",
                tensor=bar,
                _tensor_prepare_func=view,
            )

    def _verify_copy_from_float_to_float(
        self,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
    ) -> None:
        """
        Copy from a float tensor to a float tensor.
        """
        src = torch.rand(200, 200, dtype=src_dtype)
        dst = torch.rand(200, 200, dtype=dst_dtype)
        tensor_copy(dst, src)

        # For verification purpose, cast the higher precision (lower
        # resolution) tensor to the dtype of the other tensor
        if torch.finfo(src_dtype).resolution < torch.finfo(dst_dtype).resolution:
            self.assertTrue(torch.allclose(src.to(dtype=dst_dtype), dst))
        else:
            self.assertTrue(torch.allclose(src, dst.to(dtype=src_dtype)))

    def _verify_copy_from_quantized_to_float(
        self,
        quantize_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Copy from a quantized tensor to a float tensor.
        """
        src = quantize_func(torch.rand(200, 200))
        dst = torch.rand(200, 200)
        tensor_copy(dst, src)
        self.assertTrue(torch.allclose(src.dequantize(), dst))

    def _verify_copy_from_float_to_quantized(
        self,
        quantize_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Copy from a float tensor to a quantized tensor.
        """
        src = torch.rand(200, 200)
        dst = quantize_func(torch.rand(200, 200))
        tensor_copy(dst, src)
        self.assertTrue(
            torch.allclose(quantize_func(src).dequantize(), dst.dequantize())
        )

    def _verify_copy_from_quantized_to_quantized(
        self,
        dst_quantize_func: Callable[[torch.Tensor], torch.Tensor],
        src_quantize_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Copy from a quantized tensor to a quantized tensor.
        """
        src = src_quantize_func(torch.rand(200, 200))
        dst = dst_quantize_func(torch.rand(200, 200))
        tensor_copy(dst, src)
        self.assertTrue(
            torch.allclose(
                dst_quantize_func(src.dequantize()).dequantize(), dst.dequantize()
            )
        )

    def _verify_copy_from_quantized_to_quantized_view(
        self,
        dst_quantize_func: Callable[[torch.Tensor], torch.Tensor],
        src_quantize_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Copy from a quantized tensor to a view of quantized tensor.
        """
        src_original = src_quantize_func(torch.rand(400, 400))
        src_view = src_original[:200, :200]
        dst = dst_quantize_func(torch.rand(200, 200))
        tensor_copy(dst, src_view)
        self.assertTrue(
            torch.allclose(
                dst_quantize_func(src_view.dequantize()).dequantize(), dst.dequantize()
            )
        )
        self.assertTrue(
            torch.allclose(
                dst_quantize_func(src_original[:200, :200].dequantize()).dequantize(),
                dst.dequantize(),
            )
        )

    def test_tensor_copy(self) -> None:
        """
        Verify the behavior of tensor_copy in various situations.
        """

        float_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        for dst_dtype, src_dtype in itertools.product(float_dtypes, float_dtypes):
            self._verify_copy_from_float_to_float(
                dst_dtype=dst_dtype, src_dtype=src_dtype
            )

        def quantize_per_tensor(
            dtype: torch.dtype, tensor: torch.Tensor
        ) -> torch.Tensor:
            return torch.quantize_per_tensor(tensor, 0.1, 10, dtype=dtype)

        def quantize_per_channel(
            dtype: torch.dtype, tensor: torch.Tensor
        ) -> torch.Tensor:
            scales = torch.arange(tensor.shape[0]).float() / 100
            zero_points = torch.arange(tensor.shape[0]) % 128  # 128 is the upperbound
            return torch.quantize_per_channel(
                tensor, scales, zero_points, axis=0, dtype=dtype
            )

        for quantize_func, dtype in itertools.product(
            [quantize_per_tensor, quantize_per_channel],
            [torch.qint8, torch.quint8],
        ):
            self._verify_copy_from_quantized_to_float(
                quantize_func=functools.partial(quantize_func, dtype)
            )
            self._verify_copy_from_float_to_quantized(
                quantize_func=functools.partial(quantize_func, dtype)
            )

        for (
            dst_quantize_func,
            dst_dtype,
            src_quantize_func,
            src_dtype,
        ) in itertools.product(
            [quantize_per_tensor, quantize_per_channel],
            [torch.qint8, torch.quint8],
            [quantize_per_tensor, quantize_per_channel],
            [torch.qint8, torch.quint8],
        ):
            self._verify_copy_from_quantized_to_quantized(
                dst_quantize_func=functools.partial(dst_quantize_func, dst_dtype),
                src_quantize_func=functools.partial(src_quantize_func, src_dtype),
            )
            self._verify_copy_from_quantized_to_quantized_view(
                dst_quantize_func=functools.partial(dst_quantize_func, dst_dtype),
                src_quantize_func=functools.partial(src_quantize_func, src_dtype),
            )
