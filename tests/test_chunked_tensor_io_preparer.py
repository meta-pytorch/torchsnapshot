#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import asyncio
import tempfile
import unittest
from typing import List, Tuple

import torch
import torchsnapshot
import torchsnapshot.io_preparer as io_preparer
from torch import distributed as dist
from torch.distributed import launcher as pet
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot import manifest
from torchsnapshot.io_preparer import Chunk, ChunkedTensorIOPreparer, get_storage_path
from torchsnapshot.io_types import WriteReq
from torchsnapshot.manifest import ChunkedTensorEntry
from torchsnapshot.scheduler import execute_read_reqs, execute_write_reqs
from torchsnapshot.serialization import string_to_dtype, tensor_from_memoryview
from torchsnapshot.stateful import AppState
from torchsnapshot.storage_plugins.fs import FSStoragePlugin
from torchsnapshot.test_utils import (
    assert_state_dict_eq,
    async_test,
    get_pet_launch_config,
)


class ChunkedTensorIOPreparerTest(unittest.TestCase):
    # tests for chunk_tensor method
    def _test_chunk_tensor_helper(
        self,
        expected_chunks: List[Chunk],
        tensor_to_chunk: torch.Tensor,
        max_chunk_sz_bytes: int,
    ) -> None:
        actual_chunks = ChunkedTensorIOPreparer.chunk_tensor(
            tensor=tensor_to_chunk,
            chunk_sz_bytes=max_chunk_sz_bytes,
        )
        self.assertEqual(len(expected_chunks), len(actual_chunks))
        for i in range(len(expected_chunks)):
            self.assertListEqual(expected_chunks[i].offsets, actual_chunks[i].offsets)
            self.assertListEqual(expected_chunks[i].sizes, actual_chunks[i].sizes)

    def test_chunk_tensor_0d(self) -> None:
        tensor = torch.tensor(0)
        max_chunk_sz_bytes = 100
        expected_chunks = [Chunk(offsets=[0], sizes=[1], dtype=str(tensor.dtype))]
        self._test_chunk_tensor_helper(expected_chunks, tensor, max_chunk_sz_bytes)

    def test_chunk_tensor_div_by_elem_size(self) -> None:
        tensor = torch.randn(7, 10)
        self.assertTrue(tensor.is_contiguous())
        max_chunk_sz_bytes = 30 * tensor.element_size()
        expected_chunks = [
            Chunk(offsets=[0, 0], sizes=[3, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[3, 0], sizes=[3, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[6, 0], sizes=[1, 10], dtype=str(tensor.dtype)),
        ]
        self._test_chunk_tensor_helper(expected_chunks, tensor, max_chunk_sz_bytes)

    def test_chunk_tensor_nondiv_by_elem_size(self) -> None:
        tensor = torch.randn(7, 10)
        self.assertTrue(tensor.is_contiguous())
        max_chunk_sz_bytes = 180
        expected_chunks = [
            Chunk(offsets=[0, 0], sizes=[4, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[4, 0], sizes=[3, 10], dtype=str(tensor.dtype)),
        ]
        self._test_chunk_tensor_helper(expected_chunks, tensor, max_chunk_sz_bytes)

    def test_chunk_tensor_more_chunks(self) -> None:
        tensor = torch.randn(10, 11)
        non_contig_tensor = tensor[:, :10]  # (10,10)
        self.assertFalse(non_contig_tensor.is_contiguous())
        expected_chunks = [
            Chunk(offsets=[0, 0], sizes=[2, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[2, 0], sizes=[2, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[4, 0], sizes=[2, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[6, 0], sizes=[2, 10], dtype=str(tensor.dtype)),
            Chunk(offsets=[8, 0], sizes=[2, 10], dtype=str(tensor.dtype)),
        ]
        max_chunk_sz_bytes = 21 * tensor.element_size()
        self._test_chunk_tensor_helper(
            expected_chunks, non_contig_tensor, max_chunk_sz_bytes
        )

    def test_chunk_3d_noncontig_tensor(self) -> None:
        tensor = torch.randn(6, 4, 2).transpose(1, 2)
        self.assertFalse(tensor.is_contiguous())
        expected_chunks = [
            Chunk(offsets=[0, 0, 0], sizes=[3, 2, 4], dtype=str(tensor.dtype)),
            Chunk(offsets=[3, 0, 0], sizes=[3, 2, 4], dtype=str(tensor.dtype)),
        ]
        max_chunk_sz_bytes = 24 * tensor.element_size()
        self._test_chunk_tensor_helper(expected_chunks, tensor, max_chunk_sz_bytes)

    # test for prepare_write method
    @staticmethod
    def _check_entry_and_write_reqs(
        global_tensor: torch.Tensor,
        expected: List[List[Tuple[torch.Tensor, List[int], List[int]]]],
        entry: ChunkedTensorEntry,
        write_reqs: List[WriteReq],
        replicated: bool,
        rank: int,
    ) -> None:
        tc = unittest.TestCase()
        dir = "replicated" if replicated else str(rank)

        n_chunks = len(expected[rank])
        tc.assertEqual(len(entry.chunks), n_chunks)
        tc.assertEqual(len(write_reqs), n_chunks)

        expected_chunks = []
        for i in range(n_chunks):
            expected_chunk_tensor = expected[rank][i][0]
            expected_offsets = expected[rank][i][1]
            expected_sizes = expected[rank][i][2]
            expected_path = f"{dir}/foo_{expected_offsets[0]}_{expected_offsets[1]}"
            expected_chunks.append(
                manifest.Shard(
                    offsets=expected_offsets,
                    sizes=expected_sizes,
                    tensor=manifest.TensorEntry(
                        location=expected_path,
                        serializer="buffer_protocol",
                        dtype=str(global_tensor.dtype),
                        shape=expected_sizes,
                        replicated=False,
                    ),
                )
            )
            tc.assertEqual(write_reqs[i].path, expected_path)

            loop = asyncio.new_event_loop()
            buf = loop.run_until_complete(write_reqs[i].buffer_stager.stage_buffer())
            # Make sure only the data described by the view gets persisted
            loaded = tensor_from_memoryview(
                mv=memoryview(buf),
                dtype=string_to_dtype(entry.chunks[i].tensor.dtype),
                shape=entry.chunks[i].tensor.shape,
            )
            tc.assertEqual(loaded.nelement(), loaded.storage().size())
            tc.assertTrue(torch.equal(expected_chunk_tensor, loaded))

        # check entry
        tc.assertEqual(
            entry,
            manifest.ChunkedTensorEntry(
                dtype=str(global_tensor.dtype),
                shape=list(global_tensor.shape),
                chunks=expected_chunks,
                replicated=replicated,
            ),
        )

    @staticmethod
    def _worker_replicated_true() -> None:
        global_tensor = torch.rand((7, 10))
        replicated = True
        chunking_instruction = [
            Chunk(offsets=[0, 0], sizes=[4, 10], dtype=str(global_tensor.dtype)),
            Chunk(offsets=[4, 0], sizes=[3, 10], dtype=str(global_tensor.dtype)),
        ]

        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()

        storage_path = get_storage_path(
            global_tensor, "foo", rank=rank, replicated=replicated
        )

        entry, write_reqs = ChunkedTensorIOPreparer.prepare_write(
            storage_path, global_tensor, chunking_instruction=chunking_instruction
        )
        entry.replicated = replicated  # need to set entry's replicated field
        expected = [
            [
                (global_tensor[0:4, :], [0, 0], [4, 10]),
                (global_tensor[4:7, :], [4, 0], [3, 10]),
            ],
            [
                (global_tensor[0:4, :], [0, 0], [4, 10]),
                (global_tensor[4:7, :], [4, 0], [3, 10]),
            ],
        ]
        ChunkedTensorIOPreparerTest._check_entry_and_write_reqs(
            global_tensor, expected, entry, write_reqs, replicated, rank
        )

    def test_prepare_write_replicated_true(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        pet.elastic_launch(lc, entrypoint=self._worker_replicated_true)()

    # test for prepare_write + prepare_read
    async def _test_chunked_read_helper(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        chunking_instruction: List[Chunk],
    ) -> None:
        """
        First save src tensor. Then load the persisted src tensor into dst
        tensor using a buffer_size_limit_bytes that would lead to chunked read.
        Finally verify that src tensor equals to dst tensor.
        """
        with tempfile.TemporaryDirectory() as path:
            storage = FSStoragePlugin(root=path)
            entry, write_reqs = ChunkedTensorIOPreparer.prepare_write(
                storage_path="src",
                tensor=src,
                chunking_instruction=chunking_instruction,
            )
            pending_io_work = await execute_write_reqs(
                write_reqs=write_reqs,
                storage=storage,
                memory_budget_bytes=32 * 1024**3,
                rank=0,
            )
            await pending_io_work.complete()

            buffer_size_limit_bytes = src.nelement() * src.element_size() // 4
            read_reqs, _ = ChunkedTensorIOPreparer.prepare_read(
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
        foo = torch.ones(2000, 2000)
        bar = torch.ones(2000, 2000)

        while torch.allclose(foo, bar):
            foo = torch.rand(2000, 2000)
            bar = torch.rand(2000, 2000)

        chunking_instruction = [
            Chunk(offsets=[0, 0], sizes=[1000, 2000], dtype=str(foo.dtype)),
            Chunk(offsets=[1000, 0], sizes=[1000, 2000], dtype=str(foo.dtype)),
        ]
        self.assertFalse(torch.allclose(foo, bar))
        await self._test_chunked_read_helper(
            src=foo, dst=bar, chunking_instruction=chunking_instruction
        )
        await self._test_chunked_read_helper(
            src=foo,
            dst=bar,
            chunking_instruction=chunking_instruction,
        )

    @staticmethod
    def _worker_write_read_with_small_chunk_size(path: str) -> None:
        dist.init_process_group(backend="gloo")
        ddp_foo = DDP(torch.nn.Linear(4, 3))
        ddp_bar = DDP(torch.nn.Linear(4, 3))
        nonddp_foo = torch.nn.Linear(16, 1)
        nonddp_bar = torch.nn.Linear(16, 1)
        app_state: AppState = {"ddp": ddp_foo, "nonddp": nonddp_foo}
        # pyrefly: ignore [missing-attribute]
        io_preparer.DEFAULT_MAX_CHUNK_SIZE_BYTES = 4
        snapshot = torchsnapshot.Snapshot.take(
            path=path,
            app_state=app_state,
        )
        snapshot.restore({"ddp": ddp_bar, "nonddp": nonddp_bar})
        tc = unittest.TestCase()
        assert_state_dict_eq(tc, ddp_foo.state_dict(), ddp_bar.state_dict())
        assert_state_dict_eq(tc, nonddp_foo.state_dict(), nonddp_bar.state_dict())

    def test_write_read_with_small_chunk_size(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        with tempfile.TemporaryDirectory() as path:
            pet.elastic_launch(
                lc, entrypoint=self._worker_write_read_with_small_chunk_size
            )(path)
