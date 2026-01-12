#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torchsnapshot.io_preparers.tensor import TensorIOPreparer
from torchsnapshot.io_types import Future, ReadReq, WriteReq
from torchsnapshot.knobs import get_max_chunk_size_bytes
from torchsnapshot.manifest import ChunkedTensorEntry, Shard
from torchsnapshot.serialization import dtype_to_string

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    offsets: List[int]
    sizes: List[int]
    dtype: str


class ChunkedTensorIOPreparer:
    @staticmethod
    def chunk_tensor(
        tensor: torch.Tensor,
        chunking_dim: int = 0,
        chunk_sz_bytes: Optional[int] = None,
    ) -> List[Chunk]:
        chunk_sz_bytes = chunk_sz_bytes or get_max_chunk_size_bytes()

        # for 0-d case, reshape to 1-d
        if tensor.ndim == 0:
            tensor = tensor.view(-1)

        tensor_sz_bytes = tensor.numel() * tensor.element_size()
        n_chunks = math.ceil(tensor_sz_bytes / chunk_sz_bytes)
        tensor_chunks = torch.chunk(tensor, chunks=n_chunks, dim=chunking_dim)

        curr_offsets = [0] * tensor.ndim
        chunking_instruction = []
        for i in range(len(tensor_chunks)):
            tensor_chunk_sizes = list(tensor_chunks[i].shape)
            chunking_instruction.append(
                Chunk(
                    offsets=curr_offsets[:],
                    sizes=tensor_chunk_sizes,
                    dtype=str(tensor.dtype),
                )
            )
            curr_offsets[chunking_dim] += tensor_chunk_sizes[chunking_dim]
        return chunking_instruction

    @staticmethod
    def _get_subtensor_view(
        tensor: torch.Tensor, chunk: Union[Shard, Chunk]
    ) -> torch.Tensor:
        # for 0-d case, reshape to 1-d
        result = tensor.view(-1) if tensor.ndim == 0 else tensor

        for d in range(len(chunk.sizes)):
            result = result.narrow(d, chunk.offsets[d], chunk.sizes[d])
        return result

    @classmethod
    def prepare_write(
        cls,
        storage_path: str,
        tensor: torch.Tensor,
        chunking_instruction: List[Chunk],
        is_async_snapshot: bool = False,
        _tensor_prepare_func: Optional[
            Callable[[torch.Tensor, bool], torch.Tensor]
        ] = None,
    ) -> Tuple[ChunkedTensorEntry, List[WriteReq]]:
        write_reqs = []
        chunks = []
        for chunk in chunking_instruction:
            suffix = "_".join(str(x) for x in chunk.offsets)
            chunk_entry, chunk_write_reqs = TensorIOPreparer.prepare_write(
                storage_path=f"{storage_path}_{suffix}",
                tensor=cls._get_subtensor_view(tensor, chunk),
                is_async_snapshot=is_async_snapshot,
                _tensor_prepare_func=_tensor_prepare_func,
            )
            chunks.append(
                Shard(offsets=chunk.offsets, sizes=chunk.sizes, tensor=chunk_entry)
            )
            write_reqs += chunk_write_reqs
        chunked_entry = ChunkedTensorEntry(
            dtype=dtype_to_string(tensor.dtype),
            shape=list(tensor.shape),
            chunks=chunks,
            replicated=False,
        )
        return chunked_entry, write_reqs

    @classmethod
    def prepare_read(
        cls,
        entry: ChunkedTensorEntry,
        tensor_out: Optional[torch.Tensor] = None,
        buffer_size_limit_bytes: Optional[int] = None,
    ) -> Tuple[List[ReadReq], Future[torch.Tensor]]:
        if tensor_out is None or not TensorIOPreparer.can_load_inplace(
            entry=entry, obj=tensor_out
        ):
            tensor_out = TensorIOPreparer.empty_tensor_from_entry(entry)
        read_reqs = []
        for chunk in entry.chunks:
            tensor_out_chunk = cls._get_subtensor_view(tensor_out, chunk)
            chunk_read_reqs, _ = TensorIOPreparer.prepare_read(
                chunk.tensor, tensor_out_chunk, buffer_size_limit_bytes
            )
            read_reqs += chunk_read_reqs
        return read_reqs, Future(obj=tensor_out)
