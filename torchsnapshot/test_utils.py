#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations

import asyncio
import functools
import unittest
import uuid
from contextlib import contextmanager
from importlib import import_module
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from unittest import mock

import torch
import torch.distributed.launcher as pet
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torchsnapshot.io_preparer import prepare_write
from torchsnapshot.io_types import WriteReq
from torchsnapshot.manifest import Entry

from .serialization import SUPPORTED_QUANTIZED_DTYPES


def tensor_eq(lhs: Union[torch.Tensor, ShardedTensor, DTensor], rhs: Any) -> bool:
    if type(lhs) != type(rhs):
        return False
    if isinstance(lhs, ShardedTensor):
        for l_shard, r_shard in zip(lhs.local_shards(), rhs.local_shards()):
            if not tensor_eq(l_shard.tensor, r_shard.tensor):
                return False
        return True
    elif isinstance(lhs, DTensor):
        if lhs.placements != rhs.placements:
            return False
        if lhs.device_mesh != rhs.device_mesh:
            return False
        lhs = lhs.redistribute(placements=[Replicate()])
        rhs = rhs.redistribute(placements=[Replicate()])
        return torch.allclose(lhs.to_local(), rhs.to_local())
    elif isinstance(lhs, torch.Tensor):
        if lhs.dtype in SUPPORTED_QUANTIZED_DTYPES:
            return torch.allclose(lhs.dequantize(), rhs.dequantize())
        else:
            return torch.allclose(lhs, rhs)
    else:
        raise AssertionError(
            "The lhs operand must be a Tensor, ShardedTensor, or DTensor."
        )


@contextmanager
def _patch_tensor_eq() -> Generator[None, None, None]:
    patchers = [
        mock.patch("torch.Tensor.__eq__", tensor_eq),
        mock.patch(
            "torch.distributed._shard.sharded_tensor.ShardedTensor.__eq__", tensor_eq
        ),
        mock.patch("torch.distributed._tensor.DTensor.__eq__", tensor_eq),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        yield
    finally:
        for patcher in patchers:
            patcher.stop()


def assert_state_dict_eq(
    tc: unittest.TestCase,
    lhs: Dict[Any, Any],
    rhs: Dict[Any, Any],
) -> None:
    """
    assertDictEqual except that it knows how to handle tensors.

    Args:
        tc: The test case.
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.
    """
    with _patch_tensor_eq():
        tc.assertDictEqual(lhs, rhs)


def check_state_dict_eq(lhs: Dict[Any, Any], rhs: Dict[Any, Any]) -> bool:
    """
    dict.__eq__ except that it knows how to handle tensors.

    Args:
        lhs: The left-hand side operand.
        rhs: The right-hand side operand.

    Returns:
        Whether the two dictionaries are equal.
    """
    with _patch_tensor_eq():
        return lhs == rhs


def rand_tensor(
    shape: Union[Tuple[int, ...], List[int], torch.Size],
    dtype: torch.dtype,
    qscheme: Optional[torch.qscheme] = torch.per_tensor_affine,
    channel_axis: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a tensor and initialize randomly for testing purposes.

    Tensors of different dtypes needs to be intialized differently. This
    function provides a unified signature for scenarios in which the random
    range is not a concern.

    Args:
        shape: The shape of the random tensor.
        dtype: The dtype of the random tensor.

    Return:
        The random tensor.
    """
    if dtype.is_floating_point or dtype.is_complex:
        return torch.rand(shape, dtype=dtype)
    elif dtype == torch.bool:
        return torch.randint(2, shape, dtype=dtype)
    elif dtype in SUPPORTED_QUANTIZED_DTYPES:
        if qscheme == torch.per_tensor_affine:
            return torch.quantize_per_tensor(
                torch.rand(shape), scale=0.1, zero_point=10, dtype=dtype
            )
        elif qscheme == torch.per_channel_affine:
            return torch.quantize_per_channel(
                torch.rand(shape),
                torch.rand(shape[channel_axis or 0]),
                torch.randint(128, (shape[channel_axis or 0],)),
                axis=channel_axis or 0,
                dtype=dtype,
            )
        else:
            raise AssertionError(f"Unrecognized qscheme: {qscheme}.")
    else:
        return torch.randint(torch.iinfo(dtype).max, shape, dtype=dtype)


def tensor_local_sz_bytes(tensor: torch.Tensor) -> int:
    if type(tensor) == torch.Tensor:
        return tensor.nelement() * tensor.element_size()
    elif type(tensor) == ShardedTensor:
        sz = 0
        for shard in tensor.local_shards():
            sz += tensor_local_sz_bytes(shard.tensor)
        return sz
    elif type(tensor) == DTensor:
        return tensor_local_sz_bytes(tensor.to_local())
    else:
        raise AssertionError(
            f"The input must be a Tensor, ShardedTensor, or DTensor (got: {type(tensor)}."
        )


def get_pet_launch_config(nproc: int) -> pet.LaunchConfig:
    """
    Initialize pet.LaunchConfig for single-node, multi-rank tests.

    Args:
        nproc: The number of processes to launch.

    Returns:
        An instance of pet.LaunchConfig for single-node, multi-rank tests.
    """
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )


@contextmanager
def _tempfile_pet_launch_config(nproc: int) -> Generator[pet.LaunchConfig, None, None]:
    with NamedTemporaryFile() as f:
        yield pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=nproc,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_configs={"timeout": 10, "store_type": "file"},
            rdzv_endpoint=f.name,
            max_restarts=0,
            monitor_interval=1,
        )


def _launch_pad(mod_name: str, func_name: str, args, kwargs) -> None:
    mod = import_module(mod_name)
    func = getattr(mod, func_name)
    func.__wrapped__(*args, **kwargs)


def run_with_pet(nproc: int) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def _run_with_pet(func: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> None:
            with _tempfile_pet_launch_config(nproc=nproc) as lc:
                pet.elastic_launch(lc, entrypoint=_launch_pad)(
                    func.__module__, func.__name__, args, kwargs
                )

        return wrapper

    return _run_with_pet


def _async_launch_pad(mod_name: str, func_name: str, args, kwargs) -> None:
    mod = import_module(mod_name)
    func = getattr(mod, func_name)
    coro = func.__wrapped__(*args, **kwargs)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_with_pet_async(
    nproc: int,
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., None]]:
    def _run_with_pet(func: Callable[..., Awaitable[None]]) -> Callable[..., None]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> None:
            with _tempfile_pet_launch_config(nproc=nproc) as lc:
                pet.elastic_launch(lc, entrypoint=_async_launch_pad)(
                    func.__module__, func.__name__, args, kwargs
                )

        return wrapper

    return _run_with_pet


T = TypeVar("T")


def async_test(coro: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator for testing asynchronous code.
    Once we drop support for Python 3.7.x, we can use `unittest.IsolatedAsyncioTestCase` instead.

    Usage:
        class MyTest(unittest.TestCase):
            @async_test
            async def test_x(self):
                ...
    """

    def wrapper(*args, **kwargs) -> T:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


def _tensor_test_case(
    dtype: torch.dtype,
    shape: List[int],
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[torch.Tensor, Entry, List[WriteReq]]:
    tensor = rand_tensor(shape, dtype=dtype)
    entry, wrs = prepare_write(
        obj=tensor, logical_path=logical_path, rank=rank, replicated=replicated
    )
    return tensor, entry, wrs


def _dtensor_test_case(
    dtype: torch.dtype,
    shape: List[int],
    logical_path: str,
    rank: int,
    replicated: bool,
) -> Tuple[DTensor, Entry, List[WriteReq]]:
    # WORLD_SIZE needs to be at least 4
    mesh = DeviceMesh("cuda", mesh=[[0, 1], [2, 3]])
    placements = [Replicate(), Shard(0)]
    local_tensor = rand_tensor(shape, dtype=dtype)
    dtensor = distribute_tensor(
        tensor=local_tensor, device_mesh=mesh, placements=placements
    )

    entry, wrs = prepare_write(
        obj=dtensor, logical_path=logical_path, rank=rank, replicated=replicated
    )
    return dtensor, entry, wrs
