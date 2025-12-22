#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchsnapshot
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torchsnapshot.test_utils import get_pet_launch_config


class ReadObjectTest(unittest.TestCase):
    def test_read_object(self) -> None:
        state = torchsnapshot.StateDict(
            foo=42,
            bar=torch.randn(20, 20),
        )

        with tempfile.TemporaryDirectory() as path:
            snapshot = torchsnapshot.Snapshot.take(
                path=path, app_state={"state": state}
            )

            self.assertEqual(snapshot.read_object("0/state/foo"), 42)
            self.assertEqual(snapshot.read_object("0/state/foo", 777), 42)

            baz = torch.randn(20, 20)
            self.assertFalse(torch.allclose(baz, state["bar"]))

            loaded_bar = snapshot.read_object("0/state/bar", baz)
            self.assertEqual(id(loaded_bar), id(baz))
            self.assertNotEqual(id(loaded_bar), id(state["bar"]))
            self.assertTrue(torch.allclose(baz, state["bar"]))

    @staticmethod
    def _test_read_sharded_tensor() -> None:
        tc = unittest.TestCase()
        dist.init_process_group(backend="gloo")
        torch.manual_seed(42 + dist.get_rank())

        # pyre-ignore [28]
        spec = ChunkShardingSpec(
            dim=0,
            placements=[f"rank:{rank}/cpu" for rank in range(dist.get_world_size())],
        )
        foo = sharded_tensor.empty(spec, 20_000, 128)
        for shard in foo.local_shards():
            torch.nn.init.uniform_(shard.tensor)

        bar = sharded_tensor.empty(spec, 20_000, 128)
        for shard in bar.local_shards():
            torch.nn.init.uniform_(shard.tensor)

        for foo_shard, bar_shard in zip(foo.local_shards(), bar.local_shards()):
            tc.assertFalse(torch.allclose(foo_shard.tensor, bar_shard.tensor))

        # Create temp directory on rank 0 and broadcast path to all ranks
        if dist.get_rank() == 0:
            path = tempfile.mkdtemp()
        else:
            path = ""
        path_list = [path]
        dist.broadcast_object_list(path_list, src=0)
        path = path_list[0]

        try:
            snapshot = torchsnapshot.Snapshot.take(
                path=path, app_state={"state": torchsnapshot.StateDict(foo=foo)}
            )
            snapshot.read_object("0/state/foo", obj_out=bar)
            baz = snapshot.read_object("0/state/foo")

            for foo_shard, bar_shard in zip(foo.local_shards(), bar.local_shards()):
                tc.assertTrue(torch.allclose(foo_shard.tensor, bar_shard.tensor))

            tc.assertEqual(baz.shape, torch.Size([20_000, 128]))

            gathered_foo_tensor = torch.empty(20_000, 128)
            if dist.get_rank() == 0:
                foo.gather(dst=0, out=gathered_foo_tensor)
                tc.assertTrue(torch.allclose(baz, gathered_foo_tensor))
            else:
                foo.gather(dst=0, out=None)
        finally:
            # Clean up the temp directory on rank 0
            dist.barrier()
            if dist.get_rank() == 0:
                import shutil

                shutil.rmtree(path, ignore_errors=True)

    def test_read_sharded_tensor(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_read_sharded_tensor)()

    @staticmethod
    def _quantize(path: str, tensor: torch.Tensor, tracing: bool) -> torch.Tensor:
        return torch.quantize_per_tensor(tensor, 0.1, 10, torch.qint8)

    @classmethod
    def _test_read_sharded_tensor_into_tensor(cls, quantized: bool) -> None:
        tc = unittest.TestCase()
        dist.init_process_group(backend="gloo")
        torch.manual_seed(42 + dist.get_rank())

        # pyre-ignore [28]
        spec = ChunkShardingSpec(
            dim=0,
            placements=[f"rank:{rank}/cpu" for rank in range(dist.get_world_size())],
        )
        foo = sharded_tensor.empty(spec, 20_000, 128)
        for shard in foo.local_shards():
            torch.nn.init.uniform_(shard.tensor)

        # Gather the sharded tensor for ease of comparison
        if dist.get_rank() == 0:
            foo_gathered = torch.empty(20_000, 128)
            foo.gather(dst=0, out=foo_gathered)
        else:
            foo_gathered = torch.empty(42)
            foo.gather(dst=0, out=None)

        # Create a tensor into which the sharded tensor will be loaded
        bar = torch.rand_like(foo_gathered)

        if quantized:
            foo_gathered = torch.quantize_per_tensor(foo_gathered, 0.1, 10, torch.qint8)
            bar = torch.quantize_per_tensor(bar, 0.1, 10, torch.qint8)

        # Control test: these two tensors should be different
        tc.assertFalse(
            torch.allclose(torch.dequantize(foo_gathered), torch.dequantize(bar))
        )

        with tempfile.TemporaryDirectory() as path:
            _custom_tensor_prepare_func = cls._quantize if quantized else None
            snapshot = torchsnapshot.Snapshot.take(
                path=path,
                app_state={"state": torchsnapshot.StateDict(foo=foo)},
                _custom_tensor_prepare_func=_custom_tensor_prepare_func,
            )
            if dist.get_rank() == 0:
                snapshot.read_object("0/state/foo", obj_out=bar)
                tc.assertTrue(
                    torch.allclose(
                        torch.dequantize(foo_gathered), torch.dequantize(bar)
                    )
                )

    def test_read_sharded_tensor_into_tensor(self) -> None:
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_read_sharded_tensor_into_tensor)(
            True  # quantize=True
        )
        lc = get_pet_launch_config(nproc=4)
        pet.elastic_launch(lc, entrypoint=self._test_read_sharded_tensor_into_tensor)(
            False  # quantized=False
        )
