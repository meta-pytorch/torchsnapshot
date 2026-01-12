#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import os
import unittest

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torchsnapshot.pg_wrapper import PGWrapper
from torchsnapshot.test_utils import get_pet_launch_config


class TestPGWrapper(unittest.TestCase):
    @staticmethod
    def _worker(backend: str) -> None:
        tc = unittest.TestCase()
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
        pg_wrapper = PGWrapper(pg=None)
        output_list = [None]
        input_list = [["foo"], ["bar"], ["quaz"]]
        pg_wrapper.scatter_object_list(output_list=output_list, input_list=input_list)
        rank = dist.get_rank()
        tc.assertEqual(output_list, [input_list[rank]])

    def test_scatter_obj_list_gloo(self) -> None:
        lc = get_pet_launch_config(nproc=3)
        pet.elastic_launch(lc, entrypoint=self._worker)("gloo")

    @unittest.skipUnless(torch.cuda.is_available(), "This test requires GPU to run.")
    def test_scatter_obj_list_nccl(self) -> None:
        lc = get_pet_launch_config(nproc=3)
        pet.elastic_launch(lc, entrypoint=self._worker)("nccl")

    def test_scatter_obj_list_dist_uninitialized(self) -> None:
        pg_wrapper = PGWrapper(pg=None)
        output_list = [None]
        input_list = [["foo"]]
        pg_wrapper.scatter_object_list(output_list=output_list, input_list=input_list)
        self.assertEqual(output_list, [input_list[0]])
