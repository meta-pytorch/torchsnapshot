#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsnapshot import Snapshot, Stateful
from torchsnapshot.test_utils import get_pet_launch_config


class DDPInferReplicatedTest(unittest.TestCase):
    @staticmethod
    def _worker_helper(replicated: List[str], expected_replicated: List[str]) -> None:
        dist.init_process_group(backend="gloo")
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 1))
        inferred_replicated = Snapshot._infer_replicated(
            replicated=replicated, app_state={"ddp": DDP(model), "nonddp": model}
        )

        unittest.TestCase().assertCountEqual(expected_replicated, inferred_replicated)

    def test_with_no_glob(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        replicated = []
        expected_replicated = ["ddp/**"]
        pet.elastic_launch(
            lc,
            entrypoint=DDPInferReplicatedTest._worker_helper,
        )(replicated, expected_replicated)

    def test_with_all_glob(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        replicated = ["**"]
        expected_replicated = ["**"]
        pet.elastic_launch(
            lc,
            entrypoint=DDPInferReplicatedTest._worker_helper,
        )(replicated, expected_replicated)

    def test_with_nonddp_glob(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        replicated = ["nonddp/**"]
        expected_replicated = ["ddp/**", "nonddp/**"]
        pet.elastic_launch(
            lc,
            entrypoint=DDPInferReplicatedTest._worker_helper,
        )(replicated, expected_replicated)

    @staticmethod
    def _worker_with_params_to_ignore(
        replicated: List[str], expected_replicated: List[str]
    ) -> None:
        dist.init_process_group(backend="gloo")
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 1))
        DDP._set_params_and_buffers_to_ignore_for_model(
            model, ["module.0.bias", "module.0.weight"]
        )
        ddp_model = DDP(model)
        # pyrefly: ignore [bad-assignment]
        app_state: Dict[str, Stateful] = {"ddp": ddp_model, "nonddp": model}

        inferred_replicated = Snapshot._infer_replicated(
            replicated=replicated, app_state=app_state
        )
        unittest.TestCase().assertCountEqual(expected_replicated, inferred_replicated)

    def test_with_params_to_ignore(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        replicated = []
        expected_replicated = ["ddp/module.1.bias", "ddp/module.1.weight"]
        pet.elastic_launch(
            lc,
            entrypoint=DDPInferReplicatedTest._worker_with_params_to_ignore,
        )(replicated, expected_replicated)

    @staticmethod
    def _worker_with_params_to_ignore_and_all_glob(
        replicated: List[str], expected_replicated: List[str]
    ) -> None:
        dist.init_process_group(backend="gloo")
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 1))
        DDP._set_params_and_buffers_to_ignore_for_model(
            model, ["module.0.bias", "module.0.weight"]
        )
        ddp_model = DDP(model)
        # pyrefly: ignore [bad-assignment]
        app_state: Dict[str, Stateful] = {"ddp": ddp_model, "nonddp": model}
        inferred_replicated = Snapshot._infer_replicated(
            replicated=replicated, app_state=app_state
        )
        unittest.TestCase().assertCountEqual(expected_replicated, inferred_replicated)

    def test_with_params_to_ignore_and_all_glob(self) -> None:
        lc = get_pet_launch_config(nproc=2)
        replicated = ["**"]
        expected_replicated = ["**"]
        pet.elastic_launch(
            lc,
            entrypoint=DDPInferReplicatedTest._worker_with_params_to_ignore_and_all_glob,
        )(replicated, expected_replicated)
