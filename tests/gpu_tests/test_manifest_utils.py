#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torchsnapshot.manifest_utils import (
    _get_replicated_ranks,
    is_partially_replicated_entry,
)
from torchsnapshot.serialization import NCCL_SUPPORTED_DTYPES
from torchsnapshot.test_utils import _dtensor_test_case, _tensor_test_case

WORLD_SIZE = 4


@instantiate_parametrized_tests
class TestManifestUtils(DTensorTestBase):
    @parametrize("dtype", NCCL_SUPPORTED_DTYPES)
    @skip_if_lt_x_gpu(WORLD_SIZE)
    # pyre-fixme[56]: While applying decorator
    #  `torch.testing._internal.distributed._tensor.common_dtensor.with_comms`: For 1st
    #  argument expected `(object) -> object` but got `(self: TestManifestUtils, dtype:
    #  dtype) -> Any`.
    @with_comms
    # pyre-fixme[3]: Return type must be annotated.
    def test_get_replicated_ranks(self, dtype: torch.dtype):
        logical_path = "foo"
        tensor, entry, wrs = _dtensor_test_case(
            dtype=dtype,
            shape=[16, 16],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=True,
        )
        # pyre-fixme[6]: For 1st argument expected `DTensorEntry` but got `Entry`.
        actual_repranks = _get_replicated_ranks(entry=entry)
        expected_repranks = [[0, 2], [1, 3]]
        assert actual_repranks == expected_repranks

    @parametrize("dtype", NCCL_SUPPORTED_DTYPES)
    @skip_if_lt_x_gpu(WORLD_SIZE)
    # pyre-fixme[56]: While applying decorator
    #  `torch.testing._internal.distributed._tensor.common_dtensor.with_comms`: For 1st
    #  argument expected `(object) -> object` but got `(self: TestManifestUtils, dtype:
    #  dtype) -> Any`.
    @with_comms
    # pyre-fixme[3]: Return type must be annotated.
    def test_is_partially_replicated(self, dtype: torch.dtype):
        logical_path = "foo"
        tensor, entry, wrs = _dtensor_test_case(
            dtype=dtype,
            shape=[16, 16],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=True,
        )
        assert is_partially_replicated_entry(entry=entry)

        # Only replicated
        # pyre-fixme[16]: `Entry` has no attribute `dim_map`.
        entry.dim_map = [-1, -1]
        assert not is_partially_replicated_entry(entry=entry)

        # Only sharded
        entry.dim_map = [0, 1]
        assert not is_partially_replicated_entry(entry=entry)

        tensor, entry, wrs = _tensor_test_case(
            dtype=dtype,
            shape=[16, 16],
            logical_path=logical_path,
            rank=dist.get_rank(),
            replicated=False,
        )

        assert not is_partially_replicated_entry(entry=entry)
