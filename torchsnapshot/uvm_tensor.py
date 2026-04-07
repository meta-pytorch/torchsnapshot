#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List

import torch

_UVM_TENSOR_AVAILABLE = False

try:
    # pyre-fixme[21]: Could not find module `fbgemm_gpu`.
    import fbgemm_gpu  # @manual  # noqa
except Exception:
    pass

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils")
except Exception:
    pass


try:
    #  Tensor`; used as `OpOverloadPacket`.
    new_managed_tensor = torch.ops.fbgemm.new_managed_tensor
    #  `OpOverloadPacket`.
    is_uvm_tensor = torch.ops.fbgemm.is_uvm_tensor
    #  `OpOverloadPacket`.
    uvm_to_cpu = torch.ops.fbgemm.uvm_to_cpu

    _UVM_TENSOR_AVAILABLE = True
except AttributeError:

    def new_managed_tensor(t: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        raise NotImplementedError()

    def is_uvm_tensor(t: torch.Tensor) -> bool:
        return False

    def uvm_to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t


__all__ = ["is_uvm_tensor", "uvm_to_cpu"]
