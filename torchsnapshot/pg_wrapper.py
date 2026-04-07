#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]

from typing import Any, List, Optional

import torch.distributed as dist


class PGWrapper:
    """
    A wrapper around ProcessGroup that allows collectives to be issued in a
    consistent fashion regardless of the following scenarios:

        pg is None, distributed is initialized:     use WORLD as pg
        pg is None, distributed is not initialized: single process app
        pg is not None:                             use pg
    """

    def __init__(self, pg: Optional[dist.ProcessGroup]) -> None:
        if pg is None and dist.is_initialized():
            self.pg = dist.group.WORLD
        else:
            self.pg = pg

    def get_rank(self) -> int:
        if self.pg is None:
            return 0
        return dist.get_rank(group=self.pg)

    def get_world_size(self) -> int:
        if self.pg is None:
            return 1
        return dist.get_world_size(group=self.pg)

    def barrier(self) -> None:
        if self.pg is None:
            return
        dist.barrier(group=self.pg)

    def broadcast_object_list(self, obj_list: List[Any], src: int = 0) -> None:
        if self.pg is None:
            return
        dist.broadcast_object_list(obj_list, src=src, group=self.pg)

    def all_gather_object(self, obj_list: List[Any], obj: Any) -> None:
        if self.pg is None:
            obj_list[0] = obj
            return
        dist.all_gather_object(obj_list, obj, group=self.pg)

    def scatter_object_list(
        self,
        output_list: List[Any],
        input_list: Optional[List[Any]],
        src: int = 0,
    ) -> None:
        rank = self.get_rank()
        world_size = self.get_world_size()
        if rank == src:
            if input_list is None:
                raise RuntimeError(
                    "The src rank's input_list for scatter_object_list must not be None."
                )
            if len(input_list) != world_size:
                raise RuntimeError(
                    f"The length of input_list {len(input_list)} for scatter_object_list "
                    f"must be the same as the process group's world size ({world_size})."
                )
        else:
            input_list = [None] * world_size

        if self.pg is None:
            output_list[0] = input_list[0]
            return

        # scatter_object_list does not yet support NCCL backend
        if dist.get_backend(self.pg) == "nccl":
            self.broadcast_object_list(obj_list=input_list, src=src)
            output_list[0] = input_list[rank]
            return

        dist.scatter_object_list(output_list, input_list, src=src, group=self.pg)
