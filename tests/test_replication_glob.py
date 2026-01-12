#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
from torchsnapshot import Snapshot
from torchsnapshot.manifest_utils import is_fully_replicated_entry
from torchsnapshot.test_utils import run_with_pet

_WORLD_SIZE: int = 2


class _TestStateful:
    def state_dict(self) -> Dict[str, Any]:
        return {
            "foo": torch.Tensor(1),
            "bar": torch.Tensor(1),
            "baz": [torch.Tensor(1), torch.Tensor(1)],
            "qux": {"quux": torch.Tensor(1), "quuz": torch.Tensor(1)},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError()


@pytest.mark.parametrize(
    "replication_globs, expected_replicated_paths",
    [
        (
            [["**"]] * _WORLD_SIZE,
            [
                "0/my_stateful/foo",
                "0/my_stateful/bar",
                "0/my_stateful/baz/0",
                "0/my_stateful/baz/1",
                "0/my_stateful/qux/quux",
                "0/my_stateful/qux/quuz",
            ],
        ),
        (
            [["my_stateful/baz/*", "my_stateful/qux/*"]] * _WORLD_SIZE,
            [
                "0/my_stateful/baz/0",
                "0/my_stateful/baz/1",
                "0/my_stateful/qux/quux",
                "0/my_stateful/qux/quuz",
            ],
        ),
        (
            [
                ["my_stateful/foo", "my_stateful/qux/*"],
                ["my_stateful/foo", "my_stateful/bax/*"],
            ],
            ["0/my_stateful/foo"],
        ),
    ],
)
@run_with_pet(nproc=_WORLD_SIZE)
def test_replication_glob(
    replication_globs: List[List[str]],
    expected_replicated_paths: List[str],
    tmp_path: Path,
) -> None:
    dist.init_process_group(backend="gloo")
    app_state = {"my_stateful": _TestStateful()}
    snapshot = Snapshot.take(
        path=str(tmp_path),
        app_state=app_state,
        replicated=replication_globs[dist.get_rank()],
    )
    replicated_paths = [
        path
        for path, entry in snapshot.get_manifest().items()
        if is_fully_replicated_entry(entry)
    ]
    assert set(replicated_paths) == set(expected_replicated_paths)


@pytest.mark.parametrize(
    "global_replicated, expected_replicated",
    [
        (
            [
                ["my_stateful/foo", "my_stateful/qux"],
                ["my_stateful/foo", "my_stateful/qux"],
            ],
            ["my_stateful/foo", "my_stateful/qux"],
        ),
        (
            [
                ["my_stateful/foo", "my_stateful/qux"],
                ["my_stateful/foo", "my_stateful/baz"],
            ],
            ["my_stateful/foo"],
        ),
        (
            [
                ["my_stateful/foo"],
                ["my_stateful/qux"],
            ],
            [],
        ),
    ],
)
def test_coalesce_replicated(
    global_replicated: List[List[str]],
    expected_replicated: List[str],
) -> None:
    assert sorted(
        Snapshot._coalesce_replicated(global_replicated=global_replicated)
    ) == sorted(expected_replicated)
