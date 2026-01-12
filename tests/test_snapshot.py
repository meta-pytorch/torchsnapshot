#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch
from torchsnapshot import Snapshot, StateDict
from torchsnapshot.manifest import PrimitiveEntry
from torchsnapshot.test_utils import check_state_dict_eq


@pytest.mark.usefixtures("toggle_batching")
def test_state_dict(tmp_path: Path) -> None:
    foo = StateDict(
        {
            "a": torch.rand(40, 40),
            "b": torch.rand(40, 40),
            "c": 42,
            "d/e": 43,
            "[@x]->&y^%": {"(z)": 44},
        },
    )
    bar = StateDict(
        {
            "a": torch.rand(40, 40),
            "b": torch.rand(40, 40),
            "c": 42,
            "d/e": 43,
            "[@x]->&y^%": {"(z)": 44},
        },
    )
    assert not check_state_dict_eq(foo.state_dict(), bar.state_dict())
    assert type(foo.state_dict()) == dict

    snapshot = Snapshot.take(str(tmp_path), {"foo": foo})
    snapshot.restore({"foo": bar})
    assert check_state_dict_eq(foo.state_dict(), bar.state_dict())


@pytest.mark.usefixtures("toggle_batching")
def test_nn_linear(tmp_path: Path) -> None:
    foo = torch.nn.Linear(128, 64)
    bar = torch.nn.Linear(128, 64)
    assert not check_state_dict_eq(foo.state_dict(), bar.state_dict())

    snapshot = Snapshot.take(str(tmp_path), {"foo": foo})
    snapshot.restore({"foo": bar})
    assert check_state_dict_eq(foo.state_dict(), bar.state_dict())


@pytest.mark.usefixtures("toggle_batching")
def test_nn_sequential(tmp_path: Path) -> None:
    foo = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    bar = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    assert not check_state_dict_eq(foo.state_dict(), bar.state_dict())

    snapshot = Snapshot.take(str(tmp_path), {"foo": foo})
    snapshot.restore({"foo": bar})
    assert check_state_dict_eq(foo.state_dict(), bar.state_dict())


@pytest.mark.usefixtures("toggle_batching")
def test_strict_false(tmp_path: Path) -> None:
    foo = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    bar = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
        torch.nn.Linear(16, 8),
    )
    assert not check_state_dict_eq(foo.state_dict(), bar.state_dict())

    expected_dict = foo.state_dict()
    snapshot = Snapshot.take(str(tmp_path), {"foo": foo})
    snapshot.restore({"foo": bar}, strict=False)
    assert check_state_dict_eq(foo.state_dict(), expected_dict)


@pytest.mark.usefixtures("toggle_batching")
def test_adagrad(tmp_path: Path) -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    expected = copy.deepcopy(optim.state_dict())

    snapshot = Snapshot.take(str(tmp_path), {"optim": optim})
    snapshot.restore({"optim": optim})

    assert check_state_dict_eq(optim.state_dict(), expected)


@pytest.mark.usefixtures("toggle_batching")
def test_model_and_optim(tmp_path: Path) -> None:
    torch.manual_seed(42)
    foo = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )

    # Initialize dst with the different seeds across ranks
    torch.manual_seed(24)
    bar = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    assert not check_state_dict_eq(foo.state_dict(), bar.state_dict())

    # Need to step and zero_grad in order to initialize all the optimizer parameters
    foo_optim = torch.optim.AdamW(foo.parameters(), lr=0.01)
    foo_optim.step(closure=None)
    foo_optim.zero_grad(set_to_none=True)

    bar_optim = torch.optim.AdamW(bar.parameters(), lr=0.02)
    bar_optim.step(closure=None)
    bar_optim.zero_grad(set_to_none=True)

    assert not check_state_dict_eq(foo_optim.state_dict(), bar_optim.state_dict())

    snapshot = Snapshot.take(str(tmp_path), {"foo": foo, "optim": foo_optim})
    snapshot.restore({"foo": bar, "optim": bar_optim})

    assert check_state_dict_eq(foo_optim.state_dict(), bar_optim.state_dict())
    assert check_state_dict_eq(foo.state_dict(), bar.state_dict())


def test_invalid_app_state(tmp_path: Path) -> None:
    not_stateful = 1
    app_state = {"optim": not_stateful}

    with pytest.raises(TypeError):
        Snapshot.take(path=str(tmp_path), app_state=app_state)

    snapshot = Snapshot(path=str(tmp_path))
    with pytest.raises(TypeError):
        snapshot.restore(app_state=app_state)


@pytest.mark.usefixtures("toggle_batching")
def test_app_state_with_primitive_types(tmp_path: Path) -> None:
    state = StateDict(
        int_key=100,
        float_key=3.14,
        str_key="some_string",
        bool_key=True,
        bytes_key=b"\x00\x10",
    )
    restored_state = StateDict(
        int_key=None,
        float_key=None,
        str_key=None,
        bool_key=None,
        bytes_key=None,
    )

    # pyre-fixme[53]: Captured variable `snapshot` is not annotated.
    def _assert_primitive_entry_with_type(
        location_key: str, expected_type_name: str
    ) -> None:
        assert (
            isinstance(snapshot.metadata.manifest[location_key], PrimitiveEntry)
            and snapshot.metadata.manifest[location_key].type == expected_type_name
        )

    snapshot = Snapshot.take(path=str(tmp_path), app_state={"key": state})

    _assert_primitive_entry_with_type("0/key/int_key", "int")
    _assert_primitive_entry_with_type("0/key/str_key", "str")
    _assert_primitive_entry_with_type("0/key/bool_key", "bool")
    _assert_primitive_entry_with_type("0/key/bytes_key", "bytes")
    _assert_primitive_entry_with_type("0/key/float_key", "float")

    assert snapshot.metadata.manifest["0/key/float_key"].readable == str(
        state["float_key"]
    )

    snapshot.restore({"key": restored_state})
    assert state == restored_state


@pytest.mark.usefixtures("toggle_batching")
def test_different_state_dict_structure_on_load(tmp_path: Path) -> None:
    class TestStateful:
        def __init__(self) -> None:
            self.objs: List[Any] = []

        def state_dict(self) -> Dict[str, Any]:
            return {"objs": self.objs}

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            self.objs = state_dict["objs"]

    src = TestStateful()
    dst = TestStateful()

    for _ in range(10):
        src.objs.append(torch.rand(64, 64))
    src.objs.append([torch.rand(64, 64) for _ in range(10)])

    assert not check_state_dict_eq(src.state_dict(), dst.state_dict())
    snapshot = Snapshot.take(app_state={"state": src}, path=str(tmp_path))
    snapshot.restore(app_state={"state": dst})
    assert check_state_dict_eq(src.state_dict(), dst.state_dict())


@pytest.mark.usefixtures("toggle_batching")
def test_snapshot_metadata_error(tmp_path: Path) -> None:
    mock_storage_plugin = MagicMock()
    mock_event_loop = MagicMock()
    mock_storage_plugin.sync_read.side_effect = Exception(
        "Mock error reading from storage"
    )
    with pytest.raises(
        expected_exception=RuntimeError,
        match=(
            "Failed to read .snapshot_metadata. "
            "Ensure path to snapshot is correct, "
            "otherwise snapshot is likely incomplete or corrupted."
        ),
    ):
        Snapshot._read_snapshot_metadata(mock_storage_plugin, mock_event_loop)
