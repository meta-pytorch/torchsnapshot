#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2, 3]: allow `Any` in function signatures
import os
from collections import OrderedDict
from typing import Any, Dict

import pytest
from torchsnapshot.flatten import _encode, flatten, inflate
from torchsnapshot.manifest import DictEntry, Entry, ListEntry, OrderedDictEntry


@pytest.fixture
def test_obj() -> Any:
    return {
        "foo": 0,
        "bar": 1,
        "baz": [
            2,
            3,
            {"qux": 4, "quxx": [5, OrderedDict(quuz=6, corge=[7, 8, 9])]},
        ],
        "x/y": {"%a/b": 10},
        "": {"": []},
        "dict_with_colliding_keys": {"0": {"1": "foo", 1: "bar"}, 0: "baz"},
        "dict_with_mixed_type_keys": {0: {"0": "foo", 1: "bar"}, "1": "baz"},
        "long_list": list(range(100)),
    }


@pytest.fixture
def expected_manifest(prefix: str) -> Dict[str, Entry]:
    manifest: Dict[str, Entry] = {
        "baz": ListEntry(),
        "baz/2": DictEntry(keys=["qux", "quxx"]),
        "baz/2/quxx": ListEntry(),
        "baz/2/quxx/1": OrderedDictEntry(keys=["quuz", "corge"]),
        "baz/2/quxx/1/corge": ListEntry(),
        "x%2Fy": DictEntry(keys=["%a/b"]),
        "": DictEntry(keys=[""]),
        "/": ListEntry(),
        # dict_with_colliding_keys should not be flattened
        "dict_with_mixed_type_keys": DictEntry(keys=[0, "1"]),
        "dict_with_mixed_type_keys/0": DictEntry(keys=["0", 1]),
        "long_list": ListEntry(),
    }
    manifest = {f"{_encode(prefix)}/{k}": v for k, v in manifest.items()}
    manifest[_encode(prefix)] = DictEntry(
        keys=[
            "foo",
            "bar",
            "baz",
            "x/y",
            "",
            "dict_with_colliding_keys",
            "dict_with_mixed_type_keys",
            "long_list",
        ]
    )
    return manifest


@pytest.fixture
def expected_flattened(prefix: str) -> Dict[str, Any]:
    flattened = {
        "foo": 0,
        "bar": 1,
        "baz/0": 2,
        "baz/1": 3,
        "baz/2/qux": 4,
        "baz/2/quxx/0": 5,
        "baz/2/quxx/1/quuz": 6,
        "baz/2/quxx/1/corge/0": 7,
        "baz/2/quxx/1/corge/1": 8,
        "baz/2/quxx/1/corge/2": 9,
        "x%2Fy/%25a%2Fb": 10,
        # dict_with_colliding_keys should not be flattened
        "dict_with_colliding_keys": {
            "0": {"1": "foo", 1: "bar"},
            0: "baz",
        },
        "dict_with_mixed_type_keys/0/0": "foo",
        "dict_with_mixed_type_keys/0/1": "bar",
        "dict_with_mixed_type_keys/1": "baz",
    }
    flattened.update({f"long_list/{i}": i for i in range(100)})
    flattened = {f"{_encode(prefix)}/{k}": v for k, v in flattened.items()}
    return flattened


@pytest.mark.parametrize(
    "prefix", ["", "prefix_without_slashes", "prefix/with/slashes"]
)
def test_flatten_inflate(
    test_obj: Any,
    prefix: str,
    expected_manifest: Dict[str, Any],
    expected_flattened: Dict[str, Any],
) -> None:
    manifest, flattened = flatten(obj=test_obj, prefix=prefix)
    assert manifest == expected_manifest
    assert flattened == expected_flattened

    inflated = inflate(manifest, flattened, prefix=prefix)
    assert inflated == test_obj


@pytest.mark.parametrize(
    "prefix", ["", "prefix_with_no_slashes", "prefix/with/slashes"]
)
def test_non_flattenable_object(prefix: str) -> None:
    """
    When there's a collision among the string representations of the keys,
    the dict should not be flattened.
    """
    OBJ = 42
    EXPECTED_MANIFEST = {}
    EXPECTED_FLATTENED = {_encode(prefix): 42}

    manifest, flattened = flatten(obj=OBJ, prefix=prefix)
    assert manifest == EXPECTED_MANIFEST
    assert flattened == EXPECTED_FLATTENED

    inflated = inflate(manifest, flattened, prefix=prefix)
    assert inflated == 42


# NOTE: new tests should be introduced by augmenting the fixtures used by
# test_flatten_inflate(). The test below are kept as-is when
# test_flatten_inflate() was first introduced to ensure backward compatibility.
# They can be removed once test_flatten_inflate() is merged.

# pyre-ignore
_OBJ = {
    "foo": 0,
    "bar": 1,
    "baz": [
        2,
        3,
        {"qux": 4, "quxx": [5, OrderedDict(quuz=6, corge=[7, 8, 9])]},
    ],
    "x/y": {"%a/b": 10},
}

_EXPECTED_MANIFEST = {
    "": DictEntry(keys=["foo", "bar", "baz", "x/y"]),
    "baz": ListEntry(),
    "baz/2": DictEntry(keys=["qux", "quxx"]),
    "baz/2/quxx": ListEntry(),
    "baz/2/quxx/1": OrderedDictEntry(keys=["quuz", "corge"]),
    "baz/2/quxx/1/corge": ListEntry(),
    "x%2Fy": DictEntry(keys=["%a/b"]),
}

_EXPECTED_FLATTENED = {
    "foo": 0,
    "bar": 1,
    "baz/0": 2,
    "baz/1": 3,
    "baz/2/qux": 4,
    "baz/2/quxx/0": 5,
    "baz/2/quxx/1/quuz": 6,
    "baz/2/quxx/1/corge/0": 7,
    "baz/2/quxx/1/corge/1": 8,
    "baz/2/quxx/1/corge/2": 9,
    "x%2Fy/%25a%2Fb": 10,
}


def test_simple_flatten_with_prefix() -> None:
    manifest, flattened = flatten(obj=_OBJ, prefix="my/prefix")
    expected_manifest = {
        (
            os.path.join(_encode("my/prefix"), path) if path else _encode("my/prefix")
        ): entry
        for path, entry in _EXPECTED_MANIFEST.items()
    }
    expected_flattened = {
        os.path.join(_encode("my/prefix"), path): entry
        for path, entry in _EXPECTED_FLATTENED.items()
    }
    assert manifest == expected_manifest
    assert flattened == expected_flattened


def test_simple_inflate_with_prefix() -> None:
    manifest, flattened = flatten(obj=_OBJ, prefix="my/prefix")
    inflated = inflate(manifest, flattened, prefix="my/prefix")
    assert inflated == _OBJ


def test_keys_with_colliding_str_repr() -> None:
    """
    When there's a collision among the string representations of the keys,
    the dict should not be flattened.
    """
    OBJ = {"0": {"1": "foo", 1: "bar"}, 0: "baz"}
    EXPECTED_MANIFEST = {}
    EXPECTED_FLATTENED = {"": {"0": {"1": "foo", 1: "bar"}, 0: "baz"}}

    manifest, flattened = flatten(obj=OBJ, prefix="")
    assert manifest == EXPECTED_MANIFEST
    assert flattened == EXPECTED_FLATTENED

    inflated = inflate(manifest, flattened, prefix="")
    assert inflated == OBJ


def test_keys_of_mixed_types() -> None:
    """
    A dict with keys of mixed types can be flattened as long as there's no
    collision among the string representation of the keys.
    """
    OBJ = {0: {"0": "foo", 1: "bar"}, "1": "baz"}
    EXPECTED_MANIFEST = {
        "": DictEntry(keys=[0, "1"]),
        "/0": DictEntry(keys=["0", 1]),
    }
    EXPECTED_FLATTENED = {"/0/0": "foo", "/0/1": "bar", "/1": "baz"}

    manifest, flattened = flatten(obj=OBJ, prefix="")
    assert manifest == EXPECTED_MANIFEST
    assert flattened == EXPECTED_FLATTENED

    inflated = inflate(manifest, flattened, prefix="")
    assert inflated == OBJ
