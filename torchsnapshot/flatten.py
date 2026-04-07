#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2, 3]: allow `Any` in function signatures

import itertools
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Tuple, Union
from urllib.parse import unquote

from .manifest import DictEntry, Entry, ListEntry, Manifest, OrderedDictEntry


def flatten(obj: Any, prefix: str) -> Tuple[Manifest, Dict[str, Any]]:
    """
    Recursively flatten an object in a reversible manner.

    Args:
        obj: The object to flatten.
        prefix: The prefix to prepend to the keys of returned dictionaries.

    Returns:
        - The container manifest needed for inflating the flattened object
        - The flattened object in the form of a dictionary
    ::
        >>> collection = {'foo': [1, 2, OrderedDict(bar=3, baz=4)]}]}
        >>> manifest, flattened = flatten(collection, prefix='my/prefix')
        >>> manifest
        {
            "my%2Fprefix": {"type": "dict"},
            "my%2Fprefix/foo": {"type": "list"},
            "my%2Fprefix/foo/2": {"type": "OrderedDict", "keys": ["bar", "baz"]},
        }
        >>> flattened
        {
            "my%2Fprefix/foo/0": 1,
            "my%2Fprefix/foo/1": 2,
            "my%2Fprefix/foo/2/bar": 3,
            "my%2Fprefix/foo/2/baz": 4,
        }
    """
    # "/" in the keys of the returned dictionaries is used to denote hiearchy.
    # Encode the user-provided prefix to eliminate ambiguity.
    return _flatten(obj=obj, prefix=_encode(prefix))


def _flatten(obj: Any, prefix: str) -> Tuple[Manifest, Dict[str, Any]]:
    manifest = {}
    flattened = {}
    if type(obj) == list:
        manifest[prefix] = ListEntry()
        for idx, elem in enumerate(obj):
            path = f"{prefix}/{str(idx)}"
            m, f = _flatten(elem, path)
            manifest.update(m)
            flattened.update(f)
    elif type(obj) in (dict, OrderedDict) and _should_flatten_dict(obj):
        if type(obj) == dict:
            manifest[prefix] = DictEntry(keys=list(obj.keys()))
        else:
            manifest[prefix] = OrderedDictEntry(keys=list(obj.keys()))
        for key, elem in obj.items():
            key = _encode(str(key))
            path = f"{prefix}/{key}"
            m, f = _flatten(elem, path)
            manifest.update(m)
            flattened.update(f)
    else:
        flattened[prefix] = obj
    return manifest, flattened


def inflate(
    manifest: Manifest,
    flattened: Dict[str, Any],
    prefix: str,
) -> Dict[Any, Any]:
    """
    The reverse operation of func::`flatten`.

    Args:
        manifest: The container manifest returned by func::`flatten`.
        flattened: The flattened object.
        prefix: The prefix used for func::`flatten`.

    Returns:
        The inflated object.
    """
    # Encode the user-provided prefix
    prefix = _encode(prefix)

    # Filter the relevant items in manifest and flattened
    manifest = {k: v for k, v in manifest.items() if k.split("/")[0] == prefix}
    flattened = {k: v for k, v in flattened.items() if k.split("/")[0] == prefix}

    # When flatten() receives a non-flattenable object it returns:
    # ({}, {"[ENCODED_PREFIX]": obj})
    if prefix in flattened:
        return flattened[prefix]

    if prefix not in manifest:
        raise AssertionError(
            f"{prefix} is absent in both manifest and flattened.\n"
            f"manifest: {manifest}\n"
            f"flattened: {flattened}"
        )

    # Instantiate all containers according to the container manifest
    containers = {}
    for path, entry in manifest.items():
        containers[path] = _entry_to_container(entry)

    # Group containers/values by the parent container
    container_path_to_vals = defaultdict(dict)
    for path, obj in itertools.chain(containers.items(), flattened.items()):
        # Skip the outermost container
        if path == prefix:
            continue
        tokens = path.split("/")
        if len(tokens) < 2:
            # Impossible. Just to be defensive.
            raise AssertionError(f"Invalid path: {path}")
        key = tokens.pop()
        container_path = "/".join(tokens)
        container_path_to_vals[container_path][key] = obj

    # Populate values within all containers
    for path, values in container_path_to_vals.items():
        if not isinstance(containers[path], (list, dict)):
            raise AssertionError(
                f"inflate() does not know how to inflate container of type {type(containers[path])} "
                f"(path: {path}, container entry: {manifest.get(path)})."
            )
        _populate_container(path=path, container=containers[path], values=values)
    return containers[prefix]


def _should_flatten_dict(d: Dict[Any, Any]) -> bool:
    """
    Determine if a dict should be flattened.

    A dict shouldn't be flattened if:
    - There is a collision among the string representations of the keys.
    - The keys contain objects other than string or int.
    """
    if not all(isinstance(k, (str, int)) for k in d.keys()):
        return False
    if len({str(k) for k in d.keys()}) < len(d):
        return False
    return True


def _entry_to_container(entry: Entry) -> Any:
    """
    Initialize a container from a container entry.

    For dictionary containers, the items are populated with `None` on
    initialization to ensure the original item order.
    """
    if isinstance(entry, ListEntry):
        return []
    elif isinstance(entry, DictEntry):
        return dict.fromkeys(entry.keys)
    elif isinstance(entry, OrderedDictEntry):
        return OrderedDict.fromkeys(entry.keys)
    else:
        raise RuntimeError(
            f"Unrecognized container entry type: {type(entry)} ({entry.type})."
        )


def _populate_container(path: str, container: Any, values: Dict[str, Any]) -> None:
    if isinstance(container, list):
        items = sorted(values.items(), key=lambda e: int(e[0]))
        container.extend(item[1] for item in items)
    elif isinstance(container, dict):
        key_to_val: Dict[Union[str, int], Any] = {
            _decode(k): v for k, v in values.items()
        }
        # If a string can represent an integer, make the integer represented by
        # the string a candidate key in addition.
        for key in list(values.keys()):
            key = _decode(key)
            if _check_int(key):
                key_to_val[int(key)] = values[key]
        # NOTE: only keys that appear in both `container` and `key_to_val` will
        # be present in the poplated container. The caller of `inflate()` is
        # responsible for adding a key into the container entry if they wish
        # the key to be present in the inflated container.
        for key in list(container.keys()):
            if key in key_to_val:
                container[key] = key_to_val[key]
            else:
                del container[key]
    else:
        raise AssertionError(f"Unrecognized container type: {type(container)}.")


def _check_int(s: str) -> bool:
    if s.isdigit():
        return True
    elif len(s) > 1 and s[0] in ("-", "+"):
        return s[1:].isdigit()
    else:
        return False


def _encode(s: str) -> str:
    """
    Implements a subset of https://datatracker.ietf.org/doc/html/rfc3986.html
    for escaping "/" in user-provided strings.
    """
    s = s.replace("%", "%25")
    s = s.replace("/", "%2F")
    return s


def _decode(s: str) -> str:
    return unquote(s)
