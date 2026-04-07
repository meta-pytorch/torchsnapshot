#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
from typing import Optional


class MemoryviewStream(io.IOBase):
    def __init__(self, mv: memoryview) -> None:
        self._mv: memoryview = mv.cast("b")
        self._pos = 0

    # pyre-fixme[24]: Generic type `memoryview` expects 1 type parameter.
    def read(self, size: Optional[int] = -1) -> memoryview:
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f"{size!r} is not an integer")
            else:
                size = size_index()
        if size < 0:
            size = len(self._mv)
        if len(self._mv) <= self._pos:
            return memoryview(b"")
        newpos = min(len(self._mv), self._pos + size)
        b = self._mv[self._pos : newpos]
        self._pos = newpos
        return b

    def read1(self, size: int = -1) -> memoryview:
        """This is the same as read."""
        return self.read(size)

    def seek(self, pos: int, whence: int = 0) -> int:
        if self.closed:
            raise ValueError("seek on closed file")
        try:
            pos_index = pos.__index__
        except AttributeError:
            raise TypeError(f"{pos!r} is not an integer")
        else:
            pos = pos_index()
        if whence == 0:
            if pos < 0:
                raise ValueError("negative seek position %r" % (pos,))
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._mv) + pos)
        else:
            raise ValueError("unsupported whence value")
        return self._pos

    def tell(self) -> int:
        if self.closed:
            raise ValueError("tell on closed file")
        return self._pos

    def readable(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True

    def writable(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return False

    def seekable(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True
