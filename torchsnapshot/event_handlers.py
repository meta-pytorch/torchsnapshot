#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import lru_cache
from typing import List

from importlib_metadata import entry_points
from typing_extensions import Protocol, runtime_checkable

from .event import Event

logger: logging.Logger = logging.getLogger(__name__)


@runtime_checkable
class EventHandler(Protocol):
    def handle_event(self, event: Event) -> None: ...


_log_handlers: List[EventHandler] = []


@lru_cache(maxsize=None)
def get_event_handlers() -> List[EventHandler]:
    global _log_handlers

    # Registered event handlers through entry points
    eps = entry_points(group="event_handlers")
    for entry in eps:
        logger.debug(
            f"Attempting to register event handler {entry.name}: {entry.value}"
        )
        factory = entry.load()
        handler = factory()

        if not isinstance(handler, EventHandler):
            raise RuntimeError(
                f"The factory function for { ({entry.value}) } "
                "did not return a EventHandler object."
            )
        _log_handlers.append(handler)
    return _log_handlers


def log_event(event: Event) -> None:
    """
    Handle an event.
    Args:
        event: The event to handle.
    """
    for handler in get_event_handlers():
        handler.handle_event(event)
