# pyre-unsafe

import asyncio
import functools
import os
import sys
import threading
from contextlib import contextmanager
from heapq import heappop


# copy-pasted from nest-asyncio, but modified to avoid patching the global
# namespace and instead only patching the instance variable
def _patch_loop(loop: asyncio.AbstractEventLoop) -> None:
    def run_forever(self):
        with manage_run(self), manage_asyncgens(self):
            while True:
                self._run_once()
                if self._stopping:
                    break
        self._stopping = False

    def run_until_complete(self, future):
        with manage_run(self):
            f = asyncio.ensure_future(future, loop=self)
            if f is not future:
                f._log_destroy_pending = False
            while not f.done():
                self._run_once()
                if self._stopping:
                    break
            if not f.done():
                raise RuntimeError("Event loop stopped before Future completed.")
            return f.result()

    def _run_once(self):
        """
        Simplified re-implementation of asyncio's _run_once that
        runs handles as they become ready.
        """
        now = self.time()
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)

        timeout = (
            0
            if ready or self._stopping
            else min(max(scheduled[0]._when - now, 0), 86400)
            if scheduled
            else None
        )
        event_list = self._selector.select(timeout)
        self._process_events(event_list)

        end_time = self.time() + self._clock_resolution
        while scheduled and scheduled[0]._when < end_time:
            handle = heappop(scheduled)
            ready.append(handle)

        for _ in range(len(ready)):
            if not ready:
                break
            handle = ready.popleft()
            if not handle._cancelled:
                handle._run()
        handle = None

    @contextmanager
    def manage_run(self):
        """Set up the loop for running."""
        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = asyncio.events._get_running_loop()
        try:
            self._thread_id = threading.get_ident()
            asyncio.events._set_running_loop(self)
            self._num_runs_pending += 1
            if self._is_proactorloop:
                if self._self_reading_future is None:
                    self.call_soon(self._loop_self_reading)
            yield
        finally:
            self._thread_id = old_thread_id
            asyncio.events._set_running_loop(old_running_loop)
            self._num_runs_pending -= 1
            if self._is_proactorloop:
                if (
                    self._num_runs_pending == 0
                    and self._self_reading_future is not None
                ):
                    ov = self._self_reading_future._ov
                    self._self_reading_future.cancel()
                    if ov is not None:
                        self._proactor._unregister(ov)
                    self._self_reading_future = None

    @contextmanager
    def manage_asyncgens(self):
        old_agen_hooks = sys.get_asyncgen_hooks()
        try:
            self._set_coroutine_origin_tracking(self._debug)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(
                    firstiter=self._asyncgen_firstiter_hook,
                    finalizer=self._asyncgen_finalizer_hook,
                )
            yield
        finally:
            self._set_coroutine_origin_tracking(False)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(*old_agen_hooks)

    def _check_running(self):
        """Do not throw exception if loop is already running."""
        pass

    #  `partial[typing.Any]`.
    loop.run_forever = functools.partial(run_forever, loop)
    #  Union[Awaitable[Variable[_T]], Generator[typing.Any, None, Variable[_T]]]) ->
    #  _T`; used as `partial[typing.Any]`.
    loop.run_until_complete = functools.partial(run_until_complete, loop)
    # pyre-fixme[16]: `AbstractEventLoop` has no attribute `_run_once`.
    loop._run_once = functools.partial(_run_once, loop)
    # pyre-fixme[16]: `AbstractEventLoop` has no attribute `_check_running`.
    loop._check_running = functools.partial(_check_running, loop)
    # pyre-fixme[16]: `AbstractEventLoop` has no attribute `_nest_patched`.
    loop._nest_patched = True
    # pyre-fixme[16]: `AbstractEventLoop` has no attribute `_num_runs_pending`.
    loop._num_runs_pending = 0
    # pyre-fixme[16]: `AbstractEventLoop` has no attribute `_is_proactorloop`.
    loop._is_proactorloop = os.name == "nt" and isinstance(
        loop,
        # pyre-fixme[16]: Module `asyncio` has no attribute `ProactorEventLoop`.
        asyncio.ProactorEventLoop,
    )


# TODO: this is *not* an amazing w
def maybe_nested_loop() -> asyncio.AbstractEventLoop:
    try:
        original = asyncio.get_running_loop()
    except RuntimeError:
        original = None

    loop = asyncio.new_event_loop()
    if original is None:
        return loop
    else:
        # Need to monkey-patch the loop so it can be re-entrant, which makes things
        # work on old versions of Jupyter
        #
        # It would be better if we could refactor the code to rely more on
        # asyncio.run instead of passing the event loop into places, but oh well...
        _patch_loop(loop)
        return loop
