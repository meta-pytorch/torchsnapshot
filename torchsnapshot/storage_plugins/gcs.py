#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[2]: Allow `Any` in type annotations
# pyre-ignore-all-errors[21]: Undefined import

import asyncio
import functools
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar
from urllib.parse import quote

import google.auth.exceptions  # @manual
import requests.exceptions
import urllib3.exceptions
from google.auth import default  # @manual
from google.auth.transport.requests import AuthorizedSession  # @manual
from google.resumable_media import common  # @manual
from google.resumable_media.requests import ChunkedDownload, ResumableUpload  # @manual
from torchsnapshot.io_types import ReadIO, StoragePlugin, WriteIO
from torchsnapshot.memoryview_stream import MemoryviewStream

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_CONNECTION_POOLS: int = 8
_DEFAULT_CONNECTION_POOL_SIZE: int = 128
_DEFAULT_CONNECTION_RETRIES: int = 3
_DEFAULT_IO_CONCURRENCY: int = 8
_DEFAULT_DEADLINE_SEC: int = 180
_DEFAULT_CHUNK_SIZE_BYTE: int = 100 * 1024 * 1024


T = TypeVar("T")


class GCSStoragePlugin(StoragePlugin):
    SCOPES = (
        "https://www.googleapis.com/auth/devstorage.full_control",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/devstorage.read_write",
    )
    UPLOAD_URL_TEMPLATE = (
        "https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=resumable"
    )
    DOWNLOAD_URL_TEMPLATE = (
        "https://www.googleapis.com/download/storage/v1/b/"
        "{bucket}/o/{blob_name}?alt=media"
    )

    def __init__(
        self, root: str, storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        components = root.split("/")
        if len(components) < 2:
            raise RuntimeError(
                "The GCS root path must follow the following pattern: "
                f"[BUCKET]/[PATH] (got {root})"
            )
        self.bucket_name: str = components[0]
        self.root: str = "/".join(components[1:])

        # pyre-ignore
        credentials, _ = default(scopes=self.SCOPES)
        # pyre-ignore
        self.authed_session = AuthorizedSession(credentials)
        # https://github.com/googleapis/python-storage/issues/253
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=_DEFAULT_CONNECTION_POOLS,
            pool_maxsize=_DEFAULT_CONNECTION_POOL_SIZE,
            max_retries=_DEFAULT_CONNECTION_RETRIES,
            pool_block=True,
        )
        self.authed_session.mount("https://", adapter)
        self.executor = ThreadPoolExecutor(max_workers=_DEFAULT_IO_CONCURRENCY)
        self.retry_strategy = _RetryStrategy(deadline_sec=_DEFAULT_DEADLINE_SEC)

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        if (
            # pyre-ignore
            isinstance(e, common.InvalidResponse)
            # pyre-ignore
            and e.response.status_code in common.RETRYABLE
        ):
            return True
        return isinstance(
            e,
            (
                # pyre-ignore
                google.auth.exceptions.TransportError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
                urllib3.exceptions.ProtocolError,
                ConnectionError,
            ),
        )

    @staticmethod
    def _recover_resumable_upload(
        # pyre-ignore
        upload: ResumableUpload,
        stream: io.BytesIO,
    ) -> None:
        if upload.invalid:
            upload.recover()
        # When ResumableUpload becomes invalid, its .recover() method rewinds
        # the stream. However, certain failures can cause the cursor position
        # to be different from ResumableUpload.bytes_uploaded without rendering
        # the ResumableUpload invalid. In such cases, we need to rewind the
        # stream explicitly.
        stream.seek(upload.bytes_uploaded)

    def _async_partial(
        self,
        func: Callable[[Any], T],
        *args,
        **kwargs,
    ) -> Callable[[], Awaitable[T]]:
        event_loop = asyncio.get_running_loop()
        return functools.partial(
            event_loop.run_in_executor,
            executor=self.executor,
            func=functools.partial(func, *args, **kwargs),
        )

    async def write(self, write_io: WriteIO) -> None:
        if isinstance(write_io.buf, bytes):
            stream = io.BytesIO(write_io.buf)
        elif isinstance(write_io.buf, memoryview):
            stream = MemoryviewStream(write_io.buf)
        else:
            raise TypeError(f"Unrecognized buffer type: {type(write_io.buf)}")

        # pyre-ignore
        upload = ResumableUpload(
            upload_url=self.UPLOAD_URL_TEMPLATE.format(bucket=self.bucket_name),
            chunk_size=_DEFAULT_CHUNK_SIZE_BYTE,
        )
        await self.retry_strategy.await_with_retry(
            func=self._async_partial(
                upload.initiate,
                transport=self.authed_session,
                stream=stream,
                metadata={"name": os.path.join(self.root, write_io.path)},
                content_type="application/octet-stream",
            ),
            is_transient_error=self._is_transient_error,
        )
        while not upload.finished:
            await self.retry_strategy.await_with_retry(
                func=self._async_partial(
                    upload.transmit_next_chunk,
                    transport=self.authed_session,
                ),
                is_transient_error=self._is_transient_error,
                before_retry=self._async_partial(
                    self._recover_resumable_upload,
                    upload=upload,
                    stream=stream,
                ),
            )

    async def read(self, read_io: ReadIO) -> None:
        blob_name = quote(
            os.path.join(self.root, read_io.path).encode("utf-8"), safe=b"~"
        )

        byte_range = read_io.byte_range
        if byte_range is None:
            start = 0
            end = None
        else:
            start = byte_range[0]
            end = byte_range[1] - 1  # ChunkedDownload's end argument is inclusive

        # pyre-ignore
        download = ChunkedDownload(
            media_url=self.DOWNLOAD_URL_TEMPLATE.format(
                bucket=self.bucket_name,
                blob_name=blob_name,
            ),
            chunk_size=_DEFAULT_CHUNK_SIZE_BYTE,
            stream=read_io.buf,
            start=start,
            end=end,
        )
        while not download.finished:
            await self.retry_strategy.await_with_retry(
                func=self._async_partial(
                    download.consume_next_chunk, transport=self.authed_session
                ),
                is_transient_error=self._is_transient_error,
            )
        read_io.buf.seek(0)

    async def delete(self, path: str) -> None:
        raise NotImplementedError()

    async def delete_dir(self, path: str) -> None:
        raise NotImplementedError()

    async def close(self) -> None:
        self.authed_session.close()


class _RetryStrategy:
    """
    A retry strategy that takes the collective progress of concurrent
    coroutines into consideration.

    All concurrent coroutines share the same deadline, which is refreshed when
    a new concurrent coroutine is kicked off or when a concurrent coroutine
    completes. The targeted effect is that all coroutines can retry on
    transient error as long as some concurrent coroutines are making progress.
    The rationale behind the retry strategy is that we favor progress over
    efficiency when faced with congestion.

    NOTE: this class is not thread-safe.
    """

    INITIAL_BACKOFF = 1
    BACKOFF_MULTIPLIER = 2

    def __init__(self, deadline_sec: int) -> None:
        self.deadline_sec = deadline_sec
        self.countdown_begin_ts = 0

    async def await_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        is_transient_error: Callable[[Exception], bool],
        before_retry: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> T:
        self.countdown_begin_ts = int(time.monotonic())
        backoff_sec = self.INITIAL_BACKOFF
        while True:
            try:
                ret = await func()
            except Exception as e:
                if not is_transient_error(e):
                    raise e
                if time.monotonic() >= self.countdown_begin_ts + self.deadline_sec:
                    logger.warn(
                        f"Encountered a retryable error: {e}\n"
                        "Not retrying since no concurrent requests were "
                        f"completed within {self.deadline_sec} seconds."
                    )
                    raise e
                else:
                    backoff_sec *= self.BACKOFF_MULTIPLIER
                    backoff_sec += random.randint(0, 1000) * 0.001  # jitter
                    logger.warn(
                        f"Encountered a retryable error: {e}\n"
                        f"Retrying after {backoff_sec} seconds."
                    )
                    await asyncio.sleep(backoff_sec)
                    if before_retry is not None:
                        await before_retry()
                    continue
            else:
                self.countdown_begin_ts = int(time.monotonic())
                return ret
