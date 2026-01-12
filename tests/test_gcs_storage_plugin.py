#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import io
import logging
import os
import random
import tempfile
import uuid
from typing import Generator

import pytest
import torch
import torchsnapshot
from torchsnapshot.io_types import ReadIO, WriteIO

logger: logging.Logger = logging.getLogger(__name__)

_TEST_BUCKET = "torchsnapshot-benchmark"
_TENSOR_SZ = int(1_000_000 / 4)


@pytest.fixture
def gcs_health_check() -> None:
    """
    GCS access can be flaky on Github Action. Only run the tests if the health
    check passes.
    """
    try:
        from google.cloud import storage  # @manual  # pyre-ignore

        bucket = storage.Client().bucket(_TEST_BUCKET)  # pyre-ignore
        blob = bucket.blob(str(uuid.uuid4()))
        with blob.open("w") as f:
            f.write("hello")
        with blob.open("r") as f:
            f.read()

    except Exception as e:
        # pyre-ignore[29]
        pytest.skip(f"Skipping the test because gcs health check failed: {e}")


@pytest.fixture
def gcs_test_credential() -> Generator[None, None, None]:
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        yield
        return

    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            f.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            yield
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


@pytest.mark.gcs_integration_test
@pytest.mark.skipif(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, reason="")
@pytest.mark.usefixtures("gcs_test_credential", "gcs_health_check")
def test_gcs_read_write_via_snapshot() -> None:
    path = f"gs://{_TEST_BUCKET}/{uuid.uuid4()}"
    logger.info(path)

    tensor = torch.rand((_TENSOR_SZ,))
    app_state = {"state": torchsnapshot.StateDict(tensor=tensor)}
    snapshot = torchsnapshot.Snapshot.take(path=path, app_state=app_state)

    app_state["state"]["tensor"] = torch.rand((_TENSOR_SZ,))
    assert not torch.allclose(tensor, app_state["state"]["tensor"])

    snapshot.restore(app_state)
    assert torch.allclose(tensor, app_state["state"]["tensor"])


@pytest.mark.gcs_integration_test
@pytest.mark.skipif(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, reason="")
@pytest.mark.usefixtures("gcs_test_credential", "gcs_health_check")
@pytest.mark.asyncio
async def test_gcs_write_read_delete() -> None:
    path = f"{_TEST_BUCKET}/{uuid.uuid4()}"
    logger.info(path)

    from torchsnapshot.storage_plugins.gcs import GCSStoragePlugin

    plugin = GCSStoragePlugin(root=path)

    tensor = torch.rand((_TENSOR_SZ,))
    buf = io.BytesIO()
    torch.save(tensor, buf)
    write_io = WriteIO(path="tensor", buf=memoryview(buf.getvalue()))
    await plugin.write(write_io=write_io)

    read_io = ReadIO(path="tensor")
    await plugin.read(read_io=read_io)
    loaded = torch.load(read_io.buf)
    assert torch.allclose(tensor, loaded)

    # TODO: bring this back
    # await plugin.delete(path="tensor")
    await plugin.close()


@pytest.mark.gcs_integration_test
@pytest.mark.skipif(os.environ.get("TORCHSNAPSHOT_ENABLE_GCP_TEST") is None, reason="")
@pytest.mark.usefixtures("gcs_test_credential", "gcs_health_check")
@pytest.mark.asyncio
async def test_gcs_ranged_read() -> None:
    path = f"{_TEST_BUCKET}/{uuid.uuid4()}"
    logger.info(path)

    from torchsnapshot.storage_plugins.gcs import GCSStoragePlugin

    plugin = GCSStoragePlugin(root=path)

    buf = bytes(random.getrandbits(8) for _ in range(2000))
    write_io = WriteIO(path="rand_bytes", buf=memoryview(buf))

    await plugin.write(write_io=write_io)

    read_io = ReadIO(path="rand_bytes", byte_range=(100, 200))
    await plugin.read(read_io=read_io)
    assert len(read_io.buf.getvalue()) == 100
    assert read_io.buf.getvalue() == buf[100:200]

    await plugin.close()
