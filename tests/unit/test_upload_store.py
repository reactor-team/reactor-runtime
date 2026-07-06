import asyncio

import pytest

from reactor_runtime.core import UploadedFile
from reactor_runtime.upload_store import (
    UnknownUploadError,
    UploadAlreadyCompleteError,
    UploadIdTakenError,
    UploadSizeMismatchError,
    UploadStore,
)


def test_create_slot_returns_a_unique_id() -> None:
    store = UploadStore()
    ids = {store.create_slot("a.png", "image/png", 3) for _ in range(5)}
    assert len(ids) == 5


async def test_create_slot_honours_a_supplied_id() -> None:
    store = UploadStore()
    returned = store.create_slot("cat.png", "image/png", 4, upload_id="platform-123")
    store.put("platform-123", b"\x89PNG")

    file = await store.fetch("platform-123")

    assert returned == "platform-123"
    assert file.data == b"\x89PNG"


def test_create_slot_rejects_a_reserved_id() -> None:
    store = UploadStore()
    store.create_slot("a.bin", "application/octet-stream", 2, upload_id="dup")
    with pytest.raises(UploadIdTakenError):
        store.create_slot("b.bin", "application/octet-stream", 2, upload_id="dup")


async def test_put_then_fetch_returns_the_model_facing_view() -> None:
    store = UploadStore()
    upload_id = store.create_slot("cat.png", "image/png", 4)
    store.put(upload_id, b"\x89PNG")

    file = await store.fetch(upload_id)

    assert file == UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG")
    assert not hasattr(file, "upload_id")


async def test_fetch_is_repeatable_within_a_session() -> None:
    store = UploadStore()
    upload_id = store.create_slot("a.bin", "application/octet-stream", 2)
    store.put(upload_id, b"hi")

    first = await store.fetch(upload_id)
    second = await store.fetch(upload_id)

    assert first.data == second.data == b"hi"


def test_put_to_an_unknown_slot_is_rejected() -> None:
    store = UploadStore()
    with pytest.raises(UnknownUploadError):
        store.put("nope", b"data")


def test_put_twice_is_rejected() -> None:
    store = UploadStore()
    upload_id = store.create_slot("a.bin", "application/octet-stream", 2)
    store.put(upload_id, b"hi")
    with pytest.raises(UploadAlreadyCompleteError):
        store.put(upload_id, b"hi")


def test_put_with_a_size_mismatch_is_rejected() -> None:
    store = UploadStore()
    upload_id = store.create_slot("a.bin", "application/octet-stream", 4)
    with pytest.raises(UploadSizeMismatchError):
        store.put(upload_id, b"hi")


async def test_fetch_before_put_is_unknown() -> None:
    store = UploadStore()
    upload_id = store.create_slot("a.bin", "application/octet-stream", 2)
    with pytest.raises(UnknownUploadError):
        await store.fetch(upload_id)


async def test_fetch_waits_for_a_late_put() -> None:
    store = UploadStore()
    upload_id = store.create_slot("cat.png", "image/png", 4)

    async def deliver() -> None:
        await asyncio.sleep(0.01)
        store.put(upload_id, b"\x89PNG")

    task = asyncio.create_task(deliver())
    file = await store.fetch(upload_id, wait_seconds=1.0)
    await task

    assert file.data == b"\x89PNG"


async def test_fetch_waits_for_a_slot_reserved_after_the_reference() -> None:
    # The reference can arrive before babysitter has even reserved the slot; the
    # wait is keyed by id, so a create_slot + put that lands during the wait still
    # resolves it.
    store = UploadStore()

    async def deliver() -> None:
        await asyncio.sleep(0.01)
        store.create_slot("cat.png", "image/png", 4, upload_id="late")
        store.put("late", b"\x89PNG")

    task = asyncio.create_task(deliver())
    file = await store.fetch("late", wait_seconds=1.0)
    await task

    assert file.data == b"\x89PNG"


async def test_fetch_times_out_when_bytes_never_arrive() -> None:
    store = UploadStore()
    with pytest.raises(UnknownUploadError):
        await store.fetch("missing", wait_seconds=0.02)


async def test_clear_drops_every_slot() -> None:
    store = UploadStore()
    upload_id = store.create_slot("a.bin", "application/octet-stream", 2)
    store.put(upload_id, b"hi")

    store.clear()

    with pytest.raises(UnknownUploadError):
        await store.fetch(upload_id)


def test_expected_size_returns_the_declared_size() -> None:
    store = UploadStore()
    upload_id = store.create_slot("cat.png", "image/png", 4)

    assert store.expected_size(upload_id) == 4


def test_expected_size_rejects_an_unknown_slot() -> None:
    store = UploadStore()
    with pytest.raises(UnknownUploadError):
        store.expected_size("nope")


def test_expected_size_rejects_an_already_completed_slot() -> None:
    store = UploadStore()
    upload_id = store.create_slot("cat.png", "image/png", 4)
    store.put(upload_id, b"\x89PNG")

    with pytest.raises(UploadAlreadyCompleteError):
        store.expected_size(upload_id)
