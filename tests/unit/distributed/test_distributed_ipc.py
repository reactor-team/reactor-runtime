"""An object crosses through shared memory intact, and its arrays never touch the pickle."""

from __future__ import annotations

import enum
import logging
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
import pytest

from reactor_runtime.distributed import SharedSlotAllocationFailed, ipc
from reactor_runtime.distributed.ipc import SharedSlot, SlotReader, pack, unlink_blocks, unpack


class Mode(enum.Enum):
    FAST = enum.auto()


@dataclass(frozen=True)
class Inner:
    index: int
    label: str


@dataclass(frozen=True)
class Result:
    frames: np.ndarray
    depth: np.ndarray
    inner: Inner
    tags: list[str]
    meta: dict[str, int]
    mode: Mode
    raw: bytes
    nothing: None


@pytest.fixture
def slot() -> Iterator[SharedSlot]:
    slot = SharedSlot(initial_bytes=4096)
    yield slot
    slot.close()


@pytest.fixture
def reader() -> Iterator[SlotReader]:
    reader = SlotReader()
    yield reader
    reader.close()


def _result() -> Result:
    return Result(
        frames=np.arange(4 * 6 * 8 * 3, dtype=np.uint8).reshape(4, 6, 8, 3),
        depth=np.linspace(0.0, 1.0, 4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8),
        inner=Inner(index=7, label="chunk"),
        tags=["a", "b"],
        meta={"seed_id": 3},
        mode=Mode.FAST,
        raw=b"\x00\x01",
        nothing=None,
    )


def test_a_struct_crosses_intact_with_its_arrays_out_of_band(
    slot: SharedSlot, reader: SlotReader
) -> None:
    result = _result()
    header = pack(result, slot)
    blob, name, spans = pickle.loads(header)

    # Two arrays, two spans; the pickle carries neither array's bytes.
    assert name == slot.name
    assert len(spans) == 2
    assert len(blob) < result.frames.nbytes

    back = unpack(header, reader)
    assert back.inner == result.inner
    assert back.tags == result.tags
    assert back.meta == result.meta
    assert back.mode is Mode.FAST
    assert back.raw == result.raw
    assert back.nothing is None
    np.testing.assert_array_equal(back.frames, result.frames)
    np.testing.assert_array_equal(back.depth, result.depth)


def test_the_reader_owns_its_copy(slot: SharedSlot, reader: SlotReader) -> None:
    first = unpack(pack(_result(), slot), reader)
    # A second call reuses the block from offset zero. The first result must
    # not change under the caller's feet.
    snapshot = first.frames.copy()
    unpack(pack(Result(**{**_result().__dict__, "frames": np.zeros_like(snapshot)}), slot), reader)
    np.testing.assert_array_equal(first.frames, snapshot)
    first.frames[0, 0, 0, 0] = 255  # writable, not a read-only view


def test_the_block_grows_and_the_header_names_the_new_block(
    slot: SharedSlot, reader: SlotReader
) -> None:
    small_name = slot.name
    big = np.ones(64 * 1024, dtype=np.uint8)
    header = pack(big, slot)
    _, name, spans = pickle.loads(header)

    assert name != small_name
    assert slot.capacity >= big.nbytes
    assert spans == [(0, big.nbytes)]
    np.testing.assert_array_equal(unpack(header, reader), big)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=small_name)


def test_the_blocks_a_dead_writer_left_are_reclaimed_by_its_prefix() -> None:
    prefix = ipc.new_prefix()
    writer = SharedSlot(prefix=prefix, initial_bytes=4096)
    pack(np.ones(64 * 1024, dtype=np.uint8), writer)
    left = writer.name
    assert left != f"{prefix}0"  # grown, so the name is one only the prefix can predict
    writer._shm.close()  # the writer dies: its mapping goes, its block stays

    unlink_blocks(prefix)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=left)


def test_growth_keeps_buffers_already_written(slot: SharedSlot, reader: SlotReader) -> None:
    first = np.full(3000, 7, dtype=np.uint8)
    second = np.full(3000, 9, dtype=np.uint8)
    back = unpack(pack((first, second), slot), reader)
    np.testing.assert_array_equal(back[0], first)
    np.testing.assert_array_equal(back[1], second)


def test_a_non_contiguous_array_pickles_inline_and_is_reported_once(
    slot: SharedSlot,
    reader: SlotReader,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ipc, "_inline_reported", False)
    view = np.zeros((2000, 800), dtype=np.float64)[:, ::2]  # 6.4 MB, not contiguous

    with caplog.at_level(logging.WARNING, logger=ipc.__name__):
        header = pack(view, slot)
        pack(view, slot)
    _, _, spans = pickle.loads(header)

    assert spans == []
    np.testing.assert_array_equal(unpack(header, reader), view)
    reports = [r for r in caplog.records if "pickled inline" in r.getMessage()]
    assert len(reports) == 1


def test_allocation_failure_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**kwargs: object) -> shared_memory.SharedMemory:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", refuse)
    with pytest.raises(SharedSlotAllocationFailed, match="--shm-size"):
        SharedSlot(initial_bytes=1024)


def test_reader_without_a_block_refuses_to_copy(reader: SlotReader) -> None:
    with pytest.raises(RuntimeError, match="attach"):
        reader.copy(0, 1)
