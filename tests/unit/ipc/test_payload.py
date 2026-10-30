"""An object crosses through shared memory intact, and its arrays never touch the pickle."""

from __future__ import annotations

import enum
import logging
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.ipc import SharedSlot, SharedSlotLimitExceeded, SlotReader, pack, unpack
from reactor_runtime.ipc import payload as payload_module


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
def slot(tmp_path: Path) -> Iterator[SharedSlot]:
    slot = SharedSlot(initial_bytes=4096, directory=tmp_path)
    yield slot
    slot.close()


@pytest.fixture
def reader(tmp_path: Path) -> Iterator[SlotReader]:
    reader = SlotReader(directory=tmp_path)
    yield reader
    reader.close()


def _result(frames: np.ndarray | None = None) -> Result:
    return Result(
        frames=np.arange(4 * 6 * 8 * 3, dtype=np.uint8).reshape(4, 6, 8, 3)
        if frames is None
        else frames,
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
    assert (back.tags, back.meta, back.mode, back.raw) == (
        ["a", "b"],
        {"seed_id": 3},
        Mode.FAST,
        b"\x00\x01",
    )
    assert back.nothing is None
    np.testing.assert_array_equal(back.frames, result.frames)
    np.testing.assert_array_equal(back.depth, result.depth)


def test_the_reader_owns_its_copy(slot: SharedSlot, reader: SlotReader) -> None:
    first = unpack(pack(_result(), slot), reader)
    # A second call reuses the block from offset zero. The first result must
    # not change under the caller's feet.
    snapshot = first.frames.copy()
    unpack(pack(_result(np.zeros_like(snapshot)), slot), reader)
    np.testing.assert_array_equal(first.frames, snapshot)
    first.frames[0, 0, 0, 0] = 255  # writable, not a read-only view


def test_the_header_names_the_grown_block(slot: SharedSlot, reader: SlotReader) -> None:
    small = slot.name
    big = np.ones(64 * 1024, dtype=np.uint8)
    header = pack(big, slot)
    _, name, spans = pickle.loads(header)

    assert name != small
    assert spans == [(0, big.nbytes)]
    np.testing.assert_array_equal(unpack(header, reader), big)


def test_a_call_above_the_limit_is_refused_and_the_slot_goes_on(
    tmp_path: Path, reader: SlotReader
) -> None:
    slot = SharedSlot(initial_bytes=4096, max_bytes=4096, directory=tmp_path)
    try:
        arrays = (np.ones(3000, dtype=np.uint8), np.ones(3000, dtype=np.uint8))
        with pytest.raises(SharedSlotLimitExceeded) as refused:
            pack(arrays, slot)
        assert (refused.value.needed, refused.value.limit) == (6000, 4096)

        small = np.arange(16, dtype=np.uint8)
        np.testing.assert_array_equal(unpack(pack(small, slot), reader), small)
    finally:
        slot.close()


def test_with_no_slot_every_array_travels_inline_in_the_same_header(tmp_path: Path) -> None:
    frames = np.arange(4096, dtype=np.uint8)
    header = pack({"frames": frames}, None)
    _, name, spans = pickle.loads(header)

    assert (name, spans) == (None, [])
    assert len(header) > frames.nbytes
    back = unpack(header, SlotReader(directory=tmp_path / "unused"))
    np.testing.assert_array_equal(back["frames"], frames)


def test_an_object_without_buffers_needs_no_block(slot: SharedSlot, tmp_path: Path) -> None:
    header = pack({"step": 3}, slot)
    assert unpack(header, SlotReader(directory=tmp_path / "unused")) == {"step": 3}


def test_a_non_contiguous_array_pickles_inline_and_is_reported_once(
    slot: SharedSlot,
    reader: SlotReader,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(payload_module, "_inline_reported", False)
    view = np.zeros((2000, 800), dtype=np.float64)[:, ::2]  # 6.4 MB, not contiguous

    with caplog.at_level(logging.WARNING, logger=payload_module.__name__):
        header = pack(view, slot)
        pack(view, slot)
    _, _, spans = pickle.loads(header)

    assert spans == []
    np.testing.assert_array_equal(unpack(header, reader), view)
    reports = [r for r in caplog.records if "pickled inline" in r.getMessage()]
    assert len(reports) == 1
