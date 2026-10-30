"""A slot grows to its writer's high-water mark or stops at its limit, and a reader copies out."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.ipc import (
    Block,
    SharedSlot,
    SharedSlotAllocationFailed,
    SharedSlotLimitExceeded,
    SlotReader,
    new_prefix,
    unlink_blocks,
)
from reactor_runtime.ipc import slot as slot_module


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


def test_a_write_returns_its_span_and_a_reader_copies_it_out(
    slot: SharedSlot, reader: SlotReader
) -> None:
    slot.begin()
    first = slot.write(b"abc")
    second = slot.write(np.arange(4, dtype=np.uint8))
    assert (first, second) == ((0, 3), (3, 4))

    reader.attach(slot.name)
    assert reader.copy(*first) == bytearray(b"abc")
    assert reader.copy(*second) == bytearray(bytes(range(4)))


def test_the_block_grows_keeping_what_was_written_and_the_old_block_goes(
    slot: SharedSlot, reader: SlotReader, tmp_path: Path
) -> None:
    small = slot.name
    slot.begin()
    head = slot.write(np.full(3000, 7, dtype=np.uint8))
    tail = slot.write(np.full(3000, 9, dtype=np.uint8))

    assert slot.name != small
    assert slot.capacity >= 6000
    assert not (tmp_path / small).exists()
    reader.attach(slot.name)
    assert reader.copy(*head) == bytearray(b"\x07" * 3000)
    assert reader.copy(*tail) == bytearray(b"\x09" * 3000)


def test_a_growth_that_cannot_double_takes_exactly_what_the_call_needs(
    slot: SharedSlot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real = slot_module.create_block

    def tight(name: str, size: int, directory: Path | None = None) -> Block:
        if size > 6000:
            raise SharedSlotAllocationFailed(f"no room for {size} bytes")
        return real(name, size, directory)

    monkeypatch.setattr(slot_module, "create_block", tight)
    slot.begin()
    slot.write(b"x" * 3000)
    with caplog.at_level(logging.WARNING, logger=slot_module.__name__):
        slot.write(b"y" * 3000)  # 6000 fits; 8192, the doubled block, does not

    assert slot.capacity == 6000
    reports = [r for r in caplog.records if "cannot double" in r.getMessage()]
    assert len(reports) == 1


def test_a_growth_that_does_not_fit_at_all_leaves_the_block_usable(
    slot: SharedSlot, reader: SlotReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(name: str, size: int, directory: Path | None = None) -> Block:
        raise SharedSlotAllocationFailed(f"no room for {size} bytes")

    monkeypatch.setattr(slot_module, "create_block", full)
    slot.begin()
    with pytest.raises(SharedSlotAllocationFailed):
        slot.write(b"x" * 5000)

    slot.begin()
    span = slot.write(b"small")
    reader.attach(slot.name)
    assert reader.copy(*span) == bytearray(b"small")


def test_a_bounded_slot_grows_only_to_its_limit(tmp_path: Path) -> None:
    slot = SharedSlot(initial_bytes=4096, max_bytes=6000, directory=tmp_path)
    try:
        slot.begin()
        slot.write(b"x" * 5000)
        assert slot.capacity == 6000
    finally:
        slot.close()


def test_a_slot_at_its_limit_never_grows_and_refuses_a_larger_call_before_writing(
    tmp_path: Path, reader: SlotReader
) -> None:
    slot = SharedSlot(initial_bytes=4096, max_bytes=4096, directory=tmp_path)
    try:
        block = slot.name
        slot.begin()
        slot.write(b"x" * 4000)
        with pytest.raises(SharedSlotLimitExceeded) as refused:
            slot.write(b"y" * 100)
        assert (refused.value.needed, refused.value.limit) == (4100, 4096)

        assert (slot.name, slot.capacity) == (block, 4096)
        assert sorted(path.name for path in tmp_path.iterdir()) == [block]
    finally:
        slot.close()


def test_the_limit_error_crosses_a_process_boundary_intact() -> None:
    error = pickle.loads(pickle.dumps(SharedSlotLimitExceeded(10, 4)))
    assert (error.needed, error.limit, str(error)) == (
        10,
        4,
        "a call holds 10 bytes, above the slot's limit of 4",
    )


def test_a_first_block_above_the_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="above max_bytes"):
        SharedSlot(initial_bytes=8192, max_bytes=4096, directory=tmp_path)


def test_a_reader_refuses_a_span_past_the_block(slot: SharedSlot, reader: SlotReader) -> None:
    reader.attach(slot.name)
    with pytest.raises(ValueError, match="past the block"):
        reader.copy(slot.capacity - 1, 2)


def test_a_reader_without_a_block_refuses_to_copy(reader: SlotReader) -> None:
    with pytest.raises(RuntimeError, match="attach"):
        reader.copy(0, 1)


def test_the_blocks_a_dead_writer_left_are_reclaimed_by_its_prefix(tmp_path: Path) -> None:
    prefix = new_prefix()
    writer = SharedSlot(prefix=prefix, initial_bytes=4096, directory=tmp_path)
    writer.begin()
    writer.write(np.ones(64 * 1024, dtype=np.uint8))
    assert writer.name != f"{prefix}0"  # grown: only the prefix predicts the name
    writer._block.close()  # the writer dies: its mapping goes, its block stays

    assert unlink_blocks(prefix, tmp_path) == 1
    assert list(tmp_path.iterdir()) == []


def test_close_unlinks_the_block(tmp_path: Path) -> None:
    slot = SharedSlot(initial_bytes=64, directory=tmp_path)
    name = slot.name
    slot.close()
    slot.close()  # idempotent
    assert not (tmp_path / name).exists()
