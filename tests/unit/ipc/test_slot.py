"""A slot grows to its writer's high-water mark, and a reader copies out what it is told."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.ipc import SharedSlot, SlotReader, new_prefix, unlink_blocks


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
