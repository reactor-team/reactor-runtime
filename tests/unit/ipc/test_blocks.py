"""A block is a file every process that maps it shares, and only its creator unlinks."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reactor_runtime.ipc import (
    SharedSlotAllocationFailed,
    attach_block,
    create_block,
    default_directory,
    unlink_blocks,
)
from reactor_runtime.ipc import blocks as blocks_module


def test_two_mappings_of_one_block_share_its_bytes(tmp_path: Path) -> None:
    created = create_block("b0", 64, tmp_path)
    attached = attach_block("b0", tmp_path)
    try:
        created.buf[:5] = b"hello"
        assert bytes(attached.buf[:5]) == b"hello"
        assert attached.size == created.size == 64
    finally:
        attached.close()
        created.close()
        created.unlink()


def test_a_new_block_reads_as_zeros_and_has_its_pages_reserved(tmp_path: Path) -> None:
    block = create_block("b0", 4096, tmp_path)
    try:
        assert bytes(block.buf) == bytes(4096)
        assert os.stat(tmp_path / "b0").st_size == 4096
    finally:
        block.close()
        block.unlink()


def test_a_block_name_is_a_plain_file_name(tmp_path: Path) -> None:
    for name in ("", ".", "..", "a/b", "a\0b"):
        with pytest.raises(ValueError, match="plain block name"):
            create_block(name, 8, tmp_path)
        with pytest.raises(ValueError, match="plain block name"):
            attach_block(name, tmp_path)


def test_an_existing_block_is_not_created_again(tmp_path: Path) -> None:
    block = create_block("b0", 8, tmp_path)
    try:
        with pytest.raises(FileExistsError):
            create_block("b0", 8, tmp_path)
    finally:
        block.close()
        block.unlink()


def test_only_a_regular_file_is_mapped(tmp_path: Path) -> None:
    target = create_block("real", 8, tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    os.mkfifo(tmp_path / "fifo")
    try:
        with pytest.raises(OSError, match=rf"\[Errno {errno.ELOOP}\]"):
            attach_block("link", tmp_path)
        with pytest.raises(ValueError, match="regular file"):
            attach_block("fifo", tmp_path)
    finally:
        target.close()
        target.unlink()


def test_a_full_directory_fails_at_creation_and_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_room(fd: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(blocks_module.os, "posix_fallocate", no_room, raising=False)
    with pytest.raises(SharedSlotAllocationFailed, match="--shm-size"):
        create_block("b0", 1 << 20, tmp_path)
    assert not (tmp_path / "b0").exists()


def test_unlink_blocks_removes_only_the_prefix(tmp_path: Path) -> None:
    mine = [create_block(f"rrabc_{n}", 8, tmp_path) for n in range(3)]
    other = create_block("rrxyz_0", 8, tmp_path)
    for block in [*mine, other]:
        block.close()

    assert unlink_blocks("rrabc_", tmp_path) == 3

    assert sorted(p.name for p in tmp_path.iterdir()) == ["rrxyz_0"]


def test_a_process_that_attaches_and_exits_leaves_the_block(tmp_path: Path) -> None:
    block = create_block("b0", 8, tmp_path)
    child = (
        "import sys; from pathlib import Path\n"
        "from reactor_runtime.ipc import attach_block\n"
        "b = attach_block('b0', Path(sys.argv[1])); b.buf[0] = 7; b.close()\n"
    )
    try:
        subprocess.run([sys.executable, "-c", child, str(tmp_path)], check=True, timeout=60)
        assert (tmp_path / "b0").exists()
        assert block.buf[0] == 7
    finally:
        block.close()
        block.unlink()


def test_the_default_directory_is_dev_shm_where_the_host_has_it() -> None:
    expected = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    if expected is not None:
        assert default_directory() == expected
    else:
        assert default_directory().is_dir()


def test_a_closed_block_refuses_its_buffer(tmp_path: Path) -> None:
    block = create_block("b0", 8, tmp_path)
    block.close()
    block.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        _ = block.buf
    block.unlink()
    block.unlink()  # idempotent
