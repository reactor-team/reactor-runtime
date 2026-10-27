"""A worker that keeps each session's blocks in that session's own directory."""

from __future__ import annotations

import secrets
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed import SharedSlotAllocationFailed
from reactor_runtime.model_server import ModelClient, ModelServer, ServerRequest
from reactor_runtime.model_server.server import WorkerServer
from reactor_runtime.model_server.shm import FileBlock, RequestRefused, ViewReader, unlink_prefix
from reactor_runtime.model_server.transport import UnixShmTransport

_SHM = Path("/dev/shm")


class Doubler(ModelServer):
    max_batch = 2

    def load(self, config_path: Path | None) -> None:
        pass

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        return [{"x": np.asarray(r.payload) * 2} for r in requests]


def test_a_slot_template_needs_a_slot_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{slot\}"):
        UnixShmTransport("/run/reactor-shm")


def test_each_slot_has_its_own_directory() -> None:
    transport = UnixShmTransport("/run/reactor-shm/s{slot}")
    assert transport.slot_directory(0) == Path("/run/reactor-shm/s0")
    assert transport.slot_directory(12) == Path("/run/reactor-shm/s12")
    assert UnixShmTransport().slot_directory(3) is None


def test_the_blocks_of_a_slot_are_made_in_its_directory_and_freed_on_close(
    tmp_path: Path,
) -> None:
    (tmp_path / "s4").mkdir()
    transport = UnixShmTransport(str(tmp_path / "s{slot}"))
    assert transport.worker_started("rrd_") == {}
    carrier, extra = transport.worker_carrier("rrd_", 4, "abcd", 1, 1 << 20, 4096, 8192)
    names = {p.name for p in (tmp_path / "s4").iterdir()}
    assert extra["input_block"] in names
    assert extra["probe"] in names
    assert len(bytes.fromhex(extra["probe_nonce"])) == 8
    assert any(name.endswith("out") for name in names)
    carrier.close()
    assert list((tmp_path / "s4").iterdir()) == []


def test_a_slot_without_its_directory_gets_no_memory(tmp_path: Path) -> None:
    transport = UnixShmTransport(str(tmp_path / "s{slot}"))
    with pytest.raises(SharedSlotAllocationFailed, match="slot 2 has no shared memory volume"):
        transport.worker_carrier("rrd_", 2, "abcd", 1, 1 << 20, 4096, 4096)


def test_a_full_slot_directory_refuses_the_session(tmp_path: Path) -> None:
    (tmp_path / "s0").mkdir()
    transport = UnixShmTransport(str(tmp_path / "s{slot}"))

    def no_room(*_: Any) -> None:
        raise OSError(28, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("os.posix_fallocate", no_room)
        with pytest.raises(SharedSlotAllocationFailed):
            transport.worker_carrier("rrd_", 0, "abcd", 1, 1 << 20, 4096, 4096)
    assert list((tmp_path / "s0").iterdir()) == []


def test_a_request_cannot_name_a_path_outside_its_directory(tmp_path: Path) -> None:
    import json

    (tmp_path / "secret").write_bytes(b"x" * 64)
    prefix = "rrd_s0_abcd_"
    header = json.dumps(
        {"v": 1, "block": f"{prefix}/../secret", "spans": [[0, 4]], "tree": {"a": 0}}
    ).encode()
    reader = ViewReader(prefix, 1 << 20, tmp_path / "s0")
    with pytest.raises(RequestRefused, match="cannot be mapped"):
        reader.unpack(header)
    with pytest.raises(ValueError, match="not a block name"):
        FileBlock(tmp_path, "a/b")


def test_a_file_block_reads_back_what_was_written(tmp_path: Path) -> None:
    block = FileBlock(tmp_path, "blk", create=True, size=64)
    assert block.buf is not None
    block.buf[:3] = b"abc"
    other = FileBlock(tmp_path, "blk")
    assert other.size == 64
    assert other.buf is not None
    assert bytes(other.buf[:3]) == b"abc"
    other.close()
    block.close()
    block.unlink()
    assert not (tmp_path / "blk").exists()


def test_a_symlink_to_another_slots_block_is_refused(tmp_path: Path) -> None:
    import json
    import os

    (tmp_path / "s1").mkdir()
    (tmp_path / "s2").mkdir()
    victim = FileBlock(tmp_path / "s2", "rrd_s2_abcd_out", create=True, size=64)
    assert victim.buf is not None
    victim.buf[:6] = b"secret"
    prefix = "rrd_s1_abcd_"
    os.symlink(tmp_path / "s2" / "rrd_s2_abcd_out", tmp_path / "s1" / f"{prefix}g1")
    header = json.dumps(
        {"v": 1, "block": f"{prefix}g1", "spans": [[0, 6]], "tree": {"a": 0}}
    ).encode()
    reader = ViewReader(prefix, 1 << 20, tmp_path / "s1")
    with pytest.raises(RequestRefused, match="cannot be mapped"):
        reader.unpack(header)
    with pytest.raises(OSError, match="Too many levels of symbolic links"):
        FileBlock(tmp_path / "s1", f"{prefix}g1")
    reader.close()
    victim.close()


def test_a_block_that_is_not_a_regular_file_is_refused(tmp_path: Path) -> None:
    import os

    os.mkfifo(tmp_path / "rrd_s0_abcd_g1")
    with pytest.raises(ValueError, match="not a regular file"):
        FileBlock(tmp_path, "rrd_s0_abcd_g1")


@pytest.fixture
def shm_as_slot_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point every slot's directory at /dev/shm, where the client finds blocks by name.

    In a pod each session mounts its own volume at /dev/shm; here the test
    process is both sides, so the worker writes files the client maps through
    POSIX shared memory, the same pair of paths a pod gives them.
    """
    if not _SHM.is_dir():
        pytest.skip("needs /dev/shm")
    monkeypatch.setattr(UnixShmTransport, "slot_directory", lambda self, slot: _SHM)
    prefix = f"rrsd{secrets.token_hex(3)}_"
    yield prefix
    unlink_prefix(prefix)


def test_a_session_steps_through_blocks_kept_as_files(shm_as_slot_dir: str) -> None:
    worker = Doubler()
    worker.device = "cpu"
    worker.load(None)
    path = Path(tempfile.mkdtemp(prefix="rrsd")) / "w.sock"
    server = WorkerServer(
        worker, path, shm_prefix=shm_as_slot_dir, transport=UnixShmTransport("/x/s{slot}")
    )
    server.bind()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        client = ModelClient.connect(socket_path=path, slot=0, connect_timeout=5)
        client.reset()
        out = client.step(np.arange(4, dtype=np.int32))
        assert out["x"].tolist() == [0, 2, 4, 6]
        big = np.ones(1 << 16, dtype=np.float32)
        assert float(client.step(big)["x"].sum()) == 2.0 * big.size
        client.close()
    finally:
        server.stop()
        thread.join(10)
    assert not list(_SHM.glob(f"{shm_as_slot_dir}*"))
