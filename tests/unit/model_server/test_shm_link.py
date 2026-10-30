"""The shared-memory link: headers on the socket, arrays in the session's own directory."""

from __future__ import annotations

import contextlib
import errno
import os
import pickle
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.ipc import Block, SharedSlotAllocationFailed, blocks
from reactor_runtime.ipc import slot as slot_module
from reactor_runtime.model_server import (
    PayloadTooLarge,
    ShmLink,
    UnixShmTransport,
    WorkerLink,
    WorkerUnreachable,
    shm_link,
)
from reactor_runtime.model_server.link import NoListener, StreamLink
from reactor_runtime.model_server.protocol import (
    Cancel,
    Closed,
    Hello,
    ProtocolError,
    Result,
    Signal,
    Step,
)
from reactor_runtime.model_server.shm_link import ShmHello, ShmRefused


def _transport(
    root: Path, *, session_directory: Path | None = None, size: int = 4096
) -> UnixShmTransport:
    return UnixShmTransport(
        slot_directory=str(root / "s{slot}"),
        session_directory=session_directory or root / "s0",
        input_bytes=size,
        output_bytes=size,
    )


class Pair:
    """A model server end and a session end of one link, connected through a real socket."""

    def __init__(self, root: Path, transport: UnixShmTransport) -> None:
        self.address = str(root / "sock")
        self.listener = transport.listen(self.address)
        accepted: list[ShmLink] = []
        acceptor = threading.Thread(target=lambda: accepted.append(self.listener.accept()))
        acceptor.start()
        self.session: ShmLink | None = None
        self.transport = transport
        self._accepted = accepted
        self._acceptor = acceptor

    def open(self, slot: int = 0) -> tuple[ShmLink, ShmLink]:
        # The model server's end sets itself up on its first receive, while the
        # session's connect waits for the answer: run that receive on a thread.
        received: list[object] = []
        worker_end: list[ShmLink] = []

        def serve() -> None:
            self._acceptor.join(timeout=5)
            worker_end.append(self._accepted[0])
            received.append(self._accepted[0].receive())

        server = threading.Thread(target=serve)
        server.start()
        self.session = self.transport.connect(self.address, slot)
        self.session.send(Hello(slot=slot))
        server.join(timeout=5)
        assert received == [Hello(slot=slot)]
        return self.session, worker_end[0]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "s0").mkdir()
    return tmp_path


def _start_accepting(transport: UnixShmTransport, address: str) -> list[ShmLink]:
    listener = transport.listen(address)
    accepted: list[ShmLink] = []

    def serve() -> None:
        link = listener.accept()
        accepted.append(link)
        try:
            link.receive()
        except Exception:
            link.close()

    threading.Thread(target=serve, daemon=True).start()
    return accepted


def test_a_shm_link_is_a_worker_link(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        assert isinstance(session, WorkerLink)
        assert isinstance(worker, WorkerLink)
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_arrays_travel_in_the_slot_directory_and_arrive_as_copies(root: Path) -> None:
    pair = Pair(root, _transport(root, size=64 << 10))
    session, worker = pair.open()
    frames = np.random.default_rng(0).integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    audio = np.arange(26880, dtype=np.int16)
    sent: list[int] = []
    raw_send = session._stream.send_bytes

    def measure(data: bytes) -> None:
        sent.append(len(data))
        raw_send(data)

    session._stream.send_bytes = measure  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]  # count the bytes on the socket
    try:
        session.send(Step(request_id=1, timeout=None, payload={"audio": audio}))
        step = worker.receive()
        np.testing.assert_array_equal(step.payload["audio"], audio)
        assert sent[0] < audio.nbytes // 10  # only the header crossed the socket
        step.payload["audio"][0] = 7  # its own copy, writable

        worker.send(Result(request_id=1, output={"frames": frames}))
        result = session.receive()
        np.testing.assert_array_equal(result.output["frames"], frames)
        names = sorted(p.name for p in (root / "s0").iterdir())
        assert len(names) == 2  # one block for each end's writes; the probe is gone
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_after_a_cancel_the_next_request_is_written_to_a_new_block(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        session.send(Step(request_id=1, timeout=None, payload=np.ones(16, dtype=np.uint8)))
        worker.receive()
        before = session._writer.name if session._writer is not None else None
        session.send(Cancel(request_id=1))
        worker.receive()
        after = session._writer.name if session._writer is not None else None
        assert before != after
        assert (root / "s0" / str(before)).exists()  # retired, kept until the next rotation
        session.send(Step(request_id=2, timeout=None, payload=np.full(16, 3, dtype=np.uint8)))
        np.testing.assert_array_equal(worker.receive().payload, np.full(16, 3, dtype=np.uint8))
        session.send(Cancel(request_id=2))
        worker.receive()
        assert not (root / "s0" / str(before)).exists()  # released by the next rotation
        assert (root / "s0" / str(after)).exists()
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_request_larger_than_its_block_is_refused_and_the_link_goes_on(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        block = session._writer.name if session._writer is not None else None
        with pytest.raises(PayloadTooLarge) as refused:
            session.send(Step(request_id=1, timeout=None, payload=np.ones(5000, dtype=np.uint8)))
        assert str(refused.value) == (
            "a request's arrays hold 5000 bytes, above the 4096 bytes of its shared-memory "
            "block: raise runtime.worker.shm_input_bytes in reactor.yaml"
        )

        session.send(Step(request_id=2, timeout=None, payload=np.ones(4096, dtype=np.uint8)))
        assert worker.receive().request_id == 2  # nothing of the refused request was sent
        assert session._writer is not None
        assert (session._writer.name, session._writer.capacity) == (block, 4096)
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_result_larger_than_its_block_is_refused_naming_the_output_size(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        with pytest.raises(PayloadTooLarge, match=r"runtime\.worker\.shm_output_bytes"):
            worker.send(Result(request_id=1, output=np.ones(5000, dtype=np.uint8)))

        worker.send(Result(request_id=1, output=np.ones(16, dtype=np.uint8)))
        assert session.receive().request_id == 1
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_slot_holds_at_most_two_request_blocks_and_one_output_block(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        for request_id in range(1, 4):
            payload = np.ones(4096, dtype=np.uint8)
            session.send(Step(request_id=request_id, timeout=None, payload=payload))
            worker.receive()
            worker.send(Result(request_id=request_id, output=np.ones(4096, dtype=np.uint8)))
            session.receive()
            session.send(Cancel(request_id=request_id))
            worker.receive()
            sizes = sorted(path.stat().st_size for path in (root / "s0").iterdir())
            assert sizes in ([4096, 4096], [4096, 4096, 4096])
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_link_that_cancels_hundreds_of_times_keeps_short_block_names(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        for request_id in range(1, 301):
            session.send(Cancel(request_id=request_id))
            worker.receive()
        session.send(Step(request_id=301, timeout=None, payload=np.ones(16, dtype=np.uint8)))
        np.testing.assert_array_equal(worker.receive().payload, np.ones(16, dtype=np.uint8))
        assert max(len(path.name) for path in (root / "s0").iterdir()) < 64
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_cancel_whose_replacement_cannot_be_allocated_is_not_sent(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        block = session._writer.name if session._writer is not None else None

        def full(name: str, size: int, directory: Path | None = None) -> Block:
            raise SharedSlotAllocationFailed(f"no room for {size} bytes")

        monkeypatch.setattr(slot_module, "create_block", full)
        with pytest.raises(SharedSlotAllocationFailed):
            session.send(Cancel(request_id=1))
        monkeypatch.undo()

        session.send(Step(request_id=2, timeout=None, payload=np.ones(16, dtype=np.uint8)))
        assert worker.receive().request_id == 2  # the cancel never reached the peer
        assert session._writer is not None
        assert session._writer.name == block
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_cancel_whose_send_fails_leaves_no_replacement_behind(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        before = sorted(path.name for path in (root / "s0").iterdir())

        def broken(data: bytes) -> None:
            raise OSError(errno.EPIPE, "broken pipe")

        session._stream.send_bytes = broken  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]  # a send that fails
        with pytest.raises(OSError, match="broken pipe"):
            session.send(Cancel(request_id=1))

        assert sorted(path.name for path in (root / "s0").iterdir()) == before
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_step_cancelled_before_the_model_server_read_it_still_arrives_intact(
    root: Path,
) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        frames = np.arange(4096, dtype=np.uint8)
        session.send(Step(request_id=1, timeout=None, payload=frames))
        session.send(Cancel(request_id=1))
        session.send(Step(request_id=2, timeout=None, payload=np.zeros(4096, dtype=np.uint8)))

        np.testing.assert_array_equal(worker.receive().payload, frames)
        assert worker.receive() == Cancel(request_id=1)
        np.testing.assert_array_equal(worker.receive().payload, np.zeros(4096, dtype=np.uint8))
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_step_whose_block_two_cancels_released_is_skipped(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        second = np.full(4096, 2, dtype=np.uint8)
        third = np.full(4096, 3, dtype=np.uint8)
        # The model server reads nothing until all five are sent: the second
        # cancel releases the first step's block before its header is read.
        session.send(Step(request_id=1, timeout=None, payload=np.ones(4096, dtype=np.uint8)))
        session.send(Cancel(request_id=1))
        session.send(Step(request_id=2, timeout=None, payload=second))
        session.send(Cancel(request_id=2))
        session.send(Step(request_id=3, timeout=None, payload=third))

        assert worker.receive() == Cancel(request_id=1)
        np.testing.assert_array_equal(worker.receive().payload, second)
        assert worker.receive() == Cancel(request_id=2)
        np.testing.assert_array_equal(worker.receive().payload, third)
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def _raise_file_not_found() -> None:
    raise FileNotFoundError("raised while the payload is rebuilt")


class _MissingOnLoad:
    """A payload whose unpickling raises FileNotFoundError, unrelated to any block."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_raise_file_not_found, ())


def _open_on(
    listener: object, transport: UnixShmTransport, address: str, slot: int
) -> tuple[ShmLink, ShmLink]:
    """Accept one more link on *listener* and set it up, as Pair.open does."""
    worker_end: list[ShmLink] = []
    received: list[object] = []

    def serve() -> None:
        link = listener.accept()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # the listener under test
        worker_end.append(link)
        received.append(link.receive())

    server = threading.Thread(target=serve)
    server.start()
    session = transport.connect(address, slot)
    session.send(Hello(slot=slot))
    server.join(timeout=5)
    assert received == [Hello(slot=slot)]
    return session, worker_end[0]


def _output_blocks(root: Path, transport: UnixShmTransport) -> list[str]:
    return sorted(
        p.name for p in (root / "s0").iterdir() if p.name.startswith(transport.output_prefix(0))
    )


def test_a_payload_that_raises_file_not_found_while_unpickled_is_a_protocol_error(
    root: Path,
) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        session.send(Step(request_id=1, timeout=None, payload=_MissingOnLoad()))
        with pytest.raises(ProtocolError, match="cannot be read"):
            worker.receive()
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_the_model_server_releases_a_links_blocks_once_its_session_is_gone(root: Path) -> None:
    transport = _transport(root)
    pair = Pair(root, transport)
    session, worker = pair.open()
    try:
        assert _output_blocks(root, transport)
        session.close()  # the session runtime exits during a step
        with pytest.raises(Closed):
            worker.receive()

        assert _output_blocks(root, transport) == []
        with pytest.raises(Closed, match="the link is closed"):
            worker.send(Result(1, np.zeros(16, dtype=np.uint8)))  # the step's late result
    finally:
        worker.close()
        pair.listener.close()


def test_a_new_link_for_a_slot_releases_the_previous_links_blocks_first(root: Path) -> None:
    transport = _transport(root)
    pair = Pair(root, transport)
    first_session, first_worker = pair.open()
    try:
        first_blocks = _output_blocks(root, transport)
        second_session, second_worker = _open_on(pair.listener, transport, pair.address, 0)
        try:
            remaining = _output_blocks(root, transport)
            assert len(remaining) == 1
            assert remaining != first_blocks
            with pytest.raises(Closed, match="the link is closed"):
                first_worker.send(Result(1, np.zeros(16, dtype=np.uint8)))
            frames = np.arange(4096, dtype=np.uint8)
            second_session.send(Step(request_id=1, timeout=None, payload=frames))
            np.testing.assert_array_equal(second_worker.receive().payload, frames)
        finally:
            second_session.close()
            second_worker.close()
    finally:
        first_session.close()
        first_worker.close()
        pair.listener.close()


def test_overlapping_handshakes_for_one_slot_leave_one_output_block(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _transport(root)
    address = str(root / "sock")
    listener = transport.listen(address)
    real = shm_link._fixed_slot
    first_allocating = threading.Event()
    go_on = threading.Event()
    calls: list[str] = []

    def paused(prefix: str, size: int, directory: Path | None) -> object:
        if prefix.startswith(transport.output_prefix(0)):
            calls.append(prefix)
            if len(calls) == 1:
                first_allocating.set()
                go_on.wait(5)  # the first link holds the slot, not yet allocated
        return real(prefix, size, directory)

    monkeypatch.setattr(shm_link, "_fixed_slot", paused)
    workers: list[ShmLink] = []
    sessions: list[ShmLink] = []

    def serve() -> None:
        link = listener.accept()
        workers.append(link)
        with contextlib.suppress(Exception):
            link.receive()

    def connect() -> None:
        session = transport.connect(address, 0)
        sessions.append(session)
        session.send(Hello(slot=0))

    threads = []
    try:
        for _ in range(2):
            threading.Thread(target=serve, daemon=True).start()
        first = threading.Thread(target=connect)
        first.start()
        threads.append(first)
        assert first_allocating.wait(5)
        second = threading.Thread(target=connect)
        second.start()
        threads.append(second)
        time.sleep(0.3)  # the second handshake reaches the slot while the first allocates
        go_on.set()
        for t in threads:
            t.join(timeout=5)

        assert len(calls) == 2
        assert len(_output_blocks(root, transport)) == 1
    finally:
        go_on.set()
        for link in sessions + workers:
            link.close()
        listener.close()


def test_receive_on_a_released_link_is_closed(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        worker.release_blocks()  # as a newer link for the slot does
        session.send(Cancel(request_id=1))  # a frame the old peer still sends
        with pytest.raises(Closed, match="the link is closed"):
            worker.receive()
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_step_and_a_signal_with_arrays_sent_back_to_back_both_arrive_intact(
    root: Path,
) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        frames = np.arange(4096, dtype=np.uint8)
        hint = np.full(256, 9, dtype=np.uint8)
        session.send(Step(request_id=1, timeout=None, payload=frames))
        session.send(Signal(signal={"hint": hint}))

        np.testing.assert_array_equal(worker.receive().payload, frames)
        np.testing.assert_array_equal(worker.receive().signal["hint"], hint)
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_signal_past_the_frame_limit_is_refused_and_the_link_goes_on(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    try:
        with pytest.raises(PayloadTooLarge, match="a signal carries small values"):
            session.send(Signal(signal=bytes(2 << 20)))
        session.send(Signal(signal="small"))
        assert worker.receive() == Signal(signal="small")
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_an_inline_payload_past_the_frame_limit_is_refused_both_ways(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    inline = np.zeros((3000, 1000), dtype=np.uint8)[
        :, ::2
    ]  # 1.5 MB, not contiguous: pickled inline
    try:
        with pytest.raises(PayloadTooLarge, match=r"a request pickled \d+ bytes inline"):
            session.send(Step(request_id=1, timeout=None, payload=inline))
        with pytest.raises(PayloadTooLarge, match=r"np\.ascontiguousarray"):
            worker.send(Result(request_id=1, output=inline))

        session.send(Step(request_id=2, timeout=None, payload=b"ok"))
        assert worker.receive().request_id == 2
        worker.send(Result(request_id=2, output=b"ok"))
        assert session.receive().request_id == 2
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_closing_the_model_servers_end_reclaims_what_a_dead_session_left(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    session.send(Step(request_id=1, timeout=None, payload=np.ones(16, dtype=np.uint8)))
    worker.receive()
    session._stream.close()  # the session dies without releasing its blocks

    worker.close()

    assert list((root / "s0").iterdir()) == []
    pair.listener.close()


def test_blocks_are_named_by_role_and_slot_and_a_restart_clears_its_own(root: Path) -> None:
    (root / "s1").mkdir()
    transport = UnixShmTransport(
        slot_directory=str(root / "s{slot}"),
        session_directory=root / "s1",
        input_bytes=4096,
        output_bytes=4096,
    )
    # What a killed session runtime and a killed model server left behind.
    for name in ("rrms_s1_dead_0", "rrms_w1_dead_0", "rrms_w0_dead_0", "other_0"):
        target = root / ("s0" if "w0" in name else "s1") / name
        target.write_bytes(b"x")

    assert transport.reclaim(2) == 2
    pair = Pair(root, transport)
    session, worker = pair.open(slot=1)
    try:
        names = sorted(p.name for p in (root / "s1").iterdir())
        assert "rrms_s1_dead_0" not in names
        assert "other_0" in names
        assert any(n.startswith("rrms_s1_") for n in names)
        assert any(n.startswith("rrms_w1_") for n in names)
    finally:
        session.close()
        worker.close()
        pair.listener.close()


def test_a_session_that_does_not_see_the_servers_memory_is_unreachable(root: Path) -> None:
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    transport = _transport(root, session_directory=elsewhere)
    address = str(root / "sock")
    _start_accepting(transport, address)

    with pytest.raises(WorkerUnreachable, match="not visible in"):
        transport.connect(address, 0)


def test_a_slot_without_a_directory_is_unreachable_naming_it(root: Path) -> None:
    transport = _transport(root)
    address = str(root / "sock")
    _start_accepting(transport, address)

    with pytest.raises(WorkerUnreachable, match="slot 3 has no shared-memory directory"):
        transport.connect(address, 3)


def test_a_slot_without_room_for_its_blocks_is_unreachable_naming_it(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_room(fd: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(blocks.os, "posix_fallocate", no_room, raising=False)
    transport = _transport(root)
    address = str(root / "sock")
    _start_accepting(transport, address)

    with pytest.raises(WorkerUnreachable, match="no shared memory for slot 0"):
        transport.connect(address, 0)
    assert list((root / "s0").iterdir()) == []


def test_nothing_listening_is_worth_retrying_and_no_socket_directory_is_not(root: Path) -> None:
    transport = _transport(root)
    with pytest.raises(NoListener):
        transport.connect(str(root / "sock"), 0)
    with pytest.raises(WorkerUnreachable, match="socket directory"):
        transport.connect(str(root / "missing" / "sock"), 0)
    with pytest.raises(WorkerUnreachable, match="socket directory"):
        transport.listen(str(root / "missing" / "sock"))


def test_the_socket_is_open_to_its_user_alone_and_replaces_a_stale_one(root: Path) -> None:
    transport = _transport(root)
    address = root / "sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(address))
    stale.close()  # a socket file nothing accepts on any more
    listener = transport.listen(str(address))
    try:
        assert os.stat(address).st_mode & 0o777 == 0o600
    finally:
        listener.close()
    assert not address.exists()


def test_a_live_socket_is_never_taken_over(root: Path) -> None:
    transport = _transport(root)
    address = str(root / "sock")
    first = transport.listen(address)
    try:
        with pytest.raises(WorkerUnreachable, match="already listens"):
            transport.listen(address)
        assert Path(address).exists()
        pair_session = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        pair_session.connect(address)  # the first listener still accepts
        pair_session.close()
    finally:
        first.close()


def test_a_listener_removes_only_its_own_socket_file(root: Path) -> None:
    transport = _transport(root)
    address = root / "sock"
    first = transport.listen(str(address))
    first._sock.close()  # its socket is gone, its file looks stale
    second = transport.listen(str(address))
    try:
        first.close()
        assert address.exists()  # the second listener's file stays
    finally:
        second.close()
    assert not address.exists()


def _raw_hello(address: str, hello: object) -> object:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(address)
    stream = StreamLink(sock)
    try:
        stream.send_bytes(pickle.dumps(hello))
        return pickle.loads(stream.receive_bytes())
    finally:
        stream.close()


@pytest.mark.parametrize("prefix", ["", "rrms_s0_/x", "rrms_s0_", "rrms_s1_abc", "rrms_"])
def test_an_invalid_request_prefix_refuses_the_link_and_unlinks_nothing(
    root: Path, prefix: str
) -> None:
    transport = _transport(root)
    address = str(root / "sock")
    listener = transport.listen(address)
    keep = root / "s0" / "rrms_w0_keep"
    keep.write_bytes(b"x")
    raised: list[BaseException] = []

    def serve() -> None:
        link = listener.accept()
        try:
            link.receive()
        except BaseException as exc:
            raised.append(exc)
        finally:
            link.close()

    server = threading.Thread(target=serve)
    server.start()
    try:
        answer = _raw_hello(address, ShmHello(0, prefix))
        server.join(timeout=5)
        assert isinstance(answer, ShmRefused)
        assert "invalid request prefix" in answer.reason
        assert [type(exc) for exc in raised] == [ProtocolError]
        assert keep.exists()
    finally:
        listener.close()


def test_a_session_without_room_for_its_request_block_is_unreachable(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _transport(root)
    address = str(root / "sock")
    accepted = _start_accepting(transport, address)
    real = slot_module.create_block

    def session_full(name: str, size: int, directory: Path | None = None) -> Block:
        if name.startswith(transport.request_prefix(0)):
            raise SharedSlotAllocationFailed(f"no room for {size} bytes")
        return real(name, size, directory)

    monkeypatch.setattr(slot_module, "create_block", session_full)
    with pytest.raises(WorkerUnreachable, match="no room for a 4096-byte request block"):
        transport.connect(address, 0)
    assert not any(p.name.startswith("rrms_s0_") for p in (root / "s0").iterdir())
    for link in accepted:
        link.close()


def test_a_model_server_that_never_finishes_the_setup_is_retried_after_the_timeout(
    root: Path,
) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = str(root / "sock")
    server.bind(address)
    server.listen(1)  # accepts at the socket level, never answers the setup
    try:
        started = time.monotonic()
        with pytest.raises(NoListener, match="did not finish setting up the link in time"):
            _transport(root).connect(address, 0, timeout=0.3)
        assert time.monotonic() - started < 2
    finally:
        server.close()


def test_the_connect_timeout_bounds_a_setup_answer_that_trickles_in(root: Path) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = str(root / "sock")
    server.bind(address)
    server.listen(1)

    def trickle() -> None:
        peer, _ = server.accept()
        with peer:
            peer.recv(4096)  # the session's hello
            try:
                for chunk in (b"\x00\x00", b"\x00\x10", b"x"):
                    time.sleep(0.25)  # each read alone is inside the timeout
                    peer.sendall(chunk)
            except BrokenPipeError:
                return  # the session gave up, as it should

    answering = threading.Thread(target=trickle, daemon=True)
    answering.start()
    try:
        started = time.monotonic()
        with pytest.raises(NoListener, match="did not finish setting up the link in time"):
            _transport(root).connect(address, 0, timeout=0.4)
        assert time.monotonic() - started < 0.6
    finally:
        server.close()
        answering.join(timeout=5)


def test_a_connect_with_no_time_left_says_so(root: Path) -> None:
    address = str(root / "sock")
    listener = _transport(root).listen(address)
    try:
        with pytest.raises(NoListener, match="timeout passed"):
            _transport(root).connect(address, 0, timeout=0.0)
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("failure", "live"),
    [
        (ConnectionRefusedError(errno.ECONNREFUSED, "refused"), False),
        (FileNotFoundError(errno.ENOENT, "gone"), False),
        (BlockingIOError(errno.EAGAIN, "backlog full"), True),
        (PermissionError(errno.EACCES, "another user's"), True),
        (TimeoutError("timed out"), True),
    ],
)
def test_only_a_refused_or_missing_socket_is_stale(
    root: Path, monkeypatch: pytest.MonkeyPatch, failure: OSError, live: bool
) -> None:
    def connect(self: socket.socket, address: object) -> None:
        raise failure

    monkeypatch.setattr(socket.socket, "connect", connect)
    assert shm_link._accepts(str(root / "sock")) is live


def test_a_slot_directory_names_its_slot() -> None:
    with pytest.raises(ValueError, match="has no"):
        UnixShmTransport(slot_directory="/dev/shm")
    assert UnixShmTransport(slot_directory="/run/s{slot}").slot_directory(2) == Path("/run/s2")
    with pytest.raises(ValueError, match="can connect, not listen"):
        UnixShmTransport().listen("/tmp/unused-sock")


def test_closing_the_session_end_ends_a_blocked_receive(root: Path) -> None:
    pair = Pair(root, _transport(root))
    session, worker = pair.open()
    outcome: list[BaseException] = []

    def wait() -> None:
        try:
            session.receive()
        except BaseException as exc:
            outcome.append(exc)

    reader = threading.Thread(target=wait)
    reader.start()
    session.close()
    reader.join(timeout=5)
    worker.close()
    pair.listener.close()

    assert len(outcome) == 1
    assert isinstance(outcome[0], Closed)
