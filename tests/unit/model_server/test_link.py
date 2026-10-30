"""A link carries whole protocol messages, and closing it from any thread ends a blocked receive."""

from __future__ import annotations

import contextlib
import pickle
import socket
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from reactor_runtime.model_server import (
    LinkListener,
    LinkTransport,
    NoListener,
    PayloadTooLarge,
    StreamLink,
    WorkerLink,
)
from reactor_runtime.model_server.link import NoListener as LinkNoListener
from reactor_runtime.model_server.protocol import (
    Closed,
    Hello,
    ProtocolError,
    Result,
    Step,
    failure,
)


@pytest.fixture
def links() -> Iterator[tuple[StreamLink, StreamLink]]:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    session, worker = StreamLink(a), StreamLink(b, frame_limit=1 << 22)
    yield session, worker
    session.close()
    worker.close()


def test_a_stream_link_is_a_worker_link() -> None:
    a, b = socket.socketpair()
    with a, b:
        assert isinstance(StreamLink(a), WorkerLink)


def test_messages_cross_whole_with_their_arrays_inline(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, worker = links
    audio = np.arange(26880, dtype=np.int16)

    session.send(Hello(slot=1))
    session.send(Step(request_id=4, timeout=None, payload={"audio": audio}))
    worker.send(Result(request_id=4, output={"echo": audio}))
    worker.send(failure(5, ValueError("bad")))

    assert worker.receive() == Hello(slot=1)
    step = worker.receive()
    np.testing.assert_array_equal(step.payload["audio"], audio)
    np.testing.assert_array_equal(session.receive().output["echo"], audio)
    assert session.receive().request_id == 5


def test_only_protocol_messages_are_sent_or_received(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, worker = links
    with pytest.raises(ProtocolError, match="not a protocol message"):
        session.send({"step": 1})
    session.send_bytes(pickle.dumps({"step": 1}))
    with pytest.raises(ProtocolError, match="not a protocol message"):
        worker.receive()


def test_a_message_past_the_frame_limit_is_refused_and_the_link_goes_on(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, worker = links

    with pytest.raises(PayloadTooLarge, match="1048576-byte frame limit"):
        session.send(Step(request_id=1, timeout=None, payload=bytes(2 << 20)))
    session.send(Step(request_id=2, timeout=None, payload=b"small"))

    assert worker.receive() == Step(request_id=2, timeout=None, payload=b"small")


def test_a_message_that_does_not_pickle_is_raised_and_the_link_goes_on(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, worker = links

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        session.send(Step(request_id=1, timeout=None, payload=lambda: None))
    session.send(Step(request_id=2, timeout=None, payload=1))

    assert worker.receive().request_id == 2


@pytest.mark.parametrize(
    "data",
    [b"\x00not a pickle", pickle.dumps(Hello(slot=1), protocol=5)[:-3]],
    ids=["garbage", "truncated"],
)
def test_a_frame_that_cannot_be_unpickled_is_a_protocol_error(
    links: tuple[StreamLink, StreamLink], data: bytes
) -> None:
    session, worker = links
    session.send_bytes(data)

    with pytest.raises(ProtocolError, match="does not hold a message"):
        worker.receive()


def test_a_send_racing_close_raises_closed(links: tuple[StreamLink, StreamLink]) -> None:
    session, _ = links  # the peer never reads, so a large frame blocks in sendall
    outcome: list[BaseException] = []

    def push() -> None:
        try:
            session.send_bytes(bytes(1 << 20))
        except BaseException as exc:
            outcome.append(exc)

    sender = threading.Thread(target=push, daemon=True)
    sender.start()
    sender.join(timeout=0.2)
    assert sender.is_alive()  # blocked on the full socket buffer
    session.close()
    sender.join(timeout=5)

    assert [type(exc) for exc in outcome] == [Closed]


def test_no_listener_is_exported_with_the_package() -> None:
    assert NoListener is LinkNoListener


def test_a_frame_past_the_limit_from_a_peer_with_a_larger_limit_is_refused(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, worker = links

    # The worker end was built with a larger limit than the session end, which
    # a deployment never does; the reader still refuses the frame from its
    # header. Sent on its own thread: the frame is larger than the socket's
    # buffer, and the reader refuses it without draining it.
    def push() -> None:
        with contextlib.suppress(OSError):
            worker.send(Result(request_id=1, output=bytes(2 << 20)))

    sender = threading.Thread(target=push, daemon=True)
    sender.start()
    with pytest.raises(ProtocolError, match="passes the"):
        session.receive()
    session.close()
    sender.join(timeout=5)


def test_close_from_another_thread_ends_a_blocked_receive(
    links: tuple[StreamLink, StreamLink],
) -> None:
    session, _ = links
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

    assert not reader.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], Closed)
    with pytest.raises(Closed):
        session.send(Hello(slot=1))
    session.close()  # idempotent


def test_the_peer_closing_ends_receive(links: tuple[StreamLink, StreamLink]) -> None:
    session, worker = links
    worker.close()
    with pytest.raises(Closed):
        session.receive()


class _Listener:
    def accept(self) -> WorkerLink:
        raise OSError("closed")

    def close(self) -> None:
        pass


class _Transport:
    def listen(self, address: str) -> LinkListener:
        return _Listener()

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> WorkerLink:
        a, _ = socket.socketpair()
        return StreamLink(a)


def test_a_transport_conforms_by_shape() -> None:
    assert isinstance(_Transport(), LinkTransport)
    assert isinstance(_Listener(), LinkListener)
