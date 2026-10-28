"""The model server's messages: one frame each, pickled whole, exceptions carried as themselves."""

from __future__ import annotations

import pickle
import socket
import sys
import types
from typing import Any

import numpy as np
import pytest

from reactor_runtime.model_server import BatchFailed, WorkerError, WorkerRejected, protocol
from reactor_runtime.model_server.protocol import (
    PROTOCOL,
    Cancel,
    Closed,
    Hello,
    ProtocolError,
    Refused,
    Reset,
    ResetDone,
    Result,
    Signal,
    Step,
    Welcome,
    check_hello,
    check_message,
    error_of,
    failure,
    recv_frame,
    send_frame,
)


def _pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


# -- frames ---------------------------------------------------------------------------------


def test_a_frame_round_trips_its_bytes() -> None:
    a, b = _pair()
    with a, b:
        send_frame(a, b"body")
        send_frame(a, b"")
        assert recv_frame(b, 16) == b"body"
        assert recv_frame(b, 16) == b""


def test_a_frame_past_the_limit_is_refused_before_its_body_is_read() -> None:
    a, b = _pair()
    with a, b:
        # Announce 100 MB and send none of it: the reader must not wait for it.
        a.sendall(protocol.FRAME_HEADER.pack(100 << 20))
        b.settimeout(2)
        with pytest.raises(ProtocolError, match="passes the 1024 byte limit"):
            recv_frame(b, 1024)


def test_a_peer_that_closes_at_a_boundary_or_mid_frame_is_closed() -> None:
    a, b = _pair()
    with b:
        send_frame(a, b"whole")
        a.sendall(protocol.FRAME_HEADER.pack(10) + b"abc")
        a.close()
        assert recv_frame(b, 64) == b"whole"
        with pytest.raises(Closed):
            recv_frame(b, 64)
    c, d = _pair()
    with d:
        c.close()
        with pytest.raises(Closed):
            recv_frame(d, 64)


# -- messages -------------------------------------------------------------------------------


_MESSAGES: list[Any] = [
    Hello(slot=2),
    Step(request_id=7, timeout=1.5, payload={"audio": np.arange(4, dtype=np.int16)}),
    Step(request_id=8, timeout=None, payload=None),
    Cancel(request_id=7),
    Signal(signal=("preempt", "barge-in")),
    Reset(request_id=9, session_id="abc"),
    Welcome(slot=2, generation=3, max_batch=4, max_request_bytes=1 << 20),
    Refused(reason="slot 2 is held"),
    Result(request_id=7, output={"frames": np.zeros((2, 2), dtype=np.uint8)}),
    failure(7, ValueError("no face")),
    ResetDone(request_id=9, generation=4),
]


@pytest.mark.parametrize("message", _MESSAGES, ids=lambda m: type(m).__name__)
def test_every_message_pickles_whole_and_is_recognised(message: Any) -> None:
    back = check_message(pickle.loads(pickle.dumps(message, protocol=5)))
    assert type(back) is type(message)
    assert repr(back) == repr(message)


def test_a_frame_that_holds_no_message_is_refused() -> None:
    with pytest.raises(ProtocolError, match="not a protocol message"):
        check_message({"kind": "step"})


def test_a_hello_in_another_protocol_is_refused_naming_both_versions() -> None:
    check_hello(Hello(slot=0))
    with pytest.raises(ProtocolError, match=f"protocol {PROTOCOL + 1} and the worker speaks"):
        check_hello(Hello(slot=0, protocol=PROTOCOL + 1))


# -- exceptions -----------------------------------------------------------------------------


class NoFaceError(ValueError):
    pass


def test_a_models_exception_arrives_as_itself() -> None:
    error = error_of(failure(3, NoFaceError("no face in the portrait")))
    assert type(error) is NoFaceError
    assert str(error) == "no face in the portrait"


def test_batch_failed_carries_its_cause() -> None:
    error = error_of(failure(3, BatchFailed(NoFaceError("bad input"))))
    assert isinstance(error, BatchFailed)
    assert type(error.cause) is NoFaceError
    assert "NoFaceError: bad input" in str(error)


def test_the_runtimes_own_exceptions_arrive_as_themselves() -> None:
    error = error_of(failure(3, WorkerRejected("too large")))
    assert type(error) is WorkerRejected
    assert str(error) == "too large"


def test_an_exception_that_does_not_pickle_arrives_as_a_runtime_error_naming_it() -> None:
    class HolderError(Exception):
        def __init__(self, handle: Any) -> None:
            super().__init__(handle)

    error = error_of(failure(3, HolderError(lambda: None)))
    assert type(error) is RuntimeError
    assert "HolderError" in str(error)


def test_an_exception_this_process_cannot_import_arrives_as_a_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("only_in_the_worker")
    exec("class Missing(Exception):\n    pass\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "only_in_the_worker", module)
    failed = failure(3, module.Missing("gone"))
    monkeypatch.delitem(sys.modules, "only_in_the_worker")

    error = error_of(failed)

    assert type(error) is WorkerError
    assert str(error) == "only_in_the_worker.Missing: gone"
