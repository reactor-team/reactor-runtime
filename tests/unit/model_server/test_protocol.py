"""The model server's messages: one frame each, pickled whole, exceptions carried as themselves."""

from __future__ import annotations

import asyncio
import pickle
import socket
import sys
import types
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed.rank import portable_pickle
from reactor_runtime.model_server import (
    BatchFailed,
    PayloadTooLarge,
    WorkerError,
    WorkerRejected,
    protocol,
)
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
    Welcome(slot=2, generation=3, max_batch=4),
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


def test_a_subclass_of_a_message_is_not_that_message() -> None:
    class Extended(Step):
        pass

    with pytest.raises(ProtocolError, match="not a protocol message"):
        check_message(Extended(request_id=1, timeout=None, payload=None))


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
    failed = failure(3, BatchFailed(NoFaceError("bad input")))
    error = error_of(failed)
    assert failed.batch
    assert isinstance(error, BatchFailed)
    assert type(error.cause) is NoFaceError
    assert "NoFaceError: bad input" in str(error)


def test_a_batch_failure_whose_cause_does_not_pickle_is_still_a_batch_failure() -> None:
    error = error_of(failure(3, BatchFailed(HolderError(lambda: None))))
    assert isinstance(error, BatchFailed)
    assert type(error.cause) is RuntimeError
    assert "HolderError" in str(error.cause)


def test_a_batch_failure_whose_cause_is_too_large_is_still_a_batch_failure() -> None:
    error = error_of(failure(3, BatchFailed(ValueError("x" * (2 << 20)))))
    assert isinstance(error, BatchFailed)
    assert type(error.cause) is RuntimeError
    assert "too large to send whole" in str(error.cause)


def test_a_batch_failure_whose_cause_only_the_worker_imports_is_still_a_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("only_in_the_worker")
    exec("class Missing(Exception):\n    pass\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "only_in_the_worker", module)
    failed = failure(3, BatchFailed(module.Missing("gone")))
    monkeypatch.delitem(sys.modules, "only_in_the_worker")

    error = error_of(failed)

    assert isinstance(error, BatchFailed)
    assert type(error.cause) is WorkerError
    assert str(error.cause) == "only_in_the_worker.Missing: gone"


def test_a_batch_failure_that_falls_back_whole_is_still_a_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(request_id: int, exc: BatchFailed) -> Any:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(protocol, "_batch_failure", unexpected)
    failed = failure(3, BatchFailed(NoFaceError("bad input")))

    error = error_of(failed)

    assert isinstance(error, BatchFailed)
    assert type(error.cause) is RuntimeError
    assert "BatchFailed" in str(error.cause)


@pytest.mark.parametrize("raised", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(3)])
def test_a_base_exception_that_is_not_an_exception_arrives_as_a_runtime_error(
    raised: BaseException,
) -> None:
    error = error_of(failure(3, raised))
    assert type(error) is RuntimeError
    assert type(raised).__name__ in str(error)


@pytest.mark.parametrize("kind", [WorkerRejected, PayloadTooLarge])
def test_the_runtimes_own_exceptions_arrive_as_themselves(kind: type[Exception]) -> None:
    error = error_of(failure(3, kind("too large")))
    assert type(error) is kind
    assert str(error) == "too large"


class HolderError(Exception):
    def __init__(self, handle: Any) -> None:
        super().__init__(handle)


class UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot print")


class ProtocolPickyError(Exception):
    """Pickles in protocol 4 and not in protocol 5."""

    def __reduce_ex__(self, protocol: Any) -> Any:
        if protocol >= 5:
            raise TypeError("not in protocol 5")
        return (ProtocolPickyError, self.args)


def test_an_exception_that_does_not_pickle_arrives_as_a_runtime_error_naming_it() -> None:
    error = error_of(failure(3, HolderError(lambda: None)))
    assert type(error) is RuntimeError
    assert "HolderError" in str(error)


def test_an_exception_that_neither_pickles_nor_prints_still_answers_its_request() -> None:
    unprintable = UnprintableError()
    unprintable.handle = lambda: None  # ty: ignore[unresolved-attribute]  # an attribute that does not pickle

    failed = failure(3, unprintable)

    assert failed.request_id == 3
    assert failed.description.endswith("UnprintableError: <unprintable>")
    error = error_of(failed)
    assert type(error) is RuntimeError
    assert str(error) == failed.description


def test_the_pickle_checked_is_the_pickle_sent() -> None:
    error = error_of(failure(3, ProtocolPickyError("picky")))
    assert type(error) is RuntimeError
    assert "ProtocolPickyError: picky" in str(error)


def _rebuild_heavy(payload: bytes) -> HeavyError:
    HeavyError.rebuilt += 1
    return HeavyError(payload)


class HeavyError(Exception):
    """Counts how often it is unpickled."""

    rebuilt = 0

    def __reduce__(self) -> Any:
        return (_rebuild_heavy, self.args)


def test_portable_pickle_never_unpickles_an_exception_past_its_limit() -> None:
    replaced, _ = portable_pickle(HeavyError(b"x" * 4096), limit=1024)

    assert type(replaced) is RuntimeError
    assert "too large to send whole" in str(replaced)
    assert HeavyError.rebuilt == 0


def test_portable_pickle_checks_the_protocol_it_is_given() -> None:
    picky = ProtocolPickyError("picky")

    kept, blob = portable_pickle(picky, protocol=4)
    replaced, _ = portable_pickle(picky, protocol=5)

    assert kept is picky
    assert type(pickle.loads(blob)) is ProtocolPickyError
    assert type(replaced) is RuntimeError


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


def test_a_huge_exception_travels_bounded_as_its_description() -> None:
    failed = failure(3, ValueError("x" * (2 << 20)))

    assert (
        len(pickle.dumps(failed, protocol=5))
        < protocol.FAILURE_LIMIT + protocol.DESCRIPTION_LIMIT * 2
    )
    error = error_of(failed)
    assert type(error) is RuntimeError
    assert "ValueError" in str(error)
    assert "too large to send whole" in str(error)
