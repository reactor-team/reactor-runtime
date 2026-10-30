"""The messages a session runtime and the model server exchange, and the frames they travel in.

Every message is a frozen dataclass, and its type is what it means, as in
:mod:`reactor_runtime.distributed.protocol`. A link pickles a message whole and
sends it as one length-prefixed frame; how a message's arrays travel (in shared
memory, or inline) is the link's to decide, so the messages carry the request
and the output as the objects themselves.

From a session to the worker: :class:`Hello` once, then :class:`Step`,
:class:`Cancel`, :class:`Signal` and :class:`Reset`. From the worker to a
session: :class:`Welcome` or :class:`Refused` once, then :class:`Result`,
:class:`Failed` and :class:`ResetDone`. A request is answered by exactly one
:class:`Result` or :class:`Failed` carrying its id, or by nothing once it was
cancelled. Both halves ship in one image, so they always speak the same
protocol; :data:`PROTOCOL` makes a mismatch fail at the hello, naming both
versions.
"""

from __future__ import annotations

import pickle
import socket
import struct
from dataclasses import dataclass
from typing import Any

from reactor_runtime.distributed.rank import portable_pickle
from reactor_runtime.model_server.errors import BatchFailed, WorkerError

PROTOCOL = 1
"""The version of this exchange; it moves with any change to a message."""

FRAME_HEADER = struct.Struct("!Q")
"""A frame's header: the length of the pickled message that follows."""

FAILURE_LIMIT = 64 << 10
"""The most bytes a failure's pickled exception takes; a larger one travels as its description."""

DESCRIPTION_LIMIT = 4096
"""The most characters of a failure's description."""


class ProtocolError(Exception):
    """A frame that cannot be read: its length passes the limit, or it holds no message."""


class Closed(Exception):  # noqa: N818 (named for the event a caller catches)
    """The peer closed the connection, at a frame boundary or inside one."""


# -- session to worker ----------------------------------------------------------------------


@dataclass(frozen=True)
class Hello:
    """A session's first message: the slot it serves, and the protocol it speaks.

    Attributes:
        slot: The session runtime's slot, ``0`` to the deployment's session
            count minus one.
        protocol: The :data:`PROTOCOL` the session speaks.
    """

    slot: int
    protocol: int = PROTOCOL


@dataclass(frozen=True)
class Step:
    """One request for the model's next step.

    Attributes:
        request_id: Names the answer, which carries the same id.
        timeout: Seconds the request may wait in the worker's queue before it is
            answered as expired, or ``None`` for no bound.
        payload: The request, an instance of the model's request type.
    """

    request_id: int
    timeout: float | None
    payload: Any


@dataclass(frozen=True)
class Cancel:
    """Withdraw a request the session stopped waiting for; it gets no answer."""

    request_id: int


@dataclass(frozen=True)
class Signal:
    """An out-of-band signal for the model, such as a request to stop early.

    Attributes:
        signal: What the application passed to ``ModelClient.signal()``.
    """

    signal: Any


@dataclass(frozen=True)
class Reset:
    """Close the slot's session on the worker and open a new one.

    Attributes:
        request_id: Names the answer.
        session_id: The id of the session that starts, which the worker puts on
            its log lines about the slot until the next reset, and uses for
            nothing else; ``None`` when the session runtime has none.
    """

    request_id: int
    session_id: str | None


# -- worker to session ----------------------------------------------------------------------


@dataclass(frozen=True)
class Welcome:
    """The worker opened the session: what it serves the session with.

    Attributes:
        slot: The slot the session holds.
        generation: How many times the slot has been opened, from ``1``.
        max_batch: The most requests one step receives.
    """

    slot: int
    generation: int
    max_batch: int


@dataclass(frozen=True)
class Refused:
    """The worker refused the session, and why; it closes the connection after this."""

    reason: str


@dataclass(frozen=True)
class Result:
    """A request's output, an instance of the model's result type."""

    request_id: int
    output: Any


@dataclass(frozen=True)
class Failed:
    """A request that did not produce an output, and the exception that says why.

    Build one with :func:`failure` and read it with :func:`error_of`.

    Attributes:
        request_id: The request it answers.
        error: The exception, pickled on its own so that a class this process
            cannot import does not fail the whole message.
        description: ``"module.Type: message"`` of the exception, or of its
            cause for a batch failure, for when it cannot be rebuilt.
        batch: Whether the exception is a :class:`BatchFailed`, so that it is
            rebuilt as one even when its cause cannot be.
    """

    request_id: int
    error: bytes
    description: str
    batch: bool = False


@dataclass(frozen=True)
class ResetDone:
    """The worker reopened the slot.

    Attributes:
        request_id: The reset it answers.
        generation: The slot's new generation.
    """

    request_id: int
    generation: int


ToWorker = Hello | Step | Cancel | Signal | Reset
"""A message a session sends."""

ToSession = Welcome | Refused | Result | Failed | ResetDone
"""A message the worker sends."""

_MESSAGES = (Hello, Step, Cancel, Signal, Reset, Welcome, Refused, Result, Failed, ResetDone)


def failure(request_id: int, exc: BaseException) -> Failed:
    """Return the :class:`Failed` that carries *exc* as the answer to *request_id*. Never raises.

    The exception travels as itself when it pickles and its pickle fits
    :data:`FAILURE_LIMIT`. Otherwise a ``RuntimeError`` carrying its
    description stands in for it, so a failure always fits in a frame. A
    :class:`BatchFailed` always travels as one: when its cause cannot travel,
    only the cause is replaced. The description is ``"module.Type: message"``,
    cut at :data:`DESCRIPTION_LIMIT` characters, and an exception that cannot
    print is described by its type alone. A ``BaseException`` that is not an
    ``Exception`` (``CancelledError``, ``KeyboardInterrupt``, ``SystemExit``)
    always travels as a ``RuntimeError`` carrying its description: raised
    again on the session's side it would cancel the session's task or end its
    process.

    The model server answers with it on the thread that runs every step, so a
    raise here would stop every session's answers: anything unexpected while
    building it falls back to a fixed ``RuntimeError`` naming the type.
    """
    try:
        if isinstance(exc, BatchFailed):
            return _batch_failure(request_id, exc)
        description = _describe(exc)
        carried = exc if isinstance(exc, Exception) else RuntimeError(description)
        _, error = portable_pickle(carried, limit=FAILURE_LIMIT, description=description)
        return Failed(request_id, error, description)
    except Exception:
        kind = type(exc)
        description = f"{kind.__module__}.{kind.__qualname__}: <unprintable>"
        error = pickle.dumps(RuntimeError(description), protocol=5)
        return Failed(request_id, error, description, batch=isinstance(exc, BatchFailed))


def error_of(failed: Failed) -> BaseException:
    """Return the exception *failed* carries, or a :class:`WorkerError` if it cannot be rebuilt.

    A batch failure is always a :class:`BatchFailed`: when what arrives is
    not one (a stand-in the sender fell back to), it becomes the cause, and
    when it cannot be rebuilt at all (its cause's class is not in this
    process), the :class:`WorkerError` is.
    """
    try:
        error = pickle.loads(failed.error)
    except Exception:
        error = None
    if isinstance(error, BaseException):
        if failed.batch and not isinstance(error, BatchFailed):
            return BatchFailed(error)
        return error
    stand_in = WorkerError(failed.description)
    return BatchFailed(stand_in) if failed.batch else stand_in


def check_hello(hello: Hello) -> None:
    """Refuse a hello in another protocol, naming both versions.

    Raises:
        ProtocolError: The session speaks another protocol.
    """
    if hello.protocol != PROTOCOL:
        raise ProtocolError(
            f"the session speaks model server protocol {hello.protocol} and the worker speaks "
            f"{PROTOCOL}: run both halves from the same reactor-runtime"
        )


def check_message(message: object) -> Any:
    """Return *message* if it is one of this protocol's messages, by exact type.

    A subclass of a message is not that message: the dataclass is the message's
    meaning, and a protocol change moves :data:`PROTOCOL`.

    Raises:
        ProtocolError: It is not.
    """
    if type(message) not in _MESSAGES:
        raise ProtocolError(f"a frame held a {type(message).__name__}, not a protocol message")
    return message


def send_frame(sock: socket.socket, data: bytes) -> None:
    """Write *data* as one frame. The caller serializes the writers of one socket.

    The header and the body are written by two calls, so a large body is never
    copied to prepend its header.
    """
    sock.sendall(FRAME_HEADER.pack(len(data)))
    sock.sendall(data)


def recv_frame(sock: socket.socket, limit: int) -> bytes:
    """Read one frame, refusing one longer than *limit* before reading its body.

    Raises:
        Closed: The peer closed the connection.
        ProtocolError: The frame is longer than *limit*.
        OSError: The socket failed, a timeout included.
    """
    (length,) = FRAME_HEADER.unpack(_exactly(sock, FRAME_HEADER.size))
    if length > limit:
        raise ProtocolError(f"a frame of {length} bytes passes the {limit} byte limit")
    return _exactly(sock, length) if length else b""


def _exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        read = sock.recv_into(view[got:], n - got)
        if read == 0:
            raise Closed("the peer closed the connection")
        got += read
    return bytes(buf)


def _batch_failure(request_id: int, exc: BatchFailed) -> Failed:
    cause = exc.cause
    description = _describe(cause)
    if not isinstance(cause, Exception):
        # Raised again on the session's side, it would cancel the session's
        # task or end its process, as failure() says for a direct failure.
        exc = BatchFailed(RuntimeError(description))
    carried, error = portable_pickle(exc, limit=FAILURE_LIMIT, description=description)
    if carried is not exc:
        # The cause cannot travel as itself: keep the wrapper, replace the cause.
        stand_in, _ = portable_pickle(cause, limit=FAILURE_LIMIT, description=description)
        if stand_in is cause:
            stand_in = RuntimeError(description)
        _, error = portable_pickle(BatchFailed(stand_in), description=description)
    return Failed(request_id, error, description, batch=True)


def _describe(exc: BaseException) -> str:
    kind = type(exc)
    description = f"{kind.__module__}.{kind.__qualname__}: {_text(exc)}"
    if len(description) > DESCRIPTION_LIMIT:
        description = description[: DESCRIPTION_LIMIT - 3] + "..."
    return description


def _text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return "<unprintable>"
