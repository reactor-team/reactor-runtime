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

from reactor_runtime.distributed.rank import portable
from reactor_runtime.model_server.errors import WorkerError

PROTOCOL = 1
"""The version of this exchange; it moves with any change to a message."""

FRAME_HEADER = struct.Struct("!Q")
"""A frame's header: the length of the pickled message that follows."""


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
        signal: An instance of one of the model's declared signal types.
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
        description: ``"module.Type: message"`` of the exception, for when it
            cannot be rebuilt.
    """

    request_id: int
    error: bytes
    description: str


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
    """Return the :class:`Failed` that carries *exc* as the answer to *request_id*.

    An exception that does not pickle travels as a ``RuntimeError`` naming its
    type and message, as a distributed rank's does.
    """
    kind = type(exc)
    description = f"{kind.__module__}.{kind.__qualname__}: {_text(exc)}"
    carried = portable(exc) if isinstance(exc, Exception) else RuntimeError(description)
    return Failed(request_id, pickle.dumps(carried, protocol=5), description)


def error_of(failed: Failed) -> BaseException:
    """Return the exception *failed* carries, or a :class:`WorkerError` if it cannot be rebuilt."""
    try:
        error = pickle.loads(failed.error)
    except Exception:
        return WorkerError(failed.description)
    if not isinstance(error, BaseException):
        return WorkerError(failed.description)
    return error


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
    """Return *message* if it is one of this protocol's messages.

    Raises:
        ProtocolError: It is not.
    """
    if not isinstance(message, _MESSAGES):
        raise ProtocolError(f"a frame held a {type(message).__name__}, not a protocol message")
    return message


def send_frame(sock: socket.socket, data: bytes) -> None:
    """Write *data* as one frame. The caller serializes the writers of one socket."""
    sock.sendall(FRAME_HEADER.pack(len(data)) + data)


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


def _text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return "<unprintable>"
