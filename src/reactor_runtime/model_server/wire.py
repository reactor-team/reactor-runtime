"""The frames a session process and the model worker exchange on their socket.

Every frame is a fixed header, ``!IBQ`` (payload length, kind, request id),
followed by the payload. The socket carries control and small headers only: a
step's arrays sit in shared memory, and the payload of a step or a result is the
:func:`~reactor_runtime.distributed.ipc.pack` header that names them.

A reader refuses a frame whose length passes the limit it was given before it
reads the payload, so a corrupt or hostile length cannot make it allocate
without bound. Anything else wrong with a frame is the receiver's to judge.

Why dataclasses and not protobuf: ``reactor_wire`` is the public client
contract, so a change there is a public release and a pin bump; a private
proto would need codegen in the build, as generated bindings are not
committed and the buf hooks depend on remote plugins; both halves always ship
in one image, so no cross-version compatibility is needed; and strict decoding,
:data:`PROTOCOL`, and the schema fingerprint fail loudly where proto3 reads a
missing field as zero. The field names mirror a proto sketch, so a move is
mechanical should a cross-host transport ever need one.

The control payloads are typed: :class:`Hello`, :class:`Welcome`,
:class:`Failure`, :class:`Reset`, :class:`ResetAck`, and :class:`Refusal`, frozen dataclasses
that travel as JSON objects of their fields and are decoded strictly by
:func:`decode` (a missing, an extra, or a wrongly typed field is a
:class:`ProtocolError`). :data:`PROTOCOL` is the version of this whole
exchange; a session sends it in its :class:`Hello` and a worker of another
version refuses the session, naming both.

Which frame carries which payload:

* ``HELLO``: a :class:`Hello`. ``WELCOME``: a :class:`Welcome`.
* ``STEP``: :data:`TIMEOUT` then a :mod:`~reactor_runtime.model_server.codec`
  header. ``RESULT``: a codec header.
* ``SIGNAL``: :data:`SIGNAL_TAG` (the index of the signal's type in the
  worker's declared ``signals``) then the codec tree of its fields, as JSON.
* ``RESET``: a :class:`Reset`. ``ACK``: a :class:`ResetAck`.
* ``CANCEL``, ``EXPIRED``: empty.
* ``FAILED``: a :class:`Failure`. ``REJECTED`` and ``REFUSED``: a
  :class:`Refusal`.
"""

from __future__ import annotations

import enum
import json
import socket
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from reactor_runtime.model_server import schema

HEADER = struct.Struct("!IBQ")
TIMEOUT = struct.Struct("!d")
SIGNAL_TAG = struct.Struct("!I")

# The version of the exchange this module defines. A change to a frame kind, a
# control message, or the codec's header moves it on.
PROTOCOL = 3

# The ceiling on a frame that carries no step data: a hello, a signal, an error.
CONTROL_LIMIT = 1 << 20


class Kind(enum.IntEnum):
    """What a frame is. Values below 10 travel to the worker, the rest back."""

    HELLO = 1
    STEP = 2
    SIGNAL = 3
    RESET = 4
    CANCEL = 5
    WELCOME = 10
    RESULT = 11
    FAILED = 12
    REJECTED = 13
    EXPIRED = 14
    REFUSED = 15
    ACK = 16


class ProtocolError(Exception):
    """A frame that cannot be read: an unknown kind or a length past the limit."""


class Closed(Exception):  # noqa: N818 (named for the event a caller catches)
    """The peer closed the socket."""


def encode(kind: Kind, request_id: int, payload: bytes = b"") -> bytes:
    """Return one frame ready for ``sendall``."""
    return HEADER.pack(len(payload), kind, request_id) + payload


def send(sock: socket.socket, kind: Kind, request_id: int, payload: bytes = b"") -> None:
    """Write one frame. The caller serializes writers on one socket."""
    sock.sendall(encode(kind, request_id, payload))


def recv(sock: socket.socket, limit: int | Callable[[Kind], int]) -> tuple[Kind, int, bytes]:
    """Read one frame.

    Args:
        sock: The socket to read.
        limit: The largest payload accepted, in bytes, or a function of the
            frame's kind returning it; checked before the payload is read.

    Returns:
        The frame's kind, request id, and payload.

    Raises:
        Closed: The peer closed the socket, at a frame boundary or inside one.
        ProtocolError: The kind is unknown or the length passes *limit*.
        OSError: The socket failed, a timeout included.
    """
    length, kind, request_id = HEADER.unpack(_exactly(sock, HEADER.size))
    try:
        frame_kind = Kind(kind)
    except ValueError:
        raise ProtocolError(f"unknown frame kind {kind}") from None
    most = limit if isinstance(limit, int) else limit(frame_kind)
    if length > most:
        raise ProtocolError(f"a {frame_kind.name} frame of {length} bytes passes the {most} limit")
    return frame_kind, request_id, _exactly(sock, length) if length else b""


@dataclass(frozen=True)
class Hello:
    """A session's first frame: who it is and which contract it speaks.

    Attributes:
        slot: The slot asked for, or ``None`` for the lowest free one.
        pid: The session process's pid, for the worker's logs.
        token: A random hex token naming this connection's shared blocks.
        protocol: The :data:`PROTOCOL` the session speaks.
        schema: The fingerprint of the session's request, result, and signal
            types (:func:`~reactor_runtime.model_server.schema.fingerprint`).
    """

    slot: int | None
    pid: int
    token: str
    protocol: int
    schema: str


@dataclass(frozen=True)
class Welcome:
    """The worker's answer once it has opened the session.

    Attributes:
        slot: The slot the session holds.
        generation: The slot's count of openings.
        max_batch: The effective ``max_batch``.
        max_request_bytes: The effective ``max_request_bytes``.
        shm_slot_bytes: The ``/dev/shm`` this session holds.
        signals: The names of the worker's declared signal types, in the
            order their tags index.
        request_prefix: The prefix the session names its request blocks under.
        transport: What the transport adds for the session's side, such as
            the block that proves the two share ``/dev/shm``.
    """

    slot: int
    generation: int
    max_batch: int
    max_request_bytes: int
    shm_slot_bytes: int
    signals: tuple[str, ...]
    request_prefix: str
    transport: dict[str, str]


@dataclass(frozen=True)
class Failure:
    """Why a request failed in the worker.

    Attributes:
        type: The qualified type of the worker's exception.
        message: Its message.
        traceback: Its traceback in the worker, as text.
        worker_lost: The worker is going away (it is stopping).
        batch_failed: ``step()`` raised for the whole batch this request was
            in; the cause is the batch's, not necessarily this request's.
    """

    type: str
    message: str
    traceback: str
    worker_lost: bool
    batch_failed: bool


@dataclass(frozen=True)
class Reset:
    """A session asks the worker to reopen its slot for a new session.

    Attributes:
        session_id: The platform id of the session that starts, which the
            worker stamps on its lines about the slot until the next reset;
            ``None`` when the session runtime has none.
    """

    session_id: str | None


@dataclass(frozen=True)
class ResetAck:
    """The worker reopened the slot.

    Attributes:
        generation: The slot's new generation.
    """

    generation: int


@dataclass(frozen=True)
class Refusal:
    """Why the worker refused a session (``REFUSED``) or a request (``REJECTED``).

    Attributes:
        reason: What was refused and why, for the session's error.
    """

    reason: str


def encode_message(message: object) -> bytes:
    """Encode a control message as the JSON object of its fields.

    Raises:
        ProtocolError: The message does not match its declaration.
    """
    try:
        fields = schema.dump(message, type(message))
    except schema.SchemaError as exc:
        raise ProtocolError(f"a {type(message).__name__} does not encode: {exc}") from None
    return json.dumps(fields, separators=(",", ":")).encode()


def decode_message[M](payload: bytes, cls: type[M]) -> M:
    """Decode a control message of type *cls*, strictly.

    Raises:
        ProtocolError: The payload is not a JSON object of exactly *cls*'s
            fields, each of its declared type.
    """
    value = json_object(payload)
    try:
        return schema.load(value, cls)
    except schema.SchemaError as exc:
        raise ProtocolError(f"a malformed {cls.__name__}: {exc}") from None


def json_object(payload: bytes) -> dict[str, Any]:
    """Decode a JSON object, refusing anything else.

    Raises:
        ProtocolError: The payload is not a JSON object.
    """
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError(f"control payload is not JSON: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ProtocolError("control payload is not a JSON object")
    return value


def _exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        read = sock.recv_into(view[got:], n - got)
        if read == 0:
            raise Closed("the peer closed the socket")
        got += read
    return bytes(buf)
