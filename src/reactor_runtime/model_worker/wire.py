"""The frames a session process and the model worker exchange on their socket.

Every frame is a fixed header, ``!IBQ`` (payload length, kind, request id),
followed by the payload. The socket carries control and small headers only: a
step's arrays sit in shared memory, and the payload of a step or a result is the
:func:`~reactor_runtime.distributed.ipc.pack` header that names them.

A reader refuses a frame whose length passes the limit it was given before it
reads the payload, so a corrupt or hostile length cannot make it allocate
without bound. Anything else wrong with a frame is the receiver's to judge.
"""

from __future__ import annotations

import enum
import json
import socket
import struct
from typing import Any

HEADER = struct.Struct("!IBQ")
TIMEOUT = struct.Struct("!d")

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


def recv(sock: socket.socket, limit: int) -> tuple[Kind, int, bytes]:
    """Read one frame.

    Args:
        sock: The socket to read.
        limit: The largest payload accepted, in bytes.

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
    if length > limit:
        raise ProtocolError(f"a {frame_kind.name} frame of {length} bytes passes the {limit} limit")
    return frame_kind, request_id, _exactly(sock, length) if length else b""


def to_json(value: dict[str, Any]) -> bytes:
    """Encode a control payload."""
    return json.dumps(value, separators=(",", ":")).encode()


def from_json(payload: bytes) -> dict[str, Any]:
    """Decode a control payload, refusing anything but a JSON object.

    Raises:
        ProtocolError: The payload is not a JSON object.
    """
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"control payload is not JSON: {exc}") from None
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
