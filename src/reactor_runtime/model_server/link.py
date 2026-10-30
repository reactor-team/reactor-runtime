"""How the model server's messages travel: one link per session, and the transport that makes it.

The protocol (:mod:`reactor_runtime.model_server.protocol`) says what the two
halves say; a link says how the bytes move. A :class:`WorkerLink` sends and
receives whole protocol messages, and decides itself where a message's arrays
go: inline in the frame, or in memory both ends share. A :class:`LinkTransport`
opens links: ``connect()`` on a session's side, ``listen()`` and ``accept()`` on
the model server's. Whatever a link needs to set itself up (shared blocks, a
probe that the two ends see the same memory) it exchanges itself, before the
protocol's first message, so the protocol never carries it.

All three are protocols a class conforms to by shape, as the runtime's other
seams are. :class:`StreamLink` is the link over one connected stream socket
with every array inline, which the link over shared memory builds on.

Both links in this package unpickle what the peer sends, and unpickling can
run code, so they are for a trusted peer only: the other half of the same
model, from the same image, over a channel only it can open (a Unix socket
open to its user alone). They are not for use over a network as they are. A
link that crosses a network must authenticate its peer as that same model
(for example with mutual TLS) before any byte is unpickled, or replace the
encoding of the protocol itself, not only of the link: a failure carries a
pickled exception, and a request, a result and a signal are whatever objects
the application and the model exchange.
"""

from __future__ import annotations

import contextlib
import pickle
import socket
import threading
from typing import Any, Protocol, runtime_checkable

from reactor_runtime.model_server.errors import PayloadTooLarge
from reactor_runtime.model_server.protocol import (
    Closed,
    ProtocolError,
    check_message,
    recv_frame,
    send_frame,
)

DEFAULT_FRAME_LIMIT = 1 << 20
"""The longest frame a :class:`StreamLink` sends or accepts when its owner names no limit: 1 MiB."""


class NoListener(Exception):  # noqa: N818 (named for the state a caller catches)
    """No link yet; the model server may still be starting.

    Nothing accepts at the address, or the model server did not finish
    setting the link up within the connect's timeout. Trying again later may
    work.
    """


@runtime_checkable
class WorkerLink(Protocol):
    """One connection between a session runtime and the model server, on either side.

    ``send()`` and ``receive()`` carry whole protocol messages. ``receive()``
    blocks for the next message on one thread while ``send()`` runs on others;
    ``close()`` from any thread ends a blocked ``receive()`` with
    :class:`~reactor_runtime.model_server.protocol.Closed`.
    """

    def send(self, message: Any) -> None:
        """Send one protocol message whole.

        Only :class:`~reactor_runtime.model_server.protocol.Closed` and
        ``OSError`` mean the link is gone. Any other exception, for a message
        the link cannot encode (an object that does not pickle), is raised
        before anything is sent, and the link goes on.

        Raises:
            PayloadTooLarge: The message is larger than the link carries.
                Nothing was sent, and the link goes on.
            Closed: The link is closed, a close racing this send included.
            OSError: The connection failed.
        """
        ...

    def receive(self) -> Any:
        """Block for the next protocol message and return it.

        Raises:
            Closed: The peer or this side closed the link.
            ProtocolError: The peer sent something that is not a message.
            OSError: The connection failed.
        """
        ...

    def close(self) -> None:
        """Close the link and release what it holds. Idempotent."""
        ...


@runtime_checkable
class LinkListener(Protocol):
    """The model server's end of a transport: the links sessions open."""

    def accept(self) -> WorkerLink:
        """Block for the next session's link and return it.

        Raises:
            OSError: The listener is closed.
        """
        ...

    def close(self) -> None:
        """Stop accepting. Links already accepted stay open."""
        ...


@runtime_checkable
class LinkTransport(Protocol):
    """Opens links between session runtimes and the model server."""

    def listen(self, address: str) -> LinkListener:
        """Start accepting links at *address*, for the model server."""
        ...

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> WorkerLink:
        """Open a link to the model server at *address* for the session in *slot*.

        An implementation blocks no longer than *timeout* seconds in total,
        including whatever setup it exchanges with the model server before it
        returns the link; ``None`` is no bound.

        Raises:
            NoListener: Nothing accepts there yet, or the model server did not
                finish setting the link up within *timeout*; trying again
                later may work.
            WorkerUnreachable: This deployment cannot reach a model server at
                all; trying again will not help.
        """
        ...


class StreamLink:
    """A link over one connected stream socket, each message pickled whole, arrays inline.

    For a trusted peer only: :meth:`receive` unpickles what the peer sends
    (see the module docstring).

    Args:
        sock: The connected socket; the link owns it.
        frame_limit: The longest frame the link sends or accepts. The two ends
            of a link must use the same limit, which they do when one
            deployment builds both from the same setting: a frame this end
            refuses to send is never half-written, and a frame the peer would
            refuse is never sent.
    """

    def __init__(self, sock: socket.socket, *, frame_limit: int = DEFAULT_FRAME_LIMIT) -> None:
        self._sock = sock
        self._frame_limit = frame_limit
        self._send_lock = threading.Lock()
        self._closed = False

    @property
    def frame_limit(self) -> int:
        """The longest frame this link sends or accepts."""
        return self._frame_limit

    def send(self, message: Any) -> None:
        """Send *message* as one frame. Safe from any thread.

        Raises:
            PayloadTooLarge: The pickled message is longer than the frame
                limit; nothing was sent.
            Closed: The link is closed.
            OSError: The connection failed.
            Exception: Whatever pickling raises for a message that does not
                pickle; nothing was sent.
        """
        self.send_bytes(pickle.dumps(check_message(message), protocol=5))

    def receive(self) -> Any:
        """Block for the next message and return it.

        Raises:
            Closed: The peer or this side closed the link.
            ProtocolError: The frame passes the limit, cannot be unpickled, or
                holds no message.
            OSError: The connection failed.
        """
        data = self.receive_bytes()
        try:
            message = pickle.loads(data)
        except Exception as exc:
            raise ProtocolError(f"a frame does not hold a message: {exc!r}") from None
        return check_message(message)

    def send_bytes(self, data: bytes) -> None:
        """Send *data* as one frame, for a link that encodes messages its own way.

        Raises:
            PayloadTooLarge: *data* is longer than the frame limit; nothing
                was sent.
            Closed: The link is closed, before or during the send.
            OSError: The connection failed.
        """
        if len(data) > self._frame_limit:
            raise PayloadTooLarge(
                f"a message's frame holds {len(data)} bytes, above the link's "
                f"{self._frame_limit}-byte frame limit"
            )
        if self._closed:
            raise Closed("the link is closed")
        with self._send_lock:
            try:
                send_frame(self._sock, data)
            except OSError:
                if self._closed:
                    raise Closed("the link is closed") from None
                raise

    def receive_bytes(self, *, deadline: float | None = None) -> bytes:
        """Block for the next frame and return its bytes.

        Args:
            deadline: A ``time.monotonic()`` instant by which the whole frame
                is read; ``None`` is no bound.

        Raises:
            Closed: The peer or this side closed the link.
            ProtocolError: The frame passes the limit.
            TimeoutError: *deadline* passed before the frame was read.
            OSError: The connection failed.
        """
        try:
            return recv_frame(self._sock, self._frame_limit, deadline=deadline)
        except OSError:
            if self._closed:
                raise Closed("the link is closed") from None
            raise

    def close(self) -> None:
        """Close the socket; a :meth:`receive` blocked on another thread ends with ``Closed``."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
