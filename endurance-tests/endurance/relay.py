"""A TCP relay to the LiveKit server that can stop answering without closing.

The runtime reaches the room's signal connection through the relay. While the
relay is stalled it holds every connection open and forwards nothing, in
either direction, as a server that has stopped responding does: the runtime
sees no close and no error, only silence. Media does not pass through it.

The relay runs on threads of its own, so it keeps forwarding while a scenario
waits on the runtime's blocking HTTP calls. It forwards raw TCP, so it serves a
local ``ws://`` server only: a ``wss://`` server's certificate would not match
the relay's address.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from urllib.parse import urlparse

_CHUNK = 65_536
# How often a stalled pump checks whether to forward again, or to stop.
_POLL_S = 0.05


class StallingRelay:
    """Forward TCP connections to one upstream address, until told to stall.

    Args:
        upstream_url: The LiveKit server's ``ws://`` URL.

    Raises:
        ValueError: The URL is not a ``ws://`` URL.
    """

    def __init__(self, upstream_url: str) -> None:
        parsed = urlparse(upstream_url)
        if parsed.scheme != "ws":
            raise ValueError(f"the relay forwards raw TCP to a ws:// server, not {upstream_url!r}")
        self._scheme = parsed.scheme
        self._upstream = (parsed.hostname or "127.0.0.1", parsed.port or 7880)
        self._flowing = threading.Event()
        self._flowing.set()
        self._closed = threading.Event()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._accept, name="relay-accept", daemon=True).start()

    @property
    def url(self) -> str:
        """The URL the runtime joins through, in place of the server's."""
        return f"{self._scheme}://127.0.0.1:{self._listener.getsockname()[1]}"

    def stall(self) -> None:
        """Stop forwarding; the connections stay open."""
        self._flowing.clear()

    def resume(self) -> None:
        """Forward again, starting with what arrived while stalled."""
        self._flowing.set()

    def close(self) -> None:
        """Close the listener and every relayed connection."""
        self._closed.set()
        self._flowing.set()
        self._listener.close()
        with self._lock:
            for sock in self._sockets:
                sock.close()
            self._sockets.clear()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            try:
                upstream = socket.create_connection(self._upstream)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._sockets += [client, upstream]
            # The pair is closed once both directions have ended.
            ends = threading.Barrier(2, action=lambda c=client, u=upstream: self._release(c, u))
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(
                    target=self._pump_then_end,
                    args=(src, dst, ends),
                    name="relay-pump",
                    daemon=True,
                ).start()

    def _release(self, *pair: socket.socket) -> None:
        with self._lock:
            for sock in pair:
                sock.close()
                if sock in self._sockets:
                    self._sockets.remove(sock)

    def open_sockets(self) -> int:
        """Return how many relayed sockets are still open."""
        with self._lock:
            return len(self._sockets)

    def _pump_then_end(
        self, src: socket.socket, dst: socket.socket, ends: threading.Barrier
    ) -> None:
        self._pump(src, dst)
        with contextlib.suppress(threading.BrokenBarrierError):
            ends.wait()

    def _pump(self, src: socket.socket, dst: socket.socket) -> None:
        # A socket the other side, or close(), already shut ends the pump quietly.
        with contextlib.suppress(OSError):
            while data := src.recv(_CHUNK):
                while not self._flowing.wait(_POLL_S):
                    if self._closed.is_set():
                        return
                dst.sendall(data)
        for sock in (src, dst):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
