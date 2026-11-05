"""Keep the stalling relay's connection bookkeeping covered without a LiveKit server."""

import socket
import threading
import time

import pytest
from endurance.relay import StallingRelay


def _echo_server() -> tuple[socket.socket, int]:
    server = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                while data := conn.recv(1024):
                    conn.sendall(data)

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def test_a_relayed_pair_is_closed_once_both_directions_end() -> None:
    server, port = _echo_server()
    relay = StallingRelay(f"ws://127.0.0.1:{port}")
    try:
        for _ in range(3):
            with socket.create_connection(
                ("127.0.0.1", int(relay.url.rsplit(":", 1)[1]))
            ) as client:
                client.sendall(b"ping")
                assert client.recv(4) == b"ping"
        deadline = time.monotonic() + 5
        while relay.open_sockets() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert relay.open_sockets() == 0
    finally:
        relay.close()
        server.close()


def test_a_secure_url_is_refused() -> None:
    with pytest.raises(ValueError, match="ws://"):
        StallingRelay("wss://rooms.example")
