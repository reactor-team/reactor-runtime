"""The batch example's bench script against a stand-in runtime.

The stand-in serves the four routes the bench reads: it starts two sessions
and refuses the rest, steps the first twice and the second not at all, and
saves each step's fingerprint only on the second ask, as a runtime whose
step saves land after the session ends does.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import bench
import pytest


class _Runtime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.starts: dict[str, threading.Event] = {}
        self.asked: dict[str, int] = {}
        self.closing = threading.Event()

    def start_of(self, sid: str) -> threading.Event:
        with self.lock:
            return self.starts.setdefault(sid, threading.Event())


def _handler(runtime: _Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None: ...

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            sid = body["session_id"]
            with runtime.lock:
                admitted = len(runtime.started) < 2
                if admitted:
                    runtime.started.append(sid)
            if not admitted:
                self._reply(409, b'{"detail": "all slots are serving"}')
                return
            runtime.start_of(sid).set()
            self._reply(200, json.dumps({"session_id": sid}).encode())

        def do_GET(self) -> None:
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts[-1] == "events":
                self._stream(parts[1])
            else:
                with runtime.lock:
                    runtime.asked[self.path] = runtime.asked.get(self.path, 0) + 1
                    saved = runtime.asked[self.path] > 1
                if saved:
                    self._reply(200, f"digest-{parts[3]}".encode())
                else:
                    self._reply(404, b'{"detail": "Step result not found"}')

        def _stream(self, sid: str) -> None:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            started = runtime.start_of(sid)
            while not started.wait(0.05):
                if runtime.closing.is_set():
                    return
            steps = 2 if runtime.started.index(sid) == 0 else 0
            events = [("step_completed", 1000 + 500 * n) for n in range(steps)]
            for name, ts in [*events, ("cleanup_complete", 5000)]:
                payload = json.dumps({"event": name, "ts": ts})
                self.wfile.write(f"id: 1\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture
def runtime() -> Iterator[tuple[str, _Runtime]]:
    state = _Runtime()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        state.closing.set()
        server.shutdown()
        server.server_close()


def test_the_bench_reports_a_refused_start_and_a_stepless_session_and_waits_for_saves(
    runtime: tuple[str, _Runtime],
) -> None:
    base, _ = runtime
    report: dict[str, Any] = {}
    done = threading.Thread(
        target=lambda: report.update(bench.run(base, 3, 2, digests=True)), daemon=True
    )

    done.start()
    done.join(timeout=20)

    assert not done.is_alive(), "the bench waited on a session that never started"
    stepped, stepless, refused = report["sessions"]
    assert stepped["steps"] == 2
    assert stepped["steps_per_s"] == pytest.approx(2.0)
    assert stepped["digests"] == ["digest-1", "digest-2"]
    assert stepless["steps"] == 0
    assert stepless["steps_per_s"] is None
    assert refused["error"].startswith("not started: 409")
    assert report["aggregate_steps_per_s"] == pytest.approx(4.0)
