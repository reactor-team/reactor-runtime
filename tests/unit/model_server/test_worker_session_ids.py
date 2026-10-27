"""A session's reset names its session; the worker stamps it on its lines about the slot."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import log
from reactor_runtime.model_server import ModelServer, ServerRequest, wire


class Seen(ModelServer):
    """Echo each request's session id back, and keep the requests' ids."""

    max_batch = 4

    def load(self, config_path: Path | None) -> None:
        self.ids: list[str | None] = []

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        self.ids.extend(r.session_id for r in requests)
        return [{"session_id": r.session_id or ""} for r in requests]


@pytest.fixture
def json_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    monkeypatch.setenv("REACTOR_LOG_FORMAT", "json")
    buffer = io.StringIO()
    log.configure(level=logging.INFO, stream=buffer)
    yield buffer
    log.clear_session_id()


def _lines(buffer: io.StringIO, message: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    return [r for r in records if r["msg"] == message]


def test_a_reset_carries_the_live_session_id_to_the_worker(
    serve: Any, json_log: io.StringIO
) -> None:
    worker = Seen()
    served: Any = serve(worker)
    client = served.client(1, reset=False)
    log.set_session_id("sess-a")
    client.reset()
    assert client.step(np.zeros(2))["session_id"] == "sess-a"
    log.set_session_id("sess-b")
    client.reset()
    assert client.step(np.zeros(2))["session_id"] == "sess-b"
    assert worker.ids == ["sess-a", "sess-b"]
    resets = _lines(json_log, "session reset")
    assert [(r["slot"], r["generation"], r["session_id"]) for r in resets] == [
        (1, 2, "sess-a"),
        (1, 3, "sess-b"),
    ]


def test_an_explicit_session_id_wins_over_the_live_one(serve: Any, json_log: io.StringIO) -> None:
    worker = Seen()
    served: Any = serve(worker)
    client = served.client(0, reset=False)
    log.set_session_id("ambient")
    client.reset(session_id="named")
    client.step(np.zeros(1))
    assert worker.ids == ["named"]


def test_a_reset_without_a_session_stamps_no_id(serve: Any, json_log: io.StringIO) -> None:
    worker = Seen()
    served: Any = serve(worker)
    client = served.client(0)
    client.step(np.zeros(1))
    assert worker.ids == [None]
    (reset,) = _lines(json_log, "session reset")
    assert "session_id" not in reset


def test_an_id_unfit_for_a_log_line_is_dropped_not_refused(
    serve: Any, json_log: io.StringIO
) -> None:
    worker = Seen()
    served: Any = serve(worker)
    client = served.client(0, reset=False)
    client.reset(session_id="bad id\nwith=fields")
    client.step(np.zeros(1))
    client.reset(session_id="x" * 129)
    client.step(np.zeros(1))
    assert worker.ids == [None, None]


def test_the_session_closed_line_names_the_session(serve: Any, json_log: io.StringIO) -> None:
    served: Any = serve(Seen())
    client = served.client(3, reset=False)
    client.reset(session_id="sess-c")
    client.close()
    served.server.stop()
    served.thread.join(timeout=10)
    (closed,) = _lines(json_log, "session closed")
    assert (closed["slot"], closed["session_id"]) == (3, "sess-c")


def test_the_reset_message_is_typed_and_strict() -> None:
    assert wire.decode_message(wire.encode_message(wire.Reset("s1")), wire.Reset) == wire.Reset(
        "s1"
    )
    assert wire.decode_message(b'{"session_id":null}', wire.Reset) == wire.Reset(None)
    with pytest.raises(wire.ProtocolError):
        wire.decode_message(b"", wire.Reset)
    with pytest.raises(wire.ProtocolError):
        wire.decode_message(b'{"session_id":"s","extra":1}', wire.Reset)
