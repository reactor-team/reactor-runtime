"""Shared machinery for the external-consumer contract suite.

The tests under ``tests/contract/`` lock the runtime's observable surface — the
``/events`` journal, the session-lifecycle routes and their status codes, the
transport signalling paths, and the recording/upload mirrors — exactly as an
external consumer (a director driving the runtime from outside) reads it.

That surface is a contract, not an implementation detail: a failing test in
this suite means a breaking change for every consumer built on it, not a test
to update. Assertions here use hard-coded wire literals on purpose — never the
runtime's own enums or constants — so a rename or a shape change fails loudly
instead of silently following along.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from reactor_runtime import Input, InputField, Output, ReactorModel, Video, event
from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    InputFrame,
    MediaBundle,
    MediaChunk,
    RuntimeConfig,
    TransitionEvent,
)
from reactor_runtime.http.events import format_sse
from reactor_runtime.http.server import build_app
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.runner.runner import Runner
from reactor_runtime.transport.webrtc import (
    PeerStats,
    SdpAnswer,
    SdpOffer,
    TrackMap,
    WebRtcConfig,
    WebRtcPeerFactory,
    WebRtcRouter,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate

# The one transport session id a runtime process serves. Spelled out as a
# literal: consumers hard-code it in their route paths.
FIXED_SESSION_ID = "00000000-0000-0000-0000-000000000000"

# The exact key set of the single envelope type `/events` carries.
ENVELOPE_KEYS = frozenset({"type", "event", "from", "to", "ts", "detail"})


# -- fixture models -----------------------------------------------------------


class ContractOutput(Output):
    main: Video


class ContractInput(Input):
    webcam: Video


class ContractModel(ReactorModel):
    """A bidirectional fixture model: one outbound and one inbound track."""

    output: ContractOutput
    camera: ContractInput

    @event(name="set_mode")
    async def set_mode(self, mode: str = InputField(min_length=1)) -> None: ...

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        await asyncio.sleep(60)


class CrashingModel(ReactorModel):
    """A model whose run loop crashes as soon as it starts."""

    output: ContractOutput

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        raise RuntimeError("model exploded")


class UnloadableModel(ReactorModel):
    """A model that fails to load, terminating the process's session."""

    output: ContractOutput

    def load(self, config_path: Path | None) -> None:
        raise RuntimeError("weights missing")

    async def run(self) -> None: ...


# -- SSE wire parsing ---------------------------------------------------------


@dataclass(frozen=True)
class SseFrame:
    """One parsed SSE message: its sequence id and the decoded JSON payload."""

    seq: int
    payload: dict[str, Any]


def parse_sse_frame(raw: str) -> SseFrame:
    """Parse one SSE message, asserting the locked framing along the way.

    A frame is exactly two fields — ``id: <seq>`` then ``data: <json>`` — and
    nothing else: no ``event:`` field, no retry hints. Consumers parse only
    these two field names, so the framing itself is part of the contract.
    """
    lines = raw.rstrip("\n").split("\n")
    assert len(lines) == 2, f"an SSE frame is exactly an id line and a data line, got {lines!r}"
    assert lines[0].startswith("id: "), f"the first SSE field is 'id: ', got {lines[0]!r}"
    assert lines[1].startswith("data: "), f"the second SSE field is 'data: ', got {lines[1]!r}"
    return SseFrame(seq=int(lines[0][len("id: ") :]), payload=json.loads(lines[1][len("data: ") :]))


def envelope(seq: int, transition_event: TransitionEvent) -> dict[str, Any]:
    """Serialize a journal event through the real SSE path and decode it back.

    Runs the event through ``format_sse`` — the one place the wire shape is
    decided — so every assertion in the suite sees exactly the bytes a consumer
    sees, including JSON's rendering of enums and ids.
    """
    frame = parse_sse_frame(format_sse(seq, transition_event))
    assert frame.seq == seq
    return frame.payload


async def read_sse(app: FastAPI, target: str, count: int) -> list[SseFrame]:
    """Read *count* SSE frames from *target* through the assembled ASGI app.

    Invokes the app at the ASGI boundary so the stream is consumed incrementally
    (the journal never ends on its own), asserting the response is a ``200``
    ``text/event-stream`` before any frame is parsed.
    """
    path, _, query = target.partition("?")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(b"host", b"runtime"), (b"accept", b"text/event-stream")],
        "client": ("127.0.0.1", 40000),
        "server": ("runtime", 80),
    }
    inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    disconnected = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        await inbox.put(dict(message))

    task = asyncio.create_task(app(scope, receive, send))
    frames: list[SseFrame] = []
    status: int | None = None
    headers: dict[bytes, bytes] = {}

    async def collect() -> None:
        nonlocal status
        buffer = ""
        while len(frames) < count:
            message = await inbox.get()
            if message["type"] == "http.response.start":
                status = message["status"]
                headers.update(dict(message["headers"]))
            elif message["type"] == "http.response.body":
                buffer += message.get("body", b"").decode()
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    frames.append(parse_sse_frame(raw))

    try:
        async with asyncio.timeout(5.0):
            await collect()
    finally:
        disconnected.set()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
    assert status == 200
    assert headers.get(b"content-type", b"").startswith(b"text/event-stream")
    return frames


class JournalReader:
    """Follow a runner's egress journal as the serialized wire envelopes."""

    def __init__(self, runner: Runner, since: int = 0) -> None:
        self._events = cast(
            AsyncGenerator[tuple[int, TransitionEvent], None],
            runner.events.subscribe(since),
        )

    async def next(self) -> dict[str, Any]:
        async with asyncio.timeout(3.0):
            seq, transition_event = await anext(self._events)
        return envelope(seq, transition_event)

    async def expect(self, event_name: str) -> dict[str, Any]:
        """Return the next envelope whose ``event`` is *event_name*, skipping others."""
        async with asyncio.timeout(5.0):
            while True:
                seq, transition_event = await anext(self._events)
                payload = envelope(seq, transition_event)
                if payload["event"] == event_name:
                    return payload

    async def aclose(self) -> None:
        await self._events.aclose()


async def wait_for_state(runner: Runner, state_name: str) -> None:
    """Poll the journal's snapshot until the session reaches *state_name*."""
    async with asyncio.timeout(3.0):
        while True:
            snapshot = runner.events.snapshot()
            if snapshot.state is not None and snapshot.state.name.lower() == state_name:
                return
            await asyncio.sleep(0.01)


# -- transport doubles --------------------------------------------------------


class FakePeer:
    """A WebRtcPeer double that lets tests fire the wire-level events."""

    def __init__(self) -> None:
        self.ice: list[IceCandidate] = []
        self.protocol_version = ProtocolVersion.V0
        self._on_message: Callable[[bytes | str, ProtocolVersion, Channel], None] | None = None
        self._on_media: Callable[[str, InputFrame], None] | None = None
        self._on_ping: Callable[[], None] | None = None
        self._on_connected: Callable[[], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None

    async def add_ice(self, candidate: IceCandidate) -> None:
        self.ice.append(candidate)

    def send_message(self, payload: bytes | str) -> None: ...

    def send_control(self, payload: bytes | str) -> None: ...

    def send_media(self, bundle: MediaBundle) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def stats(self) -> PeerStats:
        return PeerStats(rtt_seconds=0.1)

    async def close(self) -> None: ...

    def on_message(self, callback: Callable[[bytes | str, ProtocolVersion, Channel], None]) -> None:
        self._on_message = callback

    def on_media(self, callback: Callable[[str, InputFrame], None]) -> None:
        self._on_media = callback

    def on_ping(self, callback: Callable[[], None]) -> None:
        self._on_ping = callback

    def on_connected(self, callback: Callable[[], None]) -> None:
        self._on_connected = callback

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        self._on_disconnect = callback

    def fire_connected(self) -> None:
        assert self._on_connected is not None
        self._on_connected()

    def fire_disconnect(self) -> None:
        assert self._on_disconnect is not None
        self._on_disconnect()


def peer_factory(peer: FakePeer, answer: str = "answer-sdp") -> WebRtcPeerFactory:
    """Build a factory that negotiates instantly with *peer* and a fixed answer."""

    async def factory(
        conn_id: int,
        offer: SdpOffer,
        tracks: TrackMap,
        config: WebRtcConfig,
        version: ProtocolVersion,
    ) -> tuple[FakePeer, SdpAnswer]:
        peer.protocol_version = version
        return peer, SdpAnswer(answer)

    return factory


class FakeConnection:
    """A neutral Connection double for driving the runner's sink face directly."""

    def __init__(self, conn_id: int) -> None:
        self.id = ConnId(conn_id)
        self.capabilities = ConnectionCapabilities(carries_video=False, carries_audio=False)

    @property
    def protocol_version(self) -> ProtocolVersion:
        return ProtocolVersion.V0

    def send_message(self, payload: bytes | str) -> None: ...

    def send_media(self, chunk: MediaChunk) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    def send_control(self, payload: bytes | str) -> None: ...

    async def close(self) -> None: ...


# -- the assembled runtime under test -----------------------------------------


@dataclass
class Harness:
    """A started runtime and the black-box handles the suite drives it through."""

    client: httpx.AsyncClient
    app: FastAPI
    runner: Runner
    peer: FakePeer


@asynccontextmanager
async def running_runtime(
    model_cls: type[ReactorModel] | None = None,
    cfg: RuntimeConfig | None = None,
    *,
    start: bool = True,
) -> AsyncIterator[Harness]:
    """Assemble the full HTTP surface over a real runner and fixture model.

    The app is built by the production assembly point (``build_app``) with the
    WebRTC router mounted over an instant fake peer factory, so every request in
    the suite crosses the same surface a real consumer calls.
    """
    resolved = model_cls or ContractModel
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: resolved)
        runner = Runner(cfg or RuntimeConfig(model_ref="contract:Model"))
        if start:
            await runner.start()
        peer = FakePeer()
        router = WebRtcRouter(WebRtcConfig(ping_timeout=0.0), peer_factory(peer))
        app = build_app(runner, [router], runner.health)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                yield Harness(client=client, app=app, runner=runner, peer=peer)
        finally:
            await runner.stop()
