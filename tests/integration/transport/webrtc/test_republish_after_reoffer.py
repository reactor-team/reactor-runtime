"""Publishing a track again after a client reconnects by re-offering.

A client whose wire dropped reconnects by re-offering on the same connection id,
and its published tracks do not survive that: the client publishes them again
on the new wire. These tests pin that the runtime grants that second publish,
over a real libwebrtc wire and the full runtime: the HTTP signalling routes, the
runner, and its connection registry. The client re-offers while its first peer
is still up, as a client that renegotiates without waiting for the old wire to
time out does.

Requires the native ``reactor_webrtc`` wheel; the module skips cleanly when it
is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pytest

rw = pytest.importorskip("reactor_webrtc")

from tests.integration.transport.webrtc._signalling import (  # noqa: E402
    TIMEOUT_S,
    negotiate,
    new_connection,
    serve,
    trickle,
)

from reactor_runtime import InputBuffer, MediaInput, ReactorApp, ReadMode, Video  # noqa: E402
from reactor_runtime.transport.webrtc import WebRtcConfig  # noqa: E402
from reactor_runtime.transport.webrtc.frames import rgb_to_bgra  # noqa: E402
from reactor_runtime.transport.webrtc.peer import _get_factory  # noqa: E402

pytestmark = pytest.mark.asyncio

_WIDTH, _HEIGHT = 320, 240


class _Webcam(MediaInput):
    webcam: Video


class _WebcamCounter:
    """The webcam frames the model has read, shared with the test across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames = 0

    def add(self, frames: int) -> None:
        with self._lock:
            self._frames += frames

    @property
    def frames(self) -> int:
        with self._lock:
            return self._frames


_seen = _WebcamCounter()


class _WebcamModel(ReactorApp):
    """Reads every webcam frame that reaches it, for thirty seconds."""

    media: _Webcam

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        webcam = cast(InputBuffer, self.media.webcam)
        for _ in range(30 * 30):
            frames = webcam.try_read(webcam.available, ReadMode.FIFO) or []
            _seen.add(len(frames))
            await asyncio.sleep(1 / 30)


@asynccontextmanager
async def _runtime() -> AsyncIterator[httpx.AsyncClient]:
    """Serve the full runtime over the webcam model, counting its frames afresh."""
    global _seen
    _seen = _WebcamCounter()
    async with serve(_WebcamModel, "republish:Model") as http:
        yield http


class _Peer:
    """One libwebrtc peer of the client: a ``webcam`` it sends and both channels."""

    def __init__(self) -> None:
        self.control_open = threading.Event()
        self.data_open = threading.Event()
        self.replies: list[dict[str, Any]] = []
        self.replied = threading.Event()
        self.ice: list[Any] = []

        factory = _get_factory(WebRtcConfig())
        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = self.ice.append
        # Dropped at teardown, which is what releases the native peer connection.
        self.pc: Any = factory.create_peer_connection(rw.RtcConfiguration(), observer)
        self.camera: Any = factory.create_video_track("webcam")
        self.send = self.pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
        self.data: Any = self._open("data", self.data_open)
        self.control: Any = self._open("control", self.control_open)
        self.control.on_message(self._on_control)

    def _open(self, label: str, opened: threading.Event) -> Any:
        channel = self.pc.create_data_channel(label)
        channel.on_state_change(
            lambda state: opened.set() if state == rw.DataChannelState.Open else None
        )
        if channel.state() == rw.DataChannelState.Open:
            opened.set()
        return channel

    def _on_control(self, data: bytes, _binary: bool) -> None:
        self.replies.append(json.loads(bytes(data)))
        self.replied.set()

    async def offer(self, http: httpx.AsyncClient, cid: int, *, method: str) -> None:
        """Offer the ``webcam`` on *cid*, apply the answer, and add its candidates."""
        await self.send.set_track(self.camera)
        mapping = [
            {
                "mid": self.send.mid() or "0",
                "name": "webcam",
                "kind": "video",
                "direction": "sendonly",
            }
        ]
        await negotiate(http, self.pc, cid, mapping, method=method)

    async def wire_open(self) -> None:
        for opened, label in ((self.control_open, "control"), (self.data_open, "data")):
            assert await asyncio.to_thread(opened.wait, TIMEOUT_S), (
                f"the {label} channel never opened"
            )

    async def publish(self, name: str, request_id: str) -> dict[str, Any]:
        """Send a ``publish_track`` request and return the runtime's reply to it."""
        self.replied.clear()
        self.control.send(
            json.dumps(
                {
                    "type": "request",
                    "method": "publish_track",
                    "request_id": request_id,
                    "data": {"name": name},
                }
            ).encode(),
            binary=False,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TIMEOUT_S
        while loop.time() < deadline:
            for reply in self.replies:
                if reply.get("request_id") == request_id:
                    return reply
            await asyncio.to_thread(self.replied.wait, 0.1)
            self.replied.clear()
        raise AssertionError(f"no reply to the publish request {request_id!r}")

    def push_frame(self, value: int) -> None:
        frame = np.full((_HEIGHT, _WIDTH, 3), value % 256, dtype=np.uint8)
        bgra, width, height = rgb_to_bgra(frame)
        self.camera.push_video_frame(bgra, width, height)

    def release(self) -> None:
        self.control = self.data = self.camera = None
        self.pc = None


@asynccontextmanager
async def _wire(http: httpx.AsyncClient, cid: int, *, method: str) -> AsyncIterator[_Peer]:
    """Negotiate one peer on *cid* and trickle its candidates while it is in use.

    The peer itself outlives the block, so a reconnect can re-offer while the
    wire it replaces is still up; the caller releases it.
    """
    peer = _Peer()
    stop = asyncio.Event()
    trickling: asyncio.Task[None] | None = None
    try:
        await peer.offer(http, cid, method=method)
        trickling = asyncio.create_task(trickle(http, cid, peer.ice, stop))
        yield peer
    finally:
        stop.set()
        if trickling is not None:
            trickling.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await trickling


async def _published_then_reoffered(
    http: httpx.AsyncClient, peers: list[_Peer]
) -> contextlib.AsyncExitStack:
    """Publish ``webcam`` on a first wire, then re-offer on the same id.

    Returns a stack holding the new wire open, whose peer is the last of
    *peers*. The first peer is left up, its wire replaced by the runtime.
    """
    cid = await new_connection(http)
    async with _wire(http, cid, method="POST") as first:
        peers.append(first)
        await first.wire_open()
        # Registration follows the data channel opening on the runtime's loop.
        await asyncio.sleep(0.5)
        reply = await first.publish("webcam", "first")
        assert "error" not in reply, f"the first publish was refused: {reply}"

    stack = contextlib.AsyncExitStack()
    second = await stack.enter_async_context(_wire(http, cid, method="PUT"))
    peers.append(second)
    return stack


async def _webcam_frames_arrive(peer: _Peer) -> bool:
    """Push webcam frames from *peer* until the model reads one it sent."""
    baseline = _seen.frames
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TIMEOUT_S
    value = 0
    while loop.time() < deadline:
        if _seen.frames > baseline:
            return True
        value += 1
        peer.push_frame(value)
        await asyncio.sleep(1 / 30)
    return False


async def test_a_track_published_before_a_reoffer_is_granted_again_on_the_new_wire() -> None:
    """A client that reconnects publishes its webcam again, and the model reads it."""
    peers: list[_Peer] = []
    try:
        async with _runtime() as http:
            stack = await _published_then_reoffered(http, peers)
            async with stack:
                second = peers[-1]
                await second.wire_open()
                await asyncio.sleep(0.5)
                reply = await second.publish("webcam", "again")
                assert "error" not in reply, f"the publish after the reconnect was refused: {reply}"
                assert reply["method"] == "publish_track"
                assert await _webcam_frames_arrive(second), (
                    "no webcam frame sent on the new wire reached the model"
                )
    finally:
        for peer in peers:
            peer.release()


async def test_a_publish_sent_the_moment_the_new_wire_opens_is_granted() -> None:
    """The first publish after a reconnect is answered by the new connection.

    A client publishes as soon as its channels open, with nothing awaited in
    between. That request must be served by the connection the re-offer
    registered, which holds nothing yet, rather than refused for a track the
    replaced wire published.
    """
    peers: list[_Peer] = []
    try:
        async with _runtime() as http:
            stack = await _published_then_reoffered(http, peers)
            async with stack:
                second = peers[-1]
                await second.wire_open()
                reply = await second.publish("webcam", "at-open")
                assert "error" not in reply, (
                    f"the first publish on the new wire was refused: {reply}"
                )
    finally:
        for peer in peers:
            peer.release()
