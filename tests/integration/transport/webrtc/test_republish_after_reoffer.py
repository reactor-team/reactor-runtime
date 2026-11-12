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

from reactor_runtime import InputBuffer, MediaInput, ReactorApp, ReadMode, Video  # noqa: E402
from reactor_runtime.core import RuntimeConfig  # noqa: E402
from reactor_runtime.http.server import build_app  # noqa: E402
from reactor_runtime.metrics import RuntimeMetrics  # noqa: E402
from reactor_runtime.runner.runner import Runner  # noqa: E402
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcRouter  # noqa: E402
from reactor_runtime.transport.webrtc.frames import rgb_to_bgra  # noqa: E402
from reactor_runtime.transport.webrtc.peer import _get_factory, libwebrtc_peer_factory  # noqa: E402

pytestmark = pytest.mark.asyncio

_PREFIX = "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc"
_TIMEOUT_S = 25.0
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
    """Serve the full HTTP surface over a real runner and a real libwebrtc peer."""
    global _seen
    _seen = _WebcamCounter()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: _WebcamModel)
        metrics = RuntimeMetrics(version="0.0.0", model="republish:Model")
        runner = Runner(RuntimeConfig(model_ref="republish:Model"), metrics)
        await runner.start()
        router = WebRtcRouter(
            WebRtcConfig(ping_timeout=0.0, ice_gathering_timeout_ms=4000),
            libwebrtc_peer_factory,
            metrics,
        )
        app = build_app(runner, [router], runner.health, metrics)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                response = await client.post("/start_session", json={})
                assert response.status_code == 200, response.text
                try:
                    yield client
                finally:
                    # Ending the session closes its connections and their media threads.
                    await client.post("/stop_session", json={})
        finally:
            await runner.stop()


class _Peer:
    """One libwebrtc peer of the client: a ``webcam`` it sends and both channels."""

    def __init__(self) -> None:
        self.control_open = threading.Event()
        self.data_open = threading.Event()
        self.replies: list[dict[str, Any]] = []
        self.replied = threading.Event()
        self._ice: list[Any] = []

        factory = _get_factory(WebRtcConfig())
        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = self._ice.append
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
        """Offer on *cid*, apply the answer, and add the candidates it carries."""
        await self.send.set_track(self.camera)
        offer = await self.pc.create_offer()
        await self.pc.set_local_description(offer)
        mapping = [
            {
                "mid": self.send.mid() or "0",
                "name": "webcam",
                "kind": "video",
                "direction": "sendonly",
            }
        ]
        response = await http.request(
            method,
            f"{_PREFIX}/connections/{cid}/sdp_params",
            json={"sdp_offer": offer.sdp, "track_mapping": mapping},
        )
        assert response.status_code == 202, response.text

        response = await _poll_answer(http, cid)
        assert response.status_code == 200, response.text
        answer = response.json()["sdp_answer"]
        await self.pc.set_remote_description(rw.SessionDescription("answer", answer))
        for candidate in _ice_from_answer(answer):
            await self.pc.add_ice_candidate(candidate)

    async def trickle(self, http: httpx.AsyncClient, cid: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            pending, self._ice[:] = self._ice[:], []
            if pending:
                await http.post(
                    f"{_PREFIX}/connections/{cid}/ice_candidates",
                    json={
                        "candidates": [
                            {
                                "candidate": c.candidate,
                                "sdp_mid": c.sdp_mid,
                                "sdp_mline_index": c.sdp_mline_index,
                            }
                            for c in pending
                        ]
                    },
                )
            await asyncio.sleep(0.05)

    async def wire_open(self) -> None:
        for opened, label in ((self.control_open, "control"), (self.data_open, "data")):
            assert await asyncio.to_thread(opened.wait, _TIMEOUT_S), (
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
        deadline = loop.time() + _TIMEOUT_S
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


async def _poll_answer(http: httpx.AsyncClient, cid: int) -> httpx.Response:
    """Poll the answer route until negotiation finishes, as a client does over HTTP."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TIMEOUT_S
    response = await http.get(f"{_PREFIX}/connections/{cid}/sdp_params")
    while response.status_code == 202 and loop.time() < deadline:
        await asyncio.sleep(0.05)
        response = await http.get(f"{_PREFIX}/connections/{cid}/sdp_params")
    return response


def _ice_from_answer(sdp: str) -> list[Any]:
    """Pull the candidates the runtime embedded into its non-trickle answer."""
    candidates: list[Any] = []
    mline_index = -1
    mid: str | None = None
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            mline_index += 1
            mid = None
        elif line.startswith("a=mid:"):
            mid = line[len("a=mid:") :].strip()
        elif line.startswith("a=candidate:"):
            candidates.append(
                rw.IceCandidate(
                    candidate=line[len("a=") :].strip(),
                    sdp_mid=mid,
                    sdp_mline_index=max(mline_index, 0),
                )
            )
    return candidates


@asynccontextmanager
async def _wire(http: httpx.AsyncClient, cid: int, *, method: str) -> AsyncIterator[_Peer]:
    """Negotiate one peer on *cid* and trickle its candidates while it is in use.

    The peer itself outlives the block, so a reconnect can re-offer while the
    wire it replaces is still up; the caller releases it.
    """
    peer = _Peer()
    stop = asyncio.Event()
    trickle: asyncio.Task[None] | None = None
    try:
        await peer.offer(http, cid, method=method)
        trickle = asyncio.create_task(peer.trickle(http, cid, stop))
        yield peer
    finally:
        stop.set()
        if trickle is not None:
            trickle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await trickle


async def _new_connection(http: httpx.AsyncClient) -> int:
    response = await http.post(f"{_PREFIX}/connections")
    assert response.status_code == 201, response.text
    return response.json()["connection_id"]


async def _published_then_reoffered(
    http: httpx.AsyncClient, peers: list[_Peer]
) -> contextlib.AsyncExitStack:
    """Publish ``webcam`` on a first wire, then re-offer on the same id.

    Returns a stack holding the new wire open, whose peer is the last of
    *peers*. The first peer is left up, its wire replaced by the runtime.
    """
    cid = await _new_connection(http)
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
    deadline = loop.time() + _TIMEOUT_S
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
                assert reply.get("error", {}).get("code") != "publish_refused", (
                    f"the first publish on the new wire was refused: {reply}"
                )
                assert "error" not in reply, reply
    finally:
        for peer in peers:
            peer.release()
