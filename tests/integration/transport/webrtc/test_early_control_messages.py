"""A control message a client sends before its wire is fully open.

A client may send on its control channel the moment that channel opens, which
can be before its data channel does. The runtime registers a connection only
once both channels are open, so these tests pin what happens to a
``resume_track`` sent in that window, over a real libwebrtc wire and the full
runtime: the HTTP signalling routes, the runner, and its connection registry.

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
from typing import Any

import httpx
import numpy as np
import pytest

rw = pytest.importorskip("reactor_webrtc")

from reactor_runtime import Output, ReactorApp, Video  # noqa: E402
from reactor_runtime.core import RuntimeConfig  # noqa: E402
from reactor_runtime.http.server import build_app  # noqa: E402
from reactor_runtime.metrics import RuntimeMetrics  # noqa: E402
from reactor_runtime.runner.runner import Runner  # noqa: E402
from reactor_runtime.transport.webrtc import WebRtcConfig, WebRtcRouter  # noqa: E402
from reactor_runtime.transport.webrtc.peer import _get_factory, libwebrtc_peer_factory  # noqa: E402

pytestmark = pytest.mark.asyncio

_PREFIX = "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc"
_TIMEOUT_S = 25.0
# How long a resumed track gets to deliver its first frame before the test
# calls it paused. A loopback wire that is resumed delivers within a second.
_FIRST_FRAME_S = 8.0

_RESUME_MAIN = json.dumps(
    {"type": "notification", "event": "resume_track", "data": {"name": "main"}}
)


class _StreamingOutput(Output):
    main: Video


class _StreamingModel(ReactorApp):
    """Emits a frame every tick for thirty seconds, longer than any test here runs."""

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        for value in range(30 * 30):
            frame = np.full((240, 320, 3), value % 256, dtype=np.uint8)
            await self.emit(_StreamingOutput(main=frame))
            await asyncio.sleep(1 / 30)


@asynccontextmanager
async def _runtime() -> AsyncIterator[httpx.AsyncClient]:
    """Serve the full HTTP surface over a real runner and a real libwebrtc peer."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "reactor_runtime.runner.runner.import_model_class", lambda ref: _StreamingModel
        )
        metrics = RuntimeMetrics(version="0.0.0", model="early:Model")
        runner = Runner(RuntimeConfig(model_ref="early:Model"), metrics)
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


class _Client:
    """A libwebrtc peer standing in for a client, opening its channels one at a time.

    It offers the control channel alone, so the data channel opens only when the
    test creates it later, over the association already up.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.control_open = threading.Event()
        self.data_open = threading.Event()
        self._ice: list[Any] = []
        self._tracks: list[Any] = []

        factory = _get_factory(WebRtcConfig())
        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = self._ice.append
        observer.on_track = self._on_track
        # Dropped at teardown, which is what releases the native peer connection.
        self.pc: Any = factory.create_peer_connection(rw.RtcConfiguration(), observer)
        self.recv = self.pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
        self.control = self._open("control", self.control_open)
        self.data: Any = None

    def _open(self, label: str, opened: threading.Event) -> Any:
        channel = self.pc.create_data_channel(label)
        channel.on_state_change(
            lambda state: opened.set() if state == rw.DataChannelState.Open else None
        )
        if channel.state() == rw.DataChannelState.Open:
            opened.set()
        return channel

    def open_data(self) -> None:
        self.data = self._open("data", self.data_open)

    def _on_track(self, kind: Any, track: Any) -> None:
        self._tracks.append(track)
        if kind == rw.MediaKind.Video:

            def _count(*_: Any) -> None:
                self.frames += 1

            track.on_video_frame(_count)

    async def connect(self, http: httpx.AsyncClient) -> int:
        """Register, offer, apply the answer, and trickle candidates until connected."""
        response = await http.post(f"{_PREFIX}/connections")
        assert response.status_code == 201, response.text
        cid = response.json()["connection_id"]

        offer = await self.pc.create_offer()
        await self.pc.set_local_description(offer)
        mapping = [
            {
                "mid": self.recv.mid() or "0",
                "name": "main",
                "kind": "video",
                "direction": "recvonly",
            }
        ]
        response = await http.post(
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
        return cid

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


async def _frames_arrive(client: _Client, timeout_s: float = _FIRST_FRAME_S) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if client.frames:
            return True
        await asyncio.sleep(0.05)
    return False


@asynccontextmanager
async def _connected_client(http: httpx.AsyncClient) -> AsyncIterator[_Client]:
    client = _Client()
    stop = asyncio.Event()
    trickle: asyncio.Task[None] | None = None
    try:
        cid = await client.connect(http)
        trickle = asyncio.create_task(client.trickle(http, cid, stop))
        assert await asyncio.to_thread(client.control_open.wait, _TIMEOUT_S), (
            "the control channel never opened"
        )
        yield client
    finally:
        stop.set()
        if trickle is not None:
            trickle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await trickle
        client.control = client.data = None
        client._tracks.clear()
        client.pc = None


async def test_a_resume_sent_once_both_channels_are_open_starts_the_track() -> None:
    """The control case: the same client, resuming after its wire is fully open."""
    async with _runtime() as http, _connected_client(http) as client:
        client.open_data()
        assert await asyncio.to_thread(client.data_open.wait, _TIMEOUT_S)
        # Registration follows the data channel opening on the runtime's loop.
        await asyncio.sleep(0.5)
        client.control.send(_RESUME_MAIN.encode(), binary=False)
        assert await _frames_arrive(client), "a resumed track never delivered a frame"


async def test_a_resume_sent_before_the_data_channel_opens_starts_the_track() -> None:
    """A resume sent as soon as the control channel opens still starts the track.

    The client sends it while its data channel is not yet open, so the runtime
    has not registered the connection. The track must still reach the client
    once the wire is up, rather than staying paused for the whole session.
    """
    async with _runtime() as http, _connected_client(http) as client:
        client.control.send(_RESUME_MAIN.encode(), binary=False)
        # Give the runtime time to read the resume before the wire completes.
        await asyncio.sleep(0.5)
        client.open_data()
        assert await asyncio.to_thread(client.data_open.wait, _TIMEOUT_S)
        assert await _frames_arrive(client), (
            "a resume sent before the data channel opened was lost; the track stayed paused"
        )
