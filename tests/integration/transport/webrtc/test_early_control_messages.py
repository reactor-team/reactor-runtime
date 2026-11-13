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

from tests.integration.transport.webrtc._signalling import (  # noqa: E402
    TIMEOUT_S,
    negotiate,
    new_connection,
    serve,
    trickle,
)

from reactor_runtime import Output, ReactorApp, Video  # noqa: E402
from reactor_runtime.transport.webrtc import WebRtcConfig  # noqa: E402
from reactor_runtime.transport.webrtc.peer import _get_factory  # noqa: E402

pytestmark = pytest.mark.asyncio

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


class _Client:
    """A libwebrtc peer standing in for a client, opening its channels one at a time.

    It offers the control channel alone, so the data channel opens only when the
    test creates it later, over the association already up.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.control_open = threading.Event()
        self.data_open = threading.Event()
        self.ice: list[Any] = []
        self._tracks: list[Any] = []

        factory = _get_factory(WebRtcConfig())
        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = self.ice.append
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
        """Register, offer, and apply the answer with the candidates it carries."""
        cid = await new_connection(http)
        mapping = [
            {
                "mid": self.recv.mid() or "0",
                "name": "main",
                "kind": "video",
                "direction": "recvonly",
            }
        ]
        await negotiate(http, self.pc, cid, mapping)
        return cid


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
    trickling: asyncio.Task[None] | None = None
    try:
        cid = await client.connect(http)
        trickling = asyncio.create_task(trickle(http, cid, client.ice, stop))
        assert await asyncio.to_thread(client.control_open.wait, TIMEOUT_S), (
            "the control channel never opened"
        )
        yield client
    finally:
        stop.set()
        if trickling is not None:
            trickling.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await trickling
        client.control = client.data = None
        client._tracks.clear()
        client.pc = None


async def test_a_resume_sent_once_both_channels_are_open_starts_the_track() -> None:
    """The control case: the same client, resuming after its wire is fully open."""
    async with serve(_StreamingModel, "early:Model") as http, _connected_client(http) as client:
        client.open_data()
        assert await asyncio.to_thread(client.data_open.wait, TIMEOUT_S)
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
    async with serve(_StreamingModel, "early:Model") as http, _connected_client(http) as client:
        client.control.send(_RESUME_MAIN.encode(), binary=False)
        # Give the runtime time to read the resume before the wire completes.
        await asyncio.sleep(0.5)
        client.open_data()
        assert await asyncio.to_thread(client.data_open.wait, TIMEOUT_S)
        assert await _frames_arrive(client), (
            "a resume sent before the data channel opened was lost; the track stayed paused"
        )
