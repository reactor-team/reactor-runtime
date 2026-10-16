"""A session's room connection to a real LiveKit room, seen from another participant.

Needs a LiveKit server: ``livekit-server --dev`` answers on the defaults below.
The tests skip when ``LIVEKIT_URL`` is not set.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from livekit import api, rtc

from reactor_runtime import Audio, Output, ReactorApp, Video
from reactor_runtime.core import RuntimeConfig, SessionEvent, SessionState, Transition
from reactor_runtime.http.server import build_app
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner import Runner
from reactor_runtime.runner.runner import SESSION_ID
from reactor_runtime.transport.livekit import LiveKitRouter

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

pytestmark = [
    pytest.mark.skipif(not LIVEKIT_URL, reason="needs a LiveKit server at LIVEKIT_URL"),
    pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning"),
]

_CONNECTIONS = f"/sessions/{SESSION_ID}/transport/livekit/connections"

FPS = 25
RATE = 24_000
TONE_HZ = 440.0
GREY = 180
SIZE = (96, 128)  # height, width


class Voice(Audio):
    sample_rate = RATE


class RoomOutput(Output):
    main_video: Video
    main_audio: Voice


class ToneModel(ReactorApp):
    """A flat grey picture and a continuous 440 Hz tone, one 40 ms frame per step."""

    fps = FPS

    def load(self, config_path: Path | None) -> None:
        self.sample = 0

    def generate(self, input: Any) -> RoomOutput:
        n = RATE // FPS
        t = (np.arange(n) + self.sample) / RATE
        self.sample += n
        tone = (0.5 * 32767 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.int16)
        frame = np.full((*SIZE, 3), GREY, dtype=np.uint8)
        return RoomOutput(main_video=frame, main_audio=tone.reshape(1, -1))


def _token(room: str, identity: str) -> str:
    grants = api.VideoGrants(
        room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True
    )
    return api.AccessToken(API_KEY, API_SECRET).with_identity(identity).with_grants(grants).to_jwt()


async def _until(check: Callable[[], bool], within: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not check():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting")
        await asyncio.sleep(0.05)


def _moves(runner: Runner, event: SessionEvent) -> list[Transition]:
    return [e.transition for _, e in runner.events._history if e.transition.event is event]


@pytest.fixture
async def runner(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Runner]:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: ToneModel)
    runner = Runner(RuntimeConfig(model_ref="tone:ToneModel"))
    await runner.start()
    try:
        yield runner
    finally:
        if runner._sm.current_state in (
            SessionState.WAITING,
            SessionState.STREAMING,
            SessionState.ORPHANED,
        ):
            runner.stop_session()
        await runner._drain_teardown()
        await runner.stop()


@pytest.fixture
async def http(runner: Runner) -> AsyncIterator[httpx.AsyncClient]:
    """The runtime's HTTP surface with the LiveKit routes, on the test's own loop."""
    metrics = RuntimeMetrics(version="0.0.0", model="tone:ToneModel")
    app = build_app(runner, [LiveKitRouter()], runner.health, metrics)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        yield client


@pytest.fixture
def room_name() -> str:
    return f"room-{uuid.uuid4().hex[:8]}"


async def _open(http: httpx.AsyncClient, token: str) -> int:
    """Open a room connection and return its id."""
    response = await http.post(f"{_CONNECTIONS}", json={"url": LIVEKIT_URL, "token": token})
    assert response.status_code == 202, response.text
    return int(response.json()["connection_id"])


async def _settled(http: httpx.AsyncClient, conn_id: int) -> dict[str, Any]:
    """Poll a room connection until its join has ended, and return its status."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15.0
    while True:
        response = await http.get(f"{_CONNECTIONS}/{conn_id}")
        if response.status_code == 200:
            return response.json()
        assert response.status_code == 202, response.text
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the join")
        await asyncio.sleep(0.05)


async def _status(http: httpx.AsyncClient, conn_id: int) -> dict[str, Any]:
    response = await http.get(f"{_CONNECTIONS}/{conn_id}")
    assert response.status_code == 200, response.text
    return response.json()


class Watcher:
    """A participant that subscribes to the session's tracks and records what arrives."""

    def __init__(self) -> None:
        self.room = rtc.Room()
        self.tracks: dict[str, rtc.Track] = {}
        self.video: list[np.ndarray] = []
        self.audio: list[np.ndarray] = []
        self.audio_rates: set[int] = set()
        self.left: list[str] = []
        self._tasks: list[asyncio.Task[None]] = []
        self.room.on("track_subscribed", self._on_track)
        self.room.on("participant_disconnected", lambda p: self.left.append(p.identity))

    async def join(self, room: str) -> None:
        await self.room.connect(LIVEKIT_URL, _token(room, "watcher"))

    def _on_track(self, track: rtc.Track, *_: Any) -> None:
        self.tracks[track.name] = track
        reader = self._read_video if track.kind == rtc.TrackKind.KIND_VIDEO else self._read_audio
        self._tasks.append(asyncio.ensure_future(reader(track)))

    async def _read_video(self, track: rtc.Track) -> None:
        async for event in rtc.VideoStream(track):
            frame = event.frame.convert(rtc.VideoBufferType.RGB24)
            self.video.append(
                np.frombuffer(bytes(frame.data), np.uint8).reshape(frame.height, frame.width, 3)
            )

    async def _read_audio(self, track: rtc.Track) -> None:
        async for event in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            self.audio_rates.add(event.frame.sample_rate)
            self.audio.append(np.frombuffer(bytes(event.frame.data), np.int16).copy())

    async def leave(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.room.disconnect()


async def test_a_room_connection_joins_the_room_and_publishes_the_output(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    assert await _settled(http, conn_id) == {"connection_id": conn_id, "state": "open"}

    # The room connection occupies the session as a system connection.
    _expect_state(runner, SessionState.STREAMING)
    (opened,) = _moves(runner, SessionEvent.CONNECTION_OPENED)
    assert dict(opened.detail) == {"conn_id": conn_id, "system": True, "transport": "livekit"}
    assert {"protocol": "livekit", "version": "1.0"} in runner.descriptor()["transports"]

    watcher = Watcher()
    await watcher.join(room_name)
    try:
        await _until(lambda: len(watcher.video) >= 75 and len(watcher.audio) >= 300)
        # Track declarations are process-wide, so other modules of the run add theirs.
        assert {"main_video", "main_audio"} <= set(watcher.tracks)

        # The first frame the room shows is the model's, not a black frame.
        assert abs(float(watcher.video[0].mean()) - GREY) < 12
        assert watcher.video[0].shape[:2] == SIZE

        # The tone arrives at the declared rate, whole: no gaps, and at 440 Hz.
        assert watcher.audio_rates == {RATE}
        steady = np.concatenate(watcher.audio[20:])
        blocks = steady[: len(steady) // 240 * 240].reshape(-1, 240)
        rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
        assert rms.min() > 0.25 * 32767 * 0.5, "the tone has a silent gap"
        spectrum = np.abs(np.fft.rfft(steady.astype(np.float64)))
        peak_hz = np.fft.rfftfreq(len(steady), 1 / RATE)[int(spectrum.argmax())]
        assert abs(peak_hz - TONE_HZ) < 5
    finally:
        await watcher.leave()


async def test_a_session_has_one_room_connection_at_a_time(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    await _settled(http, conn_id)

    second = await http.post(
        _CONNECTIONS, json={"url": LIVEKIT_URL, "token": _token(room_name, "model-2")}
    )

    assert second.status_code == 409, second.text


async def test_stopping_the_session_leaves_the_room(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    await _settled(http, conn_id)
    watcher = Watcher()
    await watcher.join(room_name)
    try:
        await _until(lambda: "model" in watcher.room.remote_participants)

        runner.stop_session()
        await _until(lambda: runner._sm.current_state is SessionState.READY)

        await _until(lambda: watcher.left == ["model"])
    finally:
        await watcher.leave()


async def test_leaving_the_room_closes_the_room_connection_and_keeps_the_session(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    await _settled(http, conn_id)
    watcher = Watcher()
    await watcher.join(room_name)
    try:
        await _until(lambda: "model" in watcher.room.remote_participants)

        left = await http.delete(f"{_CONNECTIONS}/{conn_id}")

        assert left.status_code == 204
        await _until(lambda: watcher.left == ["model"])
        assert await _status(http, conn_id) == {
            "connection_id": conn_id,
            "state": "closed",
            "reason": "the room was left on request",
        }
        # With no connection left the session is orphaned, not stopped.
        _expect_state(runner, SessionState.ORPHANED)
        assert _moves(runner, SessionEvent.STOP_SESSION) == []
    finally:
        await watcher.leave()


async def test_a_refused_join_reports_the_failure_and_keeps_the_session(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    bad = api.AccessToken(API_KEY, "not-the-secret-of-this-server").with_identity("model")
    token = bad.with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt()
    runner.start_session({})

    conn_id = await _open(http, token)
    status = await _settled(http, conn_id)

    assert status["state"] == "failed"
    assert status["reason"].startswith("the LiveKit room could not be joined:")
    assert token not in status["reason"]
    (error,) = _moves(runner, SessionEvent.ERROR)
    assert error.detail["conn_id"] == conn_id
    assert error.detail["transport"] == "livekit"
    assert error.detail["message"] == status["reason"]
    _expect_state(runner, SessionState.WAITING)
    assert _moves(runner, SessionEvent.CONNECTION_OPENED) == []


async def test_the_room_closing_closes_the_room_connection_and_keeps_the_session(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    await _settled(http, conn_id)

    http_url = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    service = api.LiveKitAPI(http_url, API_KEY, API_SECRET)
    try:
        await service.room.delete_room(api.DeleteRoomRequest(room=room_name))
    finally:
        await service.aclose()
    await _until(lambda: bool(_moves(runner, SessionEvent.CONNECTION_CLOSED)))

    (closed,) = _moves(runner, SessionEvent.CONNECTION_CLOSED)
    assert dict(closed.detail) == {"conn_id": conn_id, "system": True, "transport": "livekit"}
    status = await _status(http, conn_id)
    assert status["state"] == "closed"
    assert status["reason"].startswith("the LiveKit room connection closed")
    _expect_state(runner, SessionState.ORPHANED)
    assert _moves(runner, SessionEvent.STOP_SESSION) == []


async def test_the_token_never_reaches_the_journal(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    token = _token(room_name, "model")
    runner.start_session({})

    conn_id = await _open(http, token)
    await _settled(http, conn_id)

    assert all(token not in repr(e.transition.detail) for _, e in runner.events._history)


def _expect_state(runner: Runner, state: SessionState) -> None:
    assert runner._sm.current_state is state
