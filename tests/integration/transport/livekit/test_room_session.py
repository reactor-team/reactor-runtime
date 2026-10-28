"""A session's room connection to a real LiveKit room, seen from another participant.

Needs a LiveKit server: ``livekit-server --dev`` answers on the defaults below.
The tests skip when ``LIVEKIT_URL`` is not set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pytest
from livekit import api, rtc

from reactor_runtime import (
    Audio,
    CommandError,
    InputBuffer,
    MediaInput,
    MessageField,
    ModelMessage,
    Output,
    ReactorApp,
    ReadMode,
    Video,
    event,
)
from reactor_runtime.core import (
    ConnId,
    LiveKitConfig,
    LiveKitRpcMethod,
    RuntimeConfig,
    SessionEvent,
    SessionState,
    Transition,
)
from reactor_runtime.http.server import build_app
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.protocol import Channel, Direction, ProtocolVersion, select
from reactor_runtime.protocol.common import dict_to_struct, struct_to_dict
from reactor_runtime.runner import Runner
from reactor_runtime.runner.runner import SESSION_ID
from reactor_runtime.transport.livekit import LiveKitRouter
from reactor_runtime.transport.livekit.connection import AUDIO_QUEUE_MS
from reactor_runtime.transport.livekit.participant import (
    CONTROL_TOPIC,
    DATA_TOPIC,
    MAX_PACKET_BYTES,
)
from reactor_wire.v1 import data_pb2, model_pb2

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


class AgentAudio(Audio):
    sample_rate = RATE
    buffer_seconds = 120


class RoomInput(MediaInput):
    agent_audio: AgentAudio
    camera: Video


class Heard(ModelMessage):
    samples: int = MessageField(description="Samples of agent audio received so far.")
    digest: str = MessageField(description="SHA-256 of every sample received, in order.")
    streams: list[str] = MessageField(description="The streams the samples came from.")
    ended: list[str] = MessageField(description="The streams that ended.")
    stream_digests: dict[str, str] = MessageField(
        description="SHA-256 of each stream's samples, in order, by stream id."
    )
    camera_frames: int = MessageField(description="Camera frames received so far.")
    camera_mean: float = MessageField(description="The last camera frame's mean value.")


class Cut(ModelMessage):
    unheard: float = MessageField(description="Seconds of paced-out audio the flush dropped.")


class Echoed(ModelMessage):
    text: str = MessageField(description="The text the command carried.")


class Announcement(ModelMessage):
    text: str = MessageField(description="The text announced to every client.")


class ToneModel(ReactorApp):
    """A flat grey picture and a continuous 440 Hz tone, one 40 ms frame per step."""

    fps = FPS
    media: RoomInput

    @event(name="heard", description="Reply with `heard`: what the input tracks received.")
    async def heard(self) -> Heard:
        audio = cast(InputBuffer, self.media.agent_audio)
        for frame in audio.try_read(audio.available, ReadMode.FIFO) or []:
            info = json.loads(frame.metadata or b"{}")
            if info.get("ended"):
                self.ended.append(info["stream_id"])
                continue
            if info["stream_id"] not in self.streams:
                self.streams.append(info["stream_id"])
            self.received.append(frame.data[0])
            self.by_stream.setdefault(info["stream_id"], []).append(frame.data[0])
        camera = cast(InputBuffer, self.media.camera)
        frames = camera.try_read(camera.available, ReadMode.FIFO) or []
        self.camera_frames += len(frames)
        if frames:
            self.camera_mean = float(frames[-1].data.mean())
        samples = np.concatenate(self.received) if self.received else np.empty(0, np.int16)
        return Heard(
            samples=int(samples.size),
            digest=hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest(),
            streams=self.streams,
            ended=self.ended,
            stream_digests={
                sid: hashlib.sha256(np.concatenate(parts).astype("<i2").tobytes()).hexdigest()
                for sid, parts in self.by_stream.items()
            },
            camera_frames=self.camera_frames,
            camera_mean=self.camera_mean,
        )

    @event(name="echo", description="Reply with `echoed`, carrying `text`.")
    async def echo(self, text: str) -> Echoed:
        return Echoed(text=text)

    @event(name="announce", description="Broadcast `announcement` to every client.")
    async def announce(self, text: str) -> None:
        await self.send(Announcement(text=text))

    @event(name="cut", description="Flush the output, holding the picture, and reply with `cut`.")
    async def cut(self) -> Cut:
        return Cut(unheard=self.output.flush(hold=True))

    @event(name="quiet", description="Do nothing and reply with no message.")
    async def quiet(self) -> None:
        return None

    @event(name="fail", description="Refuse with `not_now`.")
    async def fail(self) -> None:
        raise CommandError("not_now", "Not now.")

    def load(self, config_path: Path | None) -> None:
        self.sample = 0
        self.received: list[np.ndarray] = []
        self.streams: list[str] = []
        self.ended: list[str] = []
        self.by_stream: dict[str, list[np.ndarray]] = {}
        self.camera_frames = 0
        self.camera_mean = 0.0

    def generate(self, input: Any) -> RoomOutput:
        n = RATE // FPS
        t = (np.arange(n) + self.sample) / RATE
        self.sample += n
        tone = (0.5 * 32767 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.int16)
        frame = np.full((*SIZE, 3), GREY, dtype=np.uint8)
        return RoomOutput(main_video=frame, main_audio=tone.reshape(1, -1))


# The room's own signals for the test model.
SIGNALS = LiveKitConfig(
    rpc=(
        LiveKitRpcMethod(method="lk.echo", command="echo"),
        LiveKitRpcMethod(method="lk.quiet", command="quiet"),
        LiveKitRpcMethod(method="lk.cut", command="cut"),
        LiveKitRpcMethod(method="lk.fail", command="fail"),
        LiveKitRpcMethod(method="lk.announced", message="announcement"),
    ),
    topics=("avatar",),
    media_inputs={"lk.audio_stream": "agent_audio"},
)


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
    runner = Runner(RuntimeConfig(model_ref="tone:ToneModel", livekit=SIGNALS))
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
    app = build_app(runner, [LiveKitRouter(SIGNALS)], runner.health, metrics)
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
        # Only the model's own tracks: other modules of the run publish theirs too.
        if track.name == "main_video":
            self._tasks.append(asyncio.ensure_future(self._read_video(track)))
        elif track.name == "main_audio":
            self._tasks.append(asyncio.ensure_future(self._read_audio(track)))

    async def _read_video(self, track: rtc.Track) -> None:
        async for received in rtc.VideoStream(track):
            frame = received.frame.convert(rtc.VideoBufferType.RGB24)
            self.video.append(
                np.frombuffer(bytes(frame.data), np.uint8).reshape(frame.height, frame.width, 3)
            )

    async def _read_audio(self, track: rtc.Track) -> None:
        async for received in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            self.audio_rates.add(received.frame.sample_rate)
            self.audio.append(np.frombuffer(bytes(received.frame.data), np.int16).copy())

    async def leave(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.room.disconnect()


async def test_a_room_connection_joins_the_room_and_publishes_the_output(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    assert 5000 <= conn_id <= 9999
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
        # The receiver's level wobbles while it starts up; once the tone has held
        # its level for 100 ms, no block after it is quiet.
        received = np.concatenate(watcher.audio)
        blocks = received[: len(received) // 240 * 240].reshape(-1, 240)
        rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
        level = 0.25 * 32767 * 0.5
        held = np.convolve(rms > level, np.ones(10), mode="valid") == 10
        assert held.any(), "the tone never held its level"
        onset = int(np.argmax(held))
        assert len(rms) - onset >= 150, "the tone held its level too late to judge"
        assert rms[onset:].min() > level, "the tone has a silent gap"
        steady = blocks[onset:].reshape(-1)
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


async def test_a_room_connection_opens_on_the_id_the_caller_chose(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    body = {"url": LIVEKIT_URL, "token": _token(room_name, "model"), "connection_id": 5120}

    response = await http.post(_CONNECTIONS, json=body)

    assert response.status_code == 202, response.text
    assert response.json() == {"connection_id": 5120}
    assert await _settled(http, 5120) == {"connection_id": 5120, "state": "open"}
    (opened,) = _moves(runner, SessionEvent.CONNECTION_OPENED)
    assert dict(opened.detail) == {"conn_id": 5120, "system": True, "transport": "livekit"}


async def test_a_room_connection_opens_through_the_ice_servers_the_caller_supplies(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    # The local server answers on its host candidates, so the join opens even
    # though nothing answers on the caller's STUN server.
    runner.start_session({})
    body = {
        "url": LIVEKIT_URL,
        "token": _token(room_name, "model"),
        "ice_servers": [{"uris": ["stun:127.0.0.1:3479"]}],
    }

    response = await http.post(_CONNECTIONS, json=body)

    assert response.status_code == 202, response.text
    conn_id = int(response.json()["connection_id"])
    assert await _settled(http, conn_id) == {"connection_id": conn_id, "state": "open"}


async def test_a_chosen_id_another_transport_holds_is_refused(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    runner.start_session({})
    runner.claim_conn_id(ConnId(5120), "webrtc")
    body = {"url": LIVEKIT_URL, "token": _token(room_name, "model"), "connection_id": 5120}

    refused = await http.post(_CONNECTIONS, json=body)

    assert refused.status_code == 409, refused.text
    assert _moves(runner, SessionEvent.CONNECTION_OPENED) == []


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


V1 = select(ProtocolVersion.V1)


class ProtocolClient(Watcher):
    """A participant that speaks the Reactor protocol over the room's data messages."""

    def __init__(self, identity: str) -> None:
        super().__init__()
        self.identity = identity
        # (how it arrived, the decoded message), in arrival order.
        self.received: list[tuple[str, Any]] = []
        self.room.on("data_received", self._on_data)
        for topic in (DATA_TOPIC, CONTROL_TOPIC):
            self.room.register_byte_stream_handler(topic, self._on_stream)

    async def join(self, room: str) -> None:
        await self.room.connect(LIVEKIT_URL, _token(room, self.identity))

    def _decode(self, topic: str, data: bytes) -> Any:
        channel = Channel.DATA if topic == DATA_TOPIC else Channel.CONTROL
        return V1.decode(data, channel, Direction.SERVER)

    def _on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic in (DATA_TOPIC, CONTROL_TOPIC):
            self.received.append(("packet", self._decode(packet.topic, packet.data)))

    def _on_stream(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        async def read() -> None:
            data = b"".join([chunk async for chunk in reader])
            self.received.append(("stream", self._decode(reader.info.topic, data)))

        self._tasks.append(asyncio.ensure_future(read()))

    async def command(self, name: str, request_id: str, **args: Any) -> None:
        message = data_pb2.DataClientMessage(
            request_id=request_id, command=model_pb2.Command(type=name, data=dict_to_struct(args))
        )
        _, frame = V1.encode(message)
        assert isinstance(frame, bytes)
        local = self.room.local_participant
        if len(frame) <= MAX_PACKET_BYTES:
            await local.publish_data(frame, destination_identities=["model"], topic=DATA_TOPIC)
        else:
            await local.send_bytes(frame, topic=DATA_TOPIC, destination_identities=["model"])

    def model_messages(self) -> list[tuple[str, str, str, dict[str, Any]]]:
        return [
            (how, m.request_id, m.message.type, struct_to_dict(m.message.data))
            for how, m in self.received
            if isinstance(m, data_pb2.DataServerMessage) and m.WhichOneof("payload") == "message"
        ]


async def _joined(runner: Runner, http: httpx.AsyncClient, room_name: str) -> int:
    """Start a session, connect it to the room, and return the room connection's id."""
    runner.start_session({})
    conn_id = await _open(http, _token(room_name, "model"))
    assert (await _settled(http, conn_id))["state"] == "open"
    return conn_id


def _participants(runner: Runner) -> list[Transition]:
    return [
        move for move in _moves(runner, SessionEvent.CONNECTION_OPENED) if "identity" in move.detail
    ]


async def test_each_participant_is_a_client_named_by_its_identity(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    room_id = await _joined(runner, http, room_name)
    client = ProtocolClient("viewer")
    await client.join(room_name)
    await _until(lambda: bool(_participants(runner)))

    (opened,) = _participants(runner)
    conn_id = opened.detail["conn_id"]
    assert dict(opened.detail) == {"conn_id": conn_id, "transport": "livekit", "identity": "viewer"}
    assert 5000 <= conn_id <= 9999
    assert conn_id != room_id
    assert (opened.from_state, opened.to_state) == (SessionState.STREAMING, SessionState.STREAMING)

    await client.leave()
    await _until(lambda: bool(_moves(runner, SessionEvent.CONNECTION_CLOSED)))
    (closed,) = _moves(runner, SessionEvent.CONNECTION_CLOSED)
    assert dict(closed.detail) == dict(opened.detail)
    _expect_state(runner, SessionState.STREAMING)


async def test_a_participant_already_in_the_room_becomes_a_client_at_the_join(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    client = ProtocolClient("early")
    await client.join(room_name)
    try:
        await _joined(runner, http, room_name)
        await _until(lambda: bool(_participants(runner)))
        assert _participants(runner)[0].detail["identity"] == "early"
    finally:
        await client.leave()


async def test_leaving_the_room_closes_every_participant_before_the_room_connection(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    room_id = await _joined(runner, http, room_name)
    client = ProtocolClient("viewer")
    await client.join(room_name)
    try:
        await _until(lambda: bool(_participants(runner)))
        (participant,) = _participants(runner)

        await http.delete(f"{_CONNECTIONS}/{room_id}")

        closed = [move.detail["conn_id"] for move in _moves(runner, SessionEvent.CONNECTION_CLOSED)]
        assert closed == [participant.detail["conn_id"], room_id]
        _expect_state(runner, SessionState.ORPHANED)
    finally:
        await client.leave()


async def test_a_command_is_answered_to_the_participant_that_sent_it(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    asker, bystander = ProtocolClient("asker"), ProtocolClient("bystander")
    await asker.join(room_name)
    await bystander.join(room_name)
    try:
        await _until(lambda: len(_participants(runner)) == 2)

        await asker.command("echo", "req-1", text="hello")
        await _until(lambda: bool(asker.model_messages()))

        assert asker.model_messages() == [("packet", "req-1", "echoed", {"text": "hello"})]
        await asyncio.sleep(0.3)
        assert bystander.model_messages() == []
    finally:
        await asker.leave()
        await bystander.leave()


async def test_a_broadcast_reaches_every_participant(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    one, two = ProtocolClient("one"), ProtocolClient("two")
    await one.join(room_name)
    await two.join(room_name)
    try:
        await _until(lambda: len(_participants(runner)) == 2)

        await one.command("announce", "req-2", text="to everyone")
        await _until(lambda: bool(one.model_messages()) and bool(two.model_messages()))

        expected = ("packet", "", "announcement", {"text": "to everyone"})
        assert one.model_messages() == [expected]
        assert two.model_messages() == [expected]
    finally:
        await one.leave()
        await two.leave()


async def test_messages_above_the_packet_limit_travel_as_byte_streams(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    client = ProtocolClient("viewer")
    await client.join(room_name)
    try:
        await _until(lambda: bool(_participants(runner)))
        text = "x" * 40_000

        # Sent as a byte stream, and answered with one.
        await client.command("echo", "req-3", text=text)
        await _until(lambda: bool(client.model_messages()))

        assert client.model_messages() == [("stream", "req-3", "echoed", {"text": text})]
    finally:
        await client.leave()


async def _client_in_session(runner: Runner, room_name: str, identity: str) -> ProtocolClient:
    client = ProtocolClient(identity)
    await client.join(room_name)
    await _until(lambda: any(t.detail["identity"] == identity for t in _participants(runner)))
    return client


def _conn_id_of(runner: Runner, identity: str) -> int:
    (opened,) = [t for t in _participants(runner) if t.detail["identity"] == identity]
    return opened.detail["conn_id"]


async def _call(client: ProtocolClient, method: str, payload: str = "") -> str:
    return await client.room.local_participant.perform_rpc(
        destination_identity="model", method=method, payload=payload
    )


async def test_an_rpc_call_runs_its_command_and_returns_the_reply(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    client = await _client_in_session(runner, room_name, "caller")
    try:
        assert await _call(client, "lk.echo", '{"text": "hi"}') == '{"text": "hi"}'
        assert await _call(client, "lk.quiet") == ""

        (command,) = [t for t in _moves(runner, SessionEvent.COMMAND) if t.detail["name"] == "echo"]
        assert command.detail["conn_id"] == _conn_id_of(runner, "caller")
    finally:
        await client.leave()


async def test_a_flush_reports_the_audio_the_room_had_not_played_out(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    client = await _client_in_session(runner, room_name, "caller")
    try:
        # The tone has been playing into the room for a while.
        await asyncio.sleep(0.5)

        unheard = json.loads(await _call(client, "lk.cut"))["unheard"]

        # The source holds about one paced frame ahead of the room, never more than its queue.
        assert 0.0 <= unheard <= AUDIO_QUEUE_MS / 1000
    finally:
        await client.leave()


async def test_an_rpc_call_whose_command_refuses_fails_with_the_code(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    client = await _client_in_session(runner, room_name, "caller")
    try:
        with pytest.raises(rtc.RpcError) as refused:
            await _call(client, "lk.fail")
        assert refused.value.code == rtc.RpcError.ErrorCode.APPLICATION_ERROR
        assert (refused.value.message, refused.value.data) == ("Not now.", "not_now")

        with pytest.raises(rtc.RpcError) as malformed:
            await _call(client, "lk.echo", "[1, 2]")
        assert malformed.value.data == "invalid_command"

        with pytest.raises(rtc.RpcError) as invalid:
            await _call(client, "lk.echo", '{"text": 7}')
        assert invalid.value.data == "invalid_command"

        with pytest.raises(rtc.RpcError) as unknown:
            await _call(client, "lk.nothing")
        assert unknown.value.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD
    finally:
        await client.leave()


async def test_a_topic_runs_its_command_and_a_mapped_message_calls_every_participant(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    one = await _client_in_session(runner, room_name, "one")
    two = ProtocolClient("two")
    await two.join(room_name)
    called: dict[str, list[str]] = {"one": [], "two": []}
    for client in (one, two):

        async def announced(call: rtc.RpcInvocationData, who: str = client.identity) -> str:
            called[who].append(call.payload)
            return ""

        client.room.local_participant.register_rpc_method("lk.announced", announced)
    try:
        await _until(lambda: len(_participants(runner)) == 2)

        await one.room.local_participant.publish_data(
            b'{"text": "via topic"}', destination_identities=["model"], topic="avatar.announce"
        )
        await _until(lambda: bool(called["one"]) and bool(called["two"]))

        assert called == {"one": ['{"text": "via topic"}'], "two": ['{"text": "via topic"}']}
        # The broadcast also reaches each participant as a protocol message.
        assert one.model_messages() == [("packet", "", "announcement", {"text": "via topic"})]
    finally:
        await one.leave()
        await two.leave()


async def test_a_topic_cannot_run_a_command_an_rpc_method_runs(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    client = await _client_in_session(runner, room_name, "sender")
    try:
        await client.room.local_participant.publish_data(
            b'{"text": "nope"}', destination_identities=["model"], topic="avatar.echo"
        )
        await client.room.local_participant.publish_data(
            b'{"text": "yes"}', destination_identities=["model"], topic="avatar.announce"
        )
        await _until(
            lambda: any(
                t.detail["name"] == "announce" for t in _moves(runner, SessionEvent.COMMAND)
            )
        )

        assert [t.detail["name"] for t in _moves(runner, SessionEvent.COMMAND)] == ["announce"]
    finally:
        await client.leave()


async def _heard(client: ProtocolClient, request_id: str) -> dict[str, Any]:
    await client.command("heard", request_id)
    await _until(lambda: any(rid == request_id for _, rid, _, _ in client.model_messages()))
    return next(data for _, rid, _, data in client.model_messages() if rid == request_id)


async def _heard_ended(client: ProtocolClient, stream_id: str, label: str) -> dict[str, Any]:
    """Ask what the model heard until it has heard *stream_id* end.

    The stream and the command travel on different topics, so one can overtake the other.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15.0
    attempt = 0
    while True:
        heard = await _heard(client, f"{label}-{attempt}")
        if stream_id in heard["ended"] or loop.time() > deadline:
            return heard
        attempt += 1
        await asyncio.sleep(0.1)


async def _stream(client: ProtocolClient, data: bytes, rate: int, channels: int) -> str:
    writer = await client.room.local_participant.stream_bytes(
        "utterance",
        topic="lk.audio_stream",
        destination_identities=["model"],
        attributes={"sample_rate": str(rate), "num_channels": str(channels)},
    )
    # Uneven chunks, some splitting a sample in two.
    for start in range(0, len(data), 3_001):
        await writer.write(data[start : start + 3_001])
    await writer.aclose()
    return writer.info.stream_id


async def test_a_byte_stream_reaches_the_input_track_sample_for_sample(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    agent = await _client_in_session(runner, room_name, "agent")
    try:
        t = np.arange(2 * RATE) / RATE
        speech = (12_000 * np.sin(2 * np.pi * 220 * t)).astype("<i2")

        stream_id = await _stream(agent, speech.tobytes(), RATE, 1)
        heard = await _heard_ended(agent, stream_id, "heard-1")

        assert heard["samples"] == speech.size
        assert heard["digest"] == hashlib.sha256(speech.tobytes()).hexdigest()
        assert heard["streams"] == [stream_id]
        assert heard["ended"] == [stream_id]
    finally:
        await agent.leave()


async def test_a_stream_opened_before_the_last_one_closes_leaves_it_whole(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    agent = await _client_in_session(runner, room_name, "agent")
    attributes = {"sample_rate": str(RATE), "num_channels": "1"}
    try:
        t = np.arange(RATE) / RATE
        first = (12_000 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes()
        second = (9_000 * np.sin(2 * np.pi * 330 * t)).astype("<i2").tobytes()
        half = len(first) // 2

        one = await agent.room.local_participant.stream_bytes(
            "first",
            topic="lk.audio_stream",
            destination_identities=["model"],
            attributes=attributes,
        )
        await one.write(first[:half])
        two = await agent.room.local_participant.stream_bytes(
            "second",
            topic="lk.audio_stream",
            destination_identities=["model"],
            attributes=attributes,
        )
        await two.write(second)
        await one.write(first[half:])
        await one.aclose()
        await two.aclose()
        await _heard_ended(agent, one.info.stream_id, "overlap-1")
        heard = await _heard_ended(agent, two.info.stream_id, "overlap-2")

        assert sorted(heard["ended"]) == sorted([one.info.stream_id, two.info.stream_id])
        assert heard["stream_digests"] == {
            one.info.stream_id: hashlib.sha256(first).hexdigest(),
            two.info.stream_id: hashlib.sha256(second).hexdigest(),
        }
    finally:
        await agent.leave()


async def test_a_byte_stream_at_another_rate_is_resampled_and_mixed_to_mono(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    agent = await _client_in_session(runner, room_name, "agent")
    try:
        t = np.arange(48_000) / 48_000
        tone = (8_000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
        stereo = np.stack([tone, tone], axis=1).reshape(-1)

        stream_id = await _stream(agent, stereo.tobytes(), 48_000, 2)
        heard = await _heard_ended(agent, stream_id, "heard-2")

        # One second of audio at the track's 24 kHz, give or take the resampler's edge.
        assert abs(heard["samples"] - RATE) <= 240
        assert heard["ended"] == [stream_id]
    finally:
        await agent.leave()


async def test_a_track_published_under_an_input_tracks_name_feeds_it(
    runner: Runner, http: httpx.AsyncClient, room_name: str
) -> None:
    await _joined(runner, http, room_name)
    viewer = await _client_in_session(runner, room_name, "viewer")
    source = rtc.VideoSource(64, 48)
    track = rtc.LocalVideoTrack.create_video_track("camera", source)
    await viewer.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    )
    try:
        frame = np.full((48, 64, 3), 77, np.uint8)
        for _ in range(30):
            source.capture_frame(rtc.VideoFrame(64, 48, rtc.VideoBufferType.RGB24, frame.tobytes()))
            await asyncio.sleep(0.05)
        heard = await _heard(viewer, "heard-3")

        assert heard["camera_frames"] > 0
        assert abs(heard["camera_mean"] - 77) < 6
    finally:
        await viewer.leave()
