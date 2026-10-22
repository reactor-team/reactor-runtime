"""The runtime under test, its probe, a bare client, and the sampler.

The runtime runs from source in its own process, started by ``launcher.py``,
and every resource reading is of that process, never of pytest's. The client
is a ``reactor_webrtc`` peer driven from here: it does the runtime's HTTP
signaling and the control messages itself, in v0 JSON (:class:`BareClient`) or
v1 protobuf (:class:`V1Client`), so no client SDK is involved.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import reactor_webrtc as rw
from google.protobuf import struct_pb2
from google.protobuf.message import DecodeError

from endurance.report import MetricResult
from endurance.trends import ENDURANCE_DURATION_SECONDS, Sample
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.frames import rgb_to_bgra
from reactor_runtime.transport.webrtc.peer import _get_factory
from reactor_wire.v1 import (
    common_pb2,
    control_pb2,
    data_pb2,
    model_pb2,
    platform_pb2,
    step_pb2,
    track_pb2,
)

_REQUEST = common_pb2.MessageKind.MESSAGE_KIND_REQUEST
_RESPONSE = common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
_NOTIFICATION = common_pb2.MessageKind.MESSAGE_KIND_NOTIFICATION

SUITE_DIR = Path(__file__).resolve().parent.parent
# The model every scenario serves unless it names another.
MODEL_DIR = SUITE_DIR / "model"
# Where the distributed package's shared-memory blocks live on Linux, named
# ``rr`` and 8 hex digits, ``_`` and a generation number.
SHM_DIR = Path("/dev/shm")
_SHM_BLOCKS = "rr????????_*"

# How long a closed connection may take to leave the runtime's registry. The
# runtime notices a client that closed its peer connection only some seconds
# later, so this is generous; a connection still registered after it is
# reported, not waited on forever.
CLOSE_TIMEOUT_S = 30.0
# How long the runtime has to exit once asked to: longer than a distributed
# model's runner gives its workers to shut down.
STOP_TIMEOUT_S = 45.0
# How long the probe is re-read after the registry is empty, for a teardown
# still in flight to finish. A count still nonzero after it is a leak.
SETTLE_TIMEOUT_S = 10.0
# Frames the client waits to receive back before it disconnects.
FRAMES_PER_CYCLE = 60
_FRAME_WAIT_S = 10.0
_FPS = 30


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RuntimeEndpoint:
    """The runtime's HTTP API at a URL: all a client needs, from any process."""

    def __init__(self, url: str) -> None:
        """Talk to the runtime served at *url*."""
        self.url = url

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        """Send one HTTP request to the runtime; return its status and decoded JSON body."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)

    def get(self, path: str) -> Any:
        """Return the decoded JSON body of ``GET`` *path*."""
        return self.request("GET", path)[1]


class RuntimeProcess(RuntimeEndpoint):
    """The runtime under test, in its own process, with the probe route mounted."""

    def __init__(self, log_path: Path, *, model_dir: Path = MODEL_DIR) -> None:
        """Start the runtime on a free local port, its output going to *log_path*.

        It serves the model in *model_dir*, read from its ``reactor.yaml``.
        """
        self.port = _free_port()
        super().__init__(f"http://127.0.0.1:{self.port}")
        env = {
            # Two malloc arenas unless the caller sets its own count. glibc
            # opens arenas as threads contend, up to 8 per core, and each keeps
            # the most it ever held: a runner's 4 cores took the runtime from
            # 550 MB to 1.5 GB in 30 minutes, a plateau set by the allocator
            # that the RSS trends read as a leak.
            "MALLOC_ARENA_MAX": "2",
            **os.environ,
            "PORT": str(self.port),
            "HOST": "127.0.0.1",
            "REACTOR_LOG_LEVEL": os.environ.get("REACTOR_LOG_LEVEL", "WARNING"),
        }
        self._log = log_path.open("w")
        self._proc = subprocess.Popen(
            [sys.executable, str(SUITE_DIR / "endurance" / "launcher.py")],
            cwd=model_dir,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self.process = psutil.Process(self._proc.pid)

    def wait_ready(self, timeout_s: float = 60.0) -> dict[str, Any]:
        """Wait for the runtime's session to be ready, and return ``/session``'s answer.

        ``/session`` can answer while the model still loads, in state
        ``created``, and a session cannot start until it reads ``ready``.

        Raises:
            RuntimeError: The runtime exited while starting.
            TimeoutError: It was not ready within *timeout_s*.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"the runtime exited with {self._proc.returncode} on startup")
            try:
                session = self.get("/session")
                if session.get("state") == "ready":
                    return session
                time.sleep(0.25)
            # An answer that is an error is a broken runtime, not a slow one.
            except urllib.error.HTTPError:
                raise
            # The server accepts before the runtime can answer, so a read that
            # times out is as much "not ready yet" as a refused connection.
            except OSError:
                time.sleep(0.25)
        raise TimeoutError("the runtime's session never became ready")

    def stop(self) -> None:
        """Stop the runtime, killing it and what it started if they do not exit in time."""
        children = self.process.children(recursive=True)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        # A worker that outlived the runtime would hold its shared memory
        # into the next scenario.
        reap(children)
        self._log.close()

    def probe(self) -> dict[str, Any]:
        """Return one reading of the probe route."""
        return self.get("/_endurance/probe")

    def input_frames(self) -> dict[str, float]:
        """Return each input track's frames received this session, without a full probe."""
        return self.get("/_endurance/input_frames")

    def wait_registered(self, count: int) -> dict[str, Any]:
        """Wait until at most *count* connections are registered, or ``CLOSE_TIMEOUT_S``.

        Returns the last probe reading either way.
        """
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        reading = self.probe()
        while reading["registered_connections"] > count and time.monotonic() < deadline:
            time.sleep(0.25)
            reading = self.probe()
        return reading

    def settled_probe(
        self, *, connections: int = 0, tasks: int | None = None
    ) -> tuple[dict[str, Any], float | None]:
        """Read the probe once the cycle's client is gone and teardown has finished.

        Waits for the registry to fall to *connections*, the ones the scenario
        keeps open across cycles, then re-reads until every live count is 0
        and, given *tasks*, until the loop runs no more tasks than that: a
        task still closing what the client held is teardown that has not
        finished. Returns the last reading either way, with the seconds the
        registry took to fall to *connections*, or ``None`` when it was still
        over after ``CLOSE_TIMEOUT_S``: a count that is still over after the
        timeouts is a leak, not a slow teardown.
        """
        start = time.monotonic()
        reading = self.wait_registered(connections)
        released_after_s = (
            time.monotonic() - start if reading["registered_connections"] <= connections else None
        )
        settle_start = time.monotonic()

        def settling(reading: dict[str, Any]) -> bool:
            return any(reading["live"].values()) or (tasks is not None and reading["tasks"] > tasks)

        while settling(reading) and time.monotonic() - settle_start < SETTLE_TIMEOUT_S:
            time.sleep(0.25)
            reading = self.probe()
        return reading, released_after_s


def _candidates_from(sdp: str) -> list[rw.IceCandidate]:
    """Return the candidates the runtime embedded in its answer, in m-section order."""
    candidates: list[rw.IceCandidate] = []
    mline_index, mid = -1, None
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


class BareClient:
    """One client connection to a running session, without any client SDK.

    Sends a synthetic webcam, receives ``main_video`` and ``main_audio``, and
    closes by dropping its peer connection, which is how a ``reactor_webrtc``
    peer is closed.
    """

    # The requests a cycle sent that were refused or never answered; ``None``
    # for a client that sends none it waits on.
    unanswered: list[str] | None = None
    # Whether the client negotiates and maps a ``webcam`` track to send.
    sends_webcam = True

    def __init__(self, runtime: RuntimeEndpoint, session_id: str) -> None:
        self._runtime = runtime
        self._base = f"/sessions/{session_id}/transport/webrtc"
        self._factory = _get_factory(WebRtcConfig())
        self.received_video = 0
        self._pc: Any = None
        self._camera: Any = None
        self._keep: list[Any] = []
        # Set from a libwebrtc thread once the peer connection is gone.
        self._dropped = threading.Event()

    async def run(self, frame: tuple[bytes, int, int]) -> int:
        """Connect, stream until enough frames came back, and close; return the frames received."""
        try:
            return await self.open(frame)
        finally:
            self.close()

    async def open(self, frame: tuple[bytes, int, int]) -> int:
        """Connect and stream until enough frames came back; return the frames received.

        The connection stays open, for the caller to close or to keep streaming on.
        """
        await self._connect()
        await self._stream_burst(frame)
        await self._settle()
        return self.received_video

    async def stream_for(self, frame: tuple[bytes, int, int], seconds: float) -> None:
        """Stream on for *seconds*."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            await self._send_frame(frame)

    async def wait_for_replies(self) -> None:
        """Wait, as a closing client does, for the replies it is still owed."""
        await self._settle()

    async def stream_while(self, frame: tuple[bytes, int, int], done: asyncio.Event) -> None:
        """Stream on until *done* is set."""
        while not done.is_set():
            await self._send_frame(frame)

    async def stream_until_dropped(self, frame: tuple[bytes, int, int], timeout_s: float) -> bool:
        """Stream on until the runtime drops the connection; return whether it did in time.

        Only webcam frames are sent: a request sent now could not be answered
        before the connection closes, so none is.
        """
        deadline = time.monotonic() + timeout_s
        while not self._dropped.is_set() and time.monotonic() < deadline:
            await self._push_frame(frame)
        return self._dropped.is_set()

    async def run_until_killed(
        self,
        frame: tuple[bytes, int, int],
        ready: Callable[[], None],
        tick: Callable[[], None] | None = None,
    ) -> None:
        """Stream as :meth:`run` does, call *ready*, then stream on until the process dies.

        *tick* runs after every frame streamed after *ready*. Never closes:
        the connection ends only with the process, the way a client that
        crashes or loses power leaves.
        """
        await self.open(frame)
        ready()
        while True:
            await self._send_frame(frame)
            if tick is not None:
                tick()

    async def _stream_burst(self, frame: tuple[bytes, int, int]) -> None:
        deadline = time.monotonic() + _FRAME_WAIT_S
        while self.received_video < FRAMES_PER_CYCLE and time.monotonic() < deadline:
            await self._send_frame(frame)

    async def _send_frame(self, frame: tuple[bytes, int, int]) -> None:
        self._on_tick()
        await self._push_frame(frame)

    async def _push_frame(self, frame: tuple[bytes, int, int]) -> None:
        bgra, width, height = frame
        self._camera.push_video_frame(bgra, width, height)
        await asyncio.sleep(1 / _FPS)

    def close(self) -> None:
        """Close the connection: a ``reactor_webrtc`` peer closes when nothing holds it."""
        self._pc = None
        self._camera = None
        self._keep.clear()

    async def _connect(self) -> None:
        _, registered = self._runtime.request("POST", f"{self._base}/connections")
        cid = registered["connection_id"]
        # Appended from a libwebrtc thread and drained here; a deque's append and
        # popleft are each atomic, so no candidate falls between the two.
        ice: collections.deque[rw.IceCandidate] = collections.deque()
        control_open = threading.Event()

        observer = rw.PeerConnectionObserver()
        observer.on_ice_candidate = ice.append
        observer.on_track = self._on_track
        observer.on_connection_state_change = self._on_state
        pc = self._factory.create_peer_connection(rw.RtcConfiguration(), observer)
        recv_video = pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
        recv_audio = pc.add_transceiver(rw.MediaKind.Audio, rw.TransceiverDirection.RecvOnly)
        send_video = None
        if self.sends_webcam:
            self._camera = self._factory.create_video_track("webcam")
            send_video = pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
            await send_video.set_track(self._camera)
            self._keep.append(send_video)
        data = pc.create_data_channel("data")
        control = pc.create_data_channel("control")
        data.on_message(self._on_data_message)
        control.on_message(self._on_control_message)
        control.on_state_change(lambda state: self._on_control_state(state, control_open))
        self._pc = pc
        self._keep += [observer, recv_video, recv_audio, data, control]

        offer = await pc.create_offer()
        await pc.set_local_description(offer)
        mapping = [
            {
                "mid": recv_video.mid(),
                "name": "main_video",
                "kind": "video",
                "direction": "recvonly",
            },
            {
                "mid": recv_audio.mid(),
                "name": "main_audio",
                "kind": "audio",
                "direction": "recvonly",
            },
        ]
        if send_video is not None:
            mapping.append(
                {
                    "mid": send_video.mid(),
                    "name": "webcam",
                    "kind": "video",
                    "direction": "sendonly",
                }
            )
        self._runtime.request(
            "POST",
            f"{self._base}/connections/{cid}/sdp_params",
            {"sdp_offer": offer.sdp, "track_mapping": mapping},
        )
        deadline = time.monotonic() + 20
        while True:
            status, answer = self._runtime.request(
                "GET", f"{self._base}/connections/{cid}/sdp_params"
            )
            if status == 200:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("the runtime never produced an SDP answer")
            await asyncio.sleep(0.05)
        await pc.set_remote_description(rw.SessionDescription("answer", answer["sdp_answer"]))
        for candidate in _candidates_from(answer["sdp_answer"]):
            await pc.add_ice_candidate(candidate)

        deadline = time.monotonic() + 20
        while not control_open.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("the control channel never opened")
            pending = [ice.popleft() for _ in range(len(ice))]
            if pending:
                self._runtime.request(
                    "POST",
                    f"{self._base}/connections/{cid}/ice_candidates",
                    {
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

        self._start_wire(control, data)

    def _start_wire(self, control: Any, data: Any) -> None:
        """Publish the webcam and resume both output tracks, in v0 JSON."""
        control.send(
            json.dumps(
                {
                    "type": "request",
                    "method": "publish_track",
                    "request_id": "publish-webcam",
                    "data": {"name": "webcam"},
                }
            ).encode(),
            binary=False,
        )
        for name in ("main_video", "main_audio"):
            control.send(
                json.dumps(
                    {"type": "notification", "event": "resume_track", "data": {"name": name}}
                ).encode(),
                binary=False,
            )

    def _on_tick(self) -> None:
        """Run once per webcam frame sent; the v0 client sends nothing more."""

    async def _settle(self) -> None:
        """Wait, before closing, for what the client is still owed; v0 is owed nothing."""

    def _on_data_message(self, data: bytes, binary: bool) -> None:
        """Read a data-channel frame from the runtime; v0 reads none."""

    def _on_control_message(self, data: bytes, binary: bool) -> None:
        """Read a control-channel frame from the runtime; v0 reads none."""

    def _on_control_state(self, state: rw.DataChannelState, control_open: threading.Event) -> None:
        if state == rw.DataChannelState.Open:
            control_open.set()
        # The runtime closes the channels as it closes the connection, ahead
        # of the peer connection's own state change.
        elif state == rw.DataChannelState.Closing or state == rw.DataChannelState.Closed:
            self._dropped.set()

    def _on_state(self, state: rw.PeerConnectionState) -> None:
        # The binding hands each callback a fresh enum instance: compare by value.
        if (
            state == rw.PeerConnectionState.Disconnected
            or state == rw.PeerConnectionState.Failed
            or state == rw.PeerConnectionState.Closed
        ):
            self._dropped.set()

    def _on_track(self, kind: rw.MediaKind, track: rw.Track) -> None:
        self._keep.append(track)
        if kind == rw.MediaKind.Video:
            track.on_video_frame(self._count_video)
        else:
            track.on_audio_frame(lambda *_: None)

    def _count_video(self, *_: Any) -> None:
        self.received_video += 1


# How often the v1 client pings, sends client stats and a command: client
# stats closer than a second apart are dropped by the runtime.
_V1_INTERVAL_S = 1.1
# How long a closing v1 client waits for the replies it is still owed.
_REPLY_WAIT_S = 3.0
_EFFECTS = ("none", "grayscale", "invert")


class V1Client(BareClient):
    """A :class:`BareClient` that speaks the v1 protobuf protocol, as current SDKs do.

    Besides the webcam it requests the model's schema, and once a second it
    pings, sends a client stats batch and a ``set_effect`` or ``set_intensity``
    command, so the churn also exercises the per-connection state that path
    keeps. Every request it sends must be answered without an error before it
    closes; :attr:`unanswered` lists the ones that were not.
    """

    def __init__(self, runtime: RuntimeEndpoint, session_id: str) -> None:
        """Create a client for *session_id*; nothing connects until :meth:`run`."""
        super().__init__(runtime, session_id)
        self._control: Any = None
        self._data: Any = None
        self._lock = threading.Lock()
        # Request id to what it asked for and the reply payloads that answer
        # it, until its reply arrives.
        self._pending: dict[str, tuple[str, frozenset[str | None]]] = {}
        self._failed: list[str] = []
        # Notifications the runtime owes this client, by payload name.
        self._owed_notices: set[str] = set()
        self._next_tick = 0.0
        self._sent = 0
        self.unanswered = []

    def _start_wire(self, control: Any, data: Any) -> None:
        self._control, self._data = control, data
        if self.sends_webcam:
            self._publish_webcam()
        for name in ("main_video", "main_audio"):
            self._send_control(
                control_pb2.ControlClientMessage(
                    kind=_NOTIFICATION, resume_track=track_pb2.ResumeTrack(name=name)
                )
            )
        self._send_control(
            control_pb2.ControlClientMessage(
                request_id="schema", kind=_REQUEST, request_schema=platform_pb2.RequestSchema()
            ),
            asked="request_schema",
            answers=frozenset({"model_schema"}),
        )

    def _publish_webcam(self) -> None:
        self._send_control(
            control_pb2.ControlClientMessage(
                request_id="publish-webcam",
                kind=_REQUEST,
                publish_track=track_pb2.PublishTrack(name="webcam"),
            ),
            asked="publish_track",
            answers=frozenset({"publish_track"}),
        )

    def _on_tick(self) -> None:
        now = time.monotonic()
        if now < self._next_tick:
            return
        self._next_tick = now + _V1_INTERVAL_S
        n = self._sent
        self._sent += 1
        self._send_control(
            control_pb2.ControlClientMessage(kind=_NOTIFICATION, ping=platform_pb2.Ping())
        )
        self._send_control(
            control_pb2.ControlClientMessage(kind=_NOTIFICATION, client_stats=self._client_stats(n))
        )
        args = struct_pb2.Struct()
        if n % 2:
            name = "set_intensity"
            args.update({"intensity": 0.5 if n % 4 == 1 else 1.0})
        else:
            name = "set_effect"
            args.update({"effect": _EFFECTS[(n // 2) % len(_EFFECTS)]})
        request_id = f"command-{n}"
        with self._lock:
            # A command is answered by a bodyless ack, or by the model's reply.
            self._pending[request_id] = (name, frozenset({None, "message"}))
        self._data.send(
            data_pb2.DataClientMessage(
                request_id=request_id,
                kind=_REQUEST,
                command=model_pb2.Command(type=name, data=args),
            ).SerializeToString(),
            binary=True,
        )

    def _client_stats(self, n: int) -> platform_pb2.ClientStats:
        now_ms = int(time.time() * 1000)
        connection = platform_pb2.ClientConnectionStat(
            timestamp=now_ms, metrics={"rtt_ms": 1.0, "available_outgoing_bitrate": 2e6}
        )
        if n == 0:
            connection.environment.CopyFrom(
                platform_pb2.ClientEnvironment(sdk_type="endurance", sdk_version="0")
            )
        track_stats = [
            platform_pb2.ClientTrackStat(
                timestamp=now_ms,
                track_name="main_video",
                kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                direction=platform_pb2.TrackDirection.TRACK_DIRECTION_RECVONLY,
                video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP8,
                metrics={"frames_decoded": float(self.received_video), "jitter": 0.01},
            )
        ]
        if self.sends_webcam:
            track_stats.append(
                platform_pb2.ClientTrackStat(
                    timestamp=now_ms,
                    track_name="webcam",
                    kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                    direction=platform_pb2.TrackDirection.TRACK_DIRECTION_SENDONLY,
                    video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP8,
                    metrics={"frames_encoded": float(n * 30)},
                )
            )
        return platform_pb2.ClientStats(track_stats=track_stats, connection_stat=connection)

    def _send_control(
        self,
        message: control_pb2.ControlClientMessage,
        *,
        asked: str | None = None,
        answers: frozenset[str | None] = frozenset(),
    ) -> None:
        if asked is not None:
            with self._lock:
                self._pending[message.request_id] = (asked, answers)
        self._control.send(message.SerializeToString(), binary=True)

    def close(self) -> None:
        """Close the connection, dropping the channels this client also holds."""
        self._control = self._data = None
        super().close()

    def failures(self) -> list[str]:
        """Return the requests refused or answered wrongly so far, and frames it could not read."""
        with self._lock:
            return list(self._failed)

    def expect_notice(self, payload: str) -> None:
        """Count a control notification with *payload* as owed until it arrives.

        :meth:`collect_unanswered` lists it as unanswered if it never does.
        """
        with self._lock:
            self._owed_notices.add(payload)

    def forget_notice(self, payload: str) -> None:
        """Stop counting the notification *payload* as owed, as when its stop was refused."""
        with self._lock:
            self._owed_notices.discard(payload)

    def collect_unanswered(self) -> list[str]:
        """Return the requests and notices still unanswered, refreshing :attr:`unanswered`."""
        with self._lock:
            self.unanswered = [
                *self._failed,
                *(f"{asked} ({rid}): no reply" for rid, (asked, _) in self._pending.items()),
                *(f"{payload} notice: never arrived" for payload in sorted(self._owed_notices)),
            ]
            return self.unanswered

    async def wait_for_notices(self, timeout_s: float) -> None:
        """Wait up to *timeout_s* for every owed notification to arrive."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._owed_notices:
                    return
            await asyncio.sleep(0.05)

    def _on_control_message(self, data: bytes, binary: bool) -> None:
        message = control_pb2.ControlServerMessage()
        if not self._parse(message, data, "control"):
            return
        payload = message.WhichOneof("payload")
        if message.kind == _NOTIFICATION:
            with self._lock:
                self._owed_notices.discard(payload or "")
            return
        self._answered(message.request_id, payload, message)

    def _on_data_message(self, data: bytes, binary: bool) -> None:
        message = data_pb2.DataServerMessage()
        if self._parse(message, data, "data") and message.kind == _RESPONSE:
            self._answered(message.request_id, message.WhichOneof("payload"), message)

    def _parse(self, message: Any, data: bytes, channel: str) -> bool:
        # Called on a libwebrtc thread: an exception here would go nowhere, so
        # a frame that is not v1 is recorded as a failure instead.
        try:
            message.ParseFromString(bytes(data))
        except DecodeError:
            with self._lock:
                self._failed.append(f"an undecodable frame on {channel}")
            return False
        return True

    def _answered(self, request_id: str, payload: str | None, message: Any) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                return
            asked, answers = pending
            if payload == "error":
                self._failed.append(f"{asked} ({request_id}): {message.error.code}")
            elif payload not in answers:
                self._failed.append(f"{asked} ({request_id}): answered with {payload or 'nothing'}")

    async def _settle(self) -> None:
        deadline = time.monotonic() + _REPLY_WAIT_S
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    break
            await asyncio.sleep(0.05)
        self.collect_unanswered()


class Viewer(V1Client):
    """A :class:`V1Client` that watches without publishing: it sends no webcam.

    It joins a session another client publishes to and counts the output that
    comes back, as a second participant in a shared session does. It
    negotiates only the output tracks, receive-only, and reports stats for
    them alone, as a client that only watches does.
    """

    sends_webcam = False

    async def _send_frame(self, frame: tuple[bytes, int, int]) -> None:
        self._on_tick()
        await asyncio.sleep(1 / _FPS)


# Steps a stepped client requests, one at a time, before it disconnects.
STEPS_PER_CYCLE = 10
# How long a stepped client waits for one step's result.
_STEP_WAIT_S = 10.0


class StepClient(BareClient):
    """A client of a stepped session, speaking v1 as current SDKs do.

    It requests :data:`STEPS_PER_CYCLE` steps one after another, each setting
    the model's effect through ``state`` and carrying one small webcam frame in
    ``media_input``, waits for each step's result, and closes. A stepped
    session streams nothing and drops live input, so nothing else is sent.
    :attr:`unanswered` lists the requests that got no result or a failed one.
    """

    def __init__(self, runtime: RuntimeProcess, session_id: str) -> None:
        """Create a client for *session_id*; nothing connects until :meth:`run`."""
        super().__init__(runtime, session_id)
        self._data: Any = None
        self._control: Any = None
        self._lock = threading.Lock()
        self._results: dict[str, step_pb2.StepResult] = {}
        self._failed: list[str] = []
        self.unanswered: list[str] = []

    async def run(self, frame: tuple[bytes, int, int]) -> int:
        """Connect, request every step in turn, and close; return the steps that ran."""
        await self._connect()
        ran = 0
        unanswered: list[str] = []
        try:
            webcam = _raw_frame(np.full((48, 64, 3), 128, dtype=np.uint8))
            for n in range(STEPS_PER_CYCLE):
                request_id = f"step-{n}"
                state = struct_pb2.Struct()
                state.update({"effect": _EFFECTS[n % len(_EFFECTS)]})
                self._data.send(
                    data_pb2.DataClientMessage(
                        request_id=request_id,
                        kind=_REQUEST,
                        request_step=step_pb2.RequestStep(
                            state=state,
                            media_input=[step_pb2.TrackFrames(track="webcam", frames=[webcam])],
                        ),
                    ).SerializeToString(),
                    binary=True,
                )
                result = await self._result(request_id)
                if result is None:
                    unanswered.append(f"request_step ({request_id}): no reply")
                elif result.failure_code:
                    unanswered.append(f"request_step ({request_id}): {result.error}")
                else:
                    ran += 1
            with self._lock:
                self.unanswered = [*unanswered, *self._failed]
            return ran
        finally:
            self.close()

    async def _result(self, request_id: str) -> step_pb2.StepResult | None:
        deadline = time.monotonic() + _STEP_WAIT_S
        while time.monotonic() < deadline:
            with self._lock:
                if request_id in self._results:
                    return self._results.pop(request_id)
            await asyncio.sleep(0.01)
        return None

    def _start_wire(self, control: Any, data: Any) -> None:
        # The control channel stays open beside the data channel, as an SDK's does.
        self._control, self._data = control, data

    def close(self) -> None:
        """Close the connection, dropping the channels this client also holds."""
        self._control = self._data = None
        super().close()

    def _on_data_message(self, data: bytes, binary: bool) -> None:
        message = data_pb2.DataServerMessage()
        try:
            message.ParseFromString(bytes(data))
        except DecodeError:
            with self._lock:
                self._failed.append("an undecodable frame on data")
            return
        if message.WhichOneof("payload") == "step_result":
            with self._lock:
                self._results[message.request_id] = message.step_result


def _raw_frame(rgb: np.ndarray) -> step_pb2.Frame:
    """Return *rgb* as one raw wire frame."""
    return step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_RAW,
        dtype="uint8",
        shape=list(rgb.shape),
        data=rgb.tobytes(),
    )


def assert_full_streams(
    received: list[int], *, expected: int = FRAMES_PER_CYCLE, unit: str = "frames"
) -> MetricResult:
    """Fail when any cycle got fewer than *expected* frames, or step results, back.

    A cycle that did not stream its full burst did not exercise what the
    scenario claims to, so a run with one is not a pass.

    Args:
        received: What each cycle got back, in cycle order.
        expected: What every cycle must get back.
        unit: What is counted, for the report.
    """
    short = [(cycle, n) for cycle, n in enumerate(received) if n < expected]
    if short:
        cycle, n = short[0]
        reason = (
            f"fewer than {expected} {unit} came back on {len(short)}/{len(received)} "
            f"cycles (first at cycle {cycle}: {n})"
        )
        print(f"[short_cycles] FAILED: {reason}")
        return MetricResult(
            name="short_cycles",
            start=0,
            end=len(short),
            change=len(short),
            unit="count",
            status="fail",
            detail=reason,
            threshold="always 0",
            first_bad_cycle=cycle,
        )
    reason = f"every one of the {len(received)} cycles got its {expected} {unit} back"
    print(f"[short_cycles] ok: {reason}")
    return MetricResult(
        name="short_cycles",
        start=0,
        end=0,
        change=0,
        unit="count",
        status="ok",
        detail=reason,
        threshold="always 0",
    )


def assert_answered(unanswered: list[list[str]]) -> MetricResult:
    """Fail when any cycle's client was refused or left without a reply.

    A v1 cycle whose publish, schema request or commands went unanswered did
    not exercise the path it claims to, so a run with one is not a pass.

    Args:
        unanswered: Each cycle's unanswered or refused requests, in cycle order.
    """
    return _assert_no_misses(
        unanswered,
        name="unanswered_requests",
        failed="requests went unanswered or were refused",
        passed="every request of the {n} cycles was answered",
    )


def assert_clean_stops(failures: list[list[str]]) -> MetricResult:
    """Fail when any cycle's session stop went wrong.

    A stop that was refused, a client the stop did not drop, a session that
    was not orphaned when an orphaned stop came, or one not ready again
    after it: each fails the cycle it happened on.

    Args:
        failures: Each cycle's stop failures, in cycle order.
    """
    return _assert_no_misses(
        failures,
        name="failed_stops",
        failed="the session stop went wrong",
        passed="every one of the {n} stops went through",
    )


def _assert_no_misses(
    misses_by_cycle: list[list[str]], *, name: str, failed: str, passed: str
) -> MetricResult:
    bad = [(cycle, misses) for cycle, misses in enumerate(misses_by_cycle) if misses]
    if bad:
        cycle, misses = bad[0]
        reason = (
            f"{failed} on {len(bad)}/{len(misses_by_cycle)} cycles "
            f"(first at cycle {cycle}: {'; '.join(misses[:3])})"
        )
        print(f"[{name}] FAILED: {reason}")
        return MetricResult(
            name=name,
            start=0,
            end=len(bad),
            change=len(bad),
            unit="count",
            status="fail",
            detail=reason,
            threshold="always 0",
            first_bad_cycle=cycle,
        )
    reason = passed.format(n=len(misses_by_cycle))
    print(f"[{name}] ok: {reason}")
    return MetricResult(
        name=name,
        start=0,
        end=0,
        change=0,
        unit="count",
        status="ok",
        detail=reason,
        threshold="always 0",
    )


def assert_released(released: list[float | None]) -> MetricResult:
    """Fail when a cycle's connection was still registered ``CLOSE_TIMEOUT_S`` after it left.

    The detail reports how long the runtime took to drop the connections it
    did drop, which is how fast it notices a client that is gone.

    Args:
        released: Each cycle's seconds until the runtime's registry emptied,
            or ``None`` when it never did, in cycle order.
    """
    stuck = [cycle for cycle, seconds in enumerate(released) if seconds is None]
    times = sorted(seconds for seconds in released if seconds is not None)
    timing = (
        f"dropped in {statistics.median(times):.1f}s median, {times[-1]:.1f}s at worst"
        if times
        else ""
    )
    threshold = f"every cycle's connections dropped within {CLOSE_TIMEOUT_S:g}s"
    if stuck:
        reason = (
            f"a connection was still registered {CLOSE_TIMEOUT_S:g}s after its client left "
            f"on {len(stuck)}/{len(released)} cycles (first at cycle {stuck[0]})"
        )
        if times:
            reason += f"; the other cycles' connections {timing}"
        print(f"[unreleased_connections] FAILED: {reason}")
        return MetricResult(
            name="unreleased_connections",
            start=0,
            end=len(stuck),
            change=len(stuck),
            unit="count",
            status="fail",
            detail=reason,
            threshold=threshold,
            first_bad_cycle=stuck[0],
        )
    reason = f"the connections of all {len(released)} cycles {timing}"
    print(f"[unreleased_connections] ok: {reason}")
    return MetricResult(
        name="unreleased_connections",
        start=0,
        end=0,
        change=0,
        unit="count",
        status="ok",
        detail=reason,
        threshold=threshold,
    )


def gray_frame(width: int = 640, height: int = 480) -> tuple[bytes, int, int]:
    """Return a solid mid-gray webcam frame, as BGRA."""
    return rgb_to_bgra(np.full((height, width, 3), 128, dtype=np.uint8))


@dataclass
class _Baseline:
    freed_by_gc: int
    holding_after_stop: int


def reap(processes: list[psutil.Process]) -> None:
    """Kill *processes*, a runtime's descendants, leaving its shared memory unlinked.

    The workers go first, and are waited for. Python's multiprocessing
    resource tracker, one of the descendants, unlinks the shared-memory
    blocks the workers registered once it sees them exit, so it gets
    ``STOP_TIMEOUT_S`` to finish on its own before anything still alive is
    killed: killed alongside the workers, it would leave their blocks behind.
    """
    trackers = [p for p in processes if _is_resource_tracker(p)]
    workers = [p for p in processes if p not in trackers]
    _kill_and_wait(workers)
    _, alive = psutil.wait_procs(trackers, timeout=STOP_TIMEOUT_S)
    _kill_and_wait(alive)


def _is_resource_tracker(process: psutil.Process) -> bool:
    try:
        return any("resource_tracker" in part for part in process.cmdline())
    except psutil.Error:
        return False


def _kill_and_wait(processes: list[psutil.Process]) -> None:
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(processes, timeout=STOP_TIMEOUT_S)


def shm_usage() -> tuple[int, int] | None:
    """Return the distributed package's shared-memory blocks and their bytes.

    ``None`` on a platform with no ``/dev/shm``, where the blocks are not
    files to count.
    """
    if not SHM_DIR.is_dir():
        return None
    blocks = size = 0
    for path in SHM_DIR.glob(_SHM_BLOCKS):
        # A block can be unlinked between the listing and the stat.
        try:
            size += path.stat().st_size
        except FileNotFoundError:
            continue
        blocks += 1
    return blocks, size


def _children(process: psutil.Process) -> tuple[int, int]:
    """Return how many processes *process* started, at any depth, and their summed RSS.

    A child that exited but was never reaped still counts, with no memory:
    leaving one behind is a leak too.
    """
    count = rss = 0
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.ZombieProcess:
            pass
        # A child can exit between the listing and the read.
        except psutil.NoSuchProcess:
            continue
        count += 1
    return count, rss


class ResourceSampler:
    """Takes one ``Sample`` of the runtime process per settled cycle."""

    def __init__(
        self, runtime: RuntimeProcess, *, duration_s: float = ENDURANCE_DURATION_SECONDS
    ) -> None:
        self._runtime = runtime
        self._process = runtime.process
        self._start = time.monotonic()
        self._duration_s = duration_s
        self.samples: list[Sample] = []
        self._process.cpu_percent(interval=None)
        first = runtime.probe()
        # The session's tasks before any client: a task the first connection
        # leaves behind must count against this, not hide in the first sample.
        self.initial_tasks: int = first["tasks"]
        # The model's worker processes are up by now: one more is a leak.
        self.initial_children: int = _children(self._process)[0]
        self._baseline = _Baseline(
            freed_by_gc=first["batches_freed_by_gc"],
            holding_after_stop=first["pacers_holding_after_stop"],
        )

    def deadline_reached(self) -> bool:
        """Whether the run's duration has passed."""
        return time.monotonic() - self._start >= self._duration_s

    def sample(self, *, cycle: int, reading: dict[str, Any], record: bool = True) -> Sample:
        """Take the runtime process and a settled probe *reading* as one sample.

        The sample joins :attr:`samples`, which the trends and the reports
        read, unless *record* is false: a reading that is not a cycle of the
        run, such as the one after teardown, is only returned.
        """
        freed = reading["batches_freed_by_gc"]
        holding = reading["pacers_holding_after_stop"]
        children, children_rss = _children(self._process)
        shm = shm_usage()
        with self._process.oneshot():
            cpu = self._process.cpu_times()
            sample = Sample(
                cycle=cycle,
                elapsed_s=time.monotonic() - self._start,
                rss_bytes=self._process.memory_info().rss,
                cpu_s=cpu.user + cpu.system,
                cpu_percent=self._process.cpu_percent(interval=None),
                num_threads=self._process.num_threads(),
                num_fds=self._process.num_fds(),
                # The same pacer can show in both: once when it stopped, again
                # while it is still alive at the probe.
                pacers_holding_output=max(
                    holding - self._baseline.holding_after_stop,
                    reading["stopped_pacers_holding_now"],
                ),
                pinned_batches=freed - self._baseline.freed_by_gc,
                live_connections=reading["live"]["connections"],
                live_peers=reading["live"]["peers"],
                live_pacers=reading["live"]["pacers"],
                live_client_stats=reading["client_stats_connections"],
                tasks=reading["tasks"],
                live_participants=reading["live"].get("participants", 0),
                child_processes=children,
                child_rss_bytes=children_rss,
                shm_blocks=shm[0] if shm else None,
                shm_bytes=shm[1] if shm else None,
                executor_threads=reading.get("executor_threads", 0),
                live_step_requests=reading.get("step_requests", 0),
            )
        self._baseline.freed_by_gc = freed
        self._baseline.holding_after_stop = holding
        if record:
            self.samples.append(sample)
        return sample
