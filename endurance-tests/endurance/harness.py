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
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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
    track_pb2,
)

_REQUEST = common_pb2.MessageKind.MESSAGE_KIND_REQUEST
_RESPONSE = common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
_NOTIFICATION = common_pb2.MessageKind.MESSAGE_KIND_NOTIFICATION

SUITE_DIR = Path(__file__).resolve().parent.parent

# How long a closed connection may take to leave the runtime's registry. The
# runtime notices a client that closed its peer connection only some seconds
# later, so this is generous; a connection still registered after it is
# reported, not waited on forever.
CLOSE_TIMEOUT_S = 30.0
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


class RuntimeProcess:
    """The runtime under test, in its own process, with the probe route mounted."""

    def __init__(self, log_path: Path) -> None:
        """Start the runtime on a free local port, its output going to *log_path*."""
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {
            **os.environ,
            "PORT": str(self.port),
            "HOST": "127.0.0.1",
            "REACTOR_LOG_LEVEL": os.environ.get("REACTOR_LOG_LEVEL", "WARNING"),
        }
        self._log = log_path.open("w")
        self._proc = subprocess.Popen(
            [sys.executable, str(SUITE_DIR / "endurance" / "launcher.py")],
            cwd=SUITE_DIR / "model",
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self.process = psutil.Process(self._proc.pid)

    def wait_ready(self, timeout_s: float = 60.0) -> dict[str, Any]:
        """Wait for the runtime to answer ``/session``, and return the answer.

        Raises:
            RuntimeError: The runtime exited while starting.
            TimeoutError: It did not answer within *timeout_s*.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"the runtime exited with {self._proc.returncode} on startup")
            try:
                return self.get("/session")
            # An answer that is an error is a broken runtime, not a slow one.
            except urllib.error.HTTPError:
                raise
            # The server accepts before the runtime can answer, so a read that
            # times out is as much "not ready yet" as a refused connection.
            except OSError:
                time.sleep(0.25)
        raise TimeoutError("the runtime never answered /session")

    def stop(self) -> None:
        """Stop the runtime, killing it if it does not exit in time."""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log.close()

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

    def probe(self) -> dict[str, Any]:
        """Return one reading of the probe route."""
        return self.get("/_endurance/probe")

    def settled_probe(self) -> tuple[dict[str, Any], float]:
        """Read the probe once nothing is connected and teardown has finished.

        Waits for the registry to empty, then re-reads until every live count is
        0. Returns the last reading either way, with the seconds it took: a count
        that is still nonzero after the timeouts is a leak, not a slow teardown.
        """
        start = time.monotonic()
        reading = self.probe()
        while reading["registered_connections"] > 0 and time.monotonic() - start < CLOSE_TIMEOUT_S:
            time.sleep(0.25)
            reading = self.probe()
        settle_start = time.monotonic()
        while any(reading["live"].values()) and time.monotonic() - settle_start < SETTLE_TIMEOUT_S:
            time.sleep(0.25)
            reading = self.probe()
        return reading, time.monotonic() - start


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

    def __init__(self, runtime: RuntimeProcess, session_id: str) -> None:
        self._runtime = runtime
        self._base = f"/sessions/{session_id}/transport/webrtc"
        self._factory = _get_factory(WebRtcConfig())
        self.received_video = 0
        self._pc: Any = None
        self._camera: Any = None
        self._keep: list[Any] = []

    async def run(self, frame: tuple[bytes, int, int]) -> int:
        """Connect, stream until enough frames came back, and close; return the frames received."""
        await self._connect()
        try:
            bgra, width, height = frame
            deadline = time.monotonic() + _FRAME_WAIT_S
            while self.received_video < FRAMES_PER_CYCLE and time.monotonic() < deadline:
                self._camera.push_video_frame(bgra, width, height)
                self._on_tick()
                await asyncio.sleep(1 / _FPS)
            await self._settle()
            return self.received_video
        finally:
            self.close()

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
        pc = self._factory.create_peer_connection(rw.RtcConfiguration(), observer)
        recv_video = pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.RecvOnly)
        recv_audio = pc.add_transceiver(rw.MediaKind.Audio, rw.TransceiverDirection.RecvOnly)
        camera = self._factory.create_video_track("webcam")
        send_video = pc.add_transceiver(rw.MediaKind.Video, rw.TransceiverDirection.SendOnly)
        await send_video.set_track(camera)
        data = pc.create_data_channel("data")
        control = pc.create_data_channel("control")
        data.on_message(self._on_data_message)
        control.on_message(self._on_control_message)
        control.on_state_change(
            lambda state: control_open.set() if state == rw.DataChannelState.Open else None
        )
        self._pc, self._camera = pc, camera
        self._keep += [observer, recv_video, recv_audio, send_video, data, control]

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
            {"mid": send_video.mid(), "name": "webcam", "kind": "video", "direction": "sendonly"},
        ]
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

    def __init__(self, runtime: RuntimeProcess, session_id: str) -> None:
        """Create a client for *session_id*; nothing connects until :meth:`run`."""
        super().__init__(runtime, session_id)
        self._control: Any = None
        self._data: Any = None
        self._lock = threading.Lock()
        # Request id to what it asked for, until its reply arrives.
        self._pending: dict[str, str] = {}
        self._failed: list[str] = []
        self._next_tick = 0.0
        self._sent = 0
        self.unanswered: list[str] = []

    def _start_wire(self, control: Any, data: Any) -> None:
        self._control, self._data = control, data
        self._send_control(
            control_pb2.ControlClientMessage(
                request_id="publish-webcam",
                kind=_REQUEST,
                publish_track=track_pb2.PublishTrack(name="webcam"),
            ),
            expects="publish_track",
        )
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
            expects="model_schema",
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
            self._pending[request_id] = name
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
        return platform_pb2.ClientStats(
            track_stats=[
                platform_pb2.ClientTrackStat(
                    timestamp=now_ms,
                    track_name="main_video",
                    kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                    direction=platform_pb2.TrackDirection.TRACK_DIRECTION_RECVONLY,
                    video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP8,
                    metrics={"frames_decoded": float(self.received_video), "jitter": 0.01},
                ),
                platform_pb2.ClientTrackStat(
                    timestamp=now_ms,
                    track_name="webcam",
                    kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                    direction=platform_pb2.TrackDirection.TRACK_DIRECTION_SENDONLY,
                    video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP8,
                    metrics={"frames_encoded": float(n * 30)},
                ),
            ],
            connection_stat=connection,
        )

    def _send_control(
        self, message: control_pb2.ControlClientMessage, *, expects: str | None = None
    ) -> None:
        if expects is not None:
            with self._lock:
                self._pending[message.request_id] = expects
        self._control.send(message.SerializeToString(), binary=True)

    def close(self) -> None:
        """Close the connection, dropping the channels this client also holds."""
        self._control = self._data = None
        super().close()

    def _on_control_message(self, data: bytes, binary: bool) -> None:
        message = control_pb2.ControlServerMessage()
        if self._parse(message, data, "control"):
            self._answered(message.request_id, message.WhichOneof("payload"), message)

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
            asked = self._pending.pop(request_id, None)
            if asked is None:
                return
            if payload == "error":
                self._failed.append(f"{asked} ({request_id}): {message.error.code}")

    async def _settle(self) -> None:
        deadline = time.monotonic() + _REPLY_WAIT_S
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    break
            await asyncio.sleep(0.05)
        with self._lock:
            self.unanswered = [
                *self._failed,
                *(f"{asked} ({rid}): no reply" for rid, asked in self._pending.items()),
            ]


def assert_full_streams(received: list[int]) -> MetricResult:
    """Fail when any cycle got fewer than ``FRAMES_PER_CYCLE`` frames back.

    A cycle that did not stream its full burst did not exercise what the
    scenario claims to, so a run with one is not a pass.

    Args:
        received: The frames each cycle got back, in cycle order.
    """
    short = [(cycle, n) for cycle, n in enumerate(received) if n < FRAMES_PER_CYCLE]
    if short:
        cycle, n = short[0]
        reason = (
            f"fewer than {FRAMES_PER_CYCLE} frames came back on {len(short)}/{len(received)} "
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
    reason = f"every one of the {len(received)} cycles got its {FRAMES_PER_CYCLE} frames back"
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
    bad = [(cycle, misses) for cycle, misses in enumerate(unanswered) if misses]
    if bad:
        cycle, misses = bad[0]
        reason = (
            f"requests went unanswered or were refused on {len(bad)}/{len(unanswered)} cycles "
            f"(first at cycle {cycle}: {'; '.join(misses[:3])})"
        )
        print(f"[unanswered_requests] FAILED: {reason}")
        return MetricResult(
            name="unanswered_requests",
            start=0,
            end=len(bad),
            change=len(bad),
            unit="count",
            status="fail",
            detail=reason,
            threshold="always 0",
            first_bad_cycle=cycle,
        )
    reason = f"every request of the {len(unanswered)} cycles was answered"
    print(f"[unanswered_requests] ok: {reason}")
    return MetricResult(
        name="unanswered_requests",
        start=0,
        end=0,
        change=0,
        unit="count",
        status="ok",
        detail=reason,
        threshold="always 0",
    )


def gray_frame(width: int = 640, height: int = 480) -> tuple[bytes, int, int]:
    """Return a solid mid-gray webcam frame, as BGRA."""
    return rgb_to_bgra(np.full((height, width, 3), 128, dtype=np.uint8))


@dataclass
class _Baseline:
    freed_by_gc: int
    holding_after_stop: int


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
        self._baseline = _Baseline(
            freed_by_gc=first["batches_freed_by_gc"],
            holding_after_stop=first["pacers_holding_after_stop"],
        )

    def deadline_reached(self) -> bool:
        """Whether the run's duration has passed."""
        return time.monotonic() - self._start >= self._duration_s

    def sample(self, *, cycle: int, reading: dict[str, Any]) -> Sample:
        """Record the runtime process and a settled probe *reading* as one sample."""
        freed = reading["batches_freed_by_gc"]
        holding = reading["pacers_holding_after_stop"]
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
            )
        self._baseline.freed_by_gc = freed
        self._baseline.holding_after_stop = holding
        self.samples.append(sample)
        return sample
