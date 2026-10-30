"""The session's connection to its room: a system connection that carries the output.

:class:`RoomConnection` is the :class:`~reactor_runtime.core.transport.Connection`
the LiveKit transport opens once a join succeeds, on a connection id minted for
the ``livekit`` transport. It is a system connection: it occupies the session,
and the model sees it as a client whose ``system`` is true. It carries the
session's output: every output track is published to the room as a LiveKit
track, and the model's media reaches it through a :class:`RoomPacer`. The room
gets one copy of the output, whoever subscribes.

It drops the messages sent to it: participants are clients of their own and
receive messages on their own connections.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

import numpy as np
from livekit import rtc

from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnectionOrigin,
    ConnId,
    MediaBundle,
    MediaChunk,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.livekit.pacer import RoomPacer
from reactor_runtime.transport.livekit.version import TRANSPORT_NAME
from reactor_runtime.transport.webrtc.frames import to_int16_mono

logger = logging.getLogger(__name__)

# How much audio each output track's AudioSource may queue ahead of the room.
# The pacer hands audio over in real time, beside its video frame, so the queue
# only absorbs scheduling jitter; the bound caps how far audio can fall behind
# its video if the room stops draining it. Every flush clears it.
AUDIO_QUEUE_MS = 1000

# The audio frames the pacer has handed over and the room has not taken yet,
# per track. Bounded so a room that stops draining cannot grow it.
_AUDIO_PUMP_MAX = 64

# The size a VideoSource starts at, until the first frame sets its own.
_INITIAL_VIDEO_SIZE = (1280, 720)


@dataclass
class _AudioOut:
    """One output audio track: its source, its rate, and the pump that feeds it."""

    source: rtc.AudioSource
    sample_rate: int
    pending: asyncio.Queue[rtc.AudioFrame]
    pump: asyncio.Task[None] | None = None


class RoomConnection:
    """The session's system connection to a LiveKit room.

    Built by :class:`~reactor_runtime.transport.livekit.room.RoomSession` once
    the join succeeds and the output tracks are published. The pacer starts
    with the connection, so the model's first frame is the first frame the
    room sees.
    """

    origin = ConnectionOrigin(transport=TRANSPORT_NAME, system=True)

    def __init__(
        self,
        conn_id: ConnId,
        loop: asyncio.AbstractEventLoop,
        video: Mapping[str, rtc.VideoSource],
        audio: Mapping[str, rtc.AudioSource],
        tracks: Mapping[str, TrackInfo],
        on_close: Callable[[], Awaitable[None]],
        on_input_freed: Callable[[str], None] | None = None,
    ) -> None:
        """Bind the connection to the sources of its published tracks.

        Args:
            conn_id: The id minted for the connection.
            loop: The loop the room runs on, which an audio source is fed on.
            video: The video source of each published output video track.
            audio: The audio source of each published output audio track.
            tracks: The model's output tracks, by name.
            on_close: Leaves the room. Awaited once, when the connection
                closes, so a session that ends leaves its room.
            on_input_freed: Takes the name of each input track the session
                frees, so the room can start a participant it held back.
        """
        self.id = conn_id
        self._on_close = on_close
        self._on_input_freed = on_input_freed
        self.capabilities = ConnectionCapabilities(
            carries_video=bool(video), carries_audio=bool(audio)
        )
        self._loop = loop
        self._video = dict(video)
        self._audio = {
            name: _AudioOut(
                source=source,
                sample_rate=int(tracks[name].rate),
                pending=asyncio.Queue(maxsize=_AUDIO_PUMP_MAX),
            )
            for name, source in audio.items()
        }
        initial_fps = max((t.rate for t in tracks.values() if t.kind is TrackKind.VIDEO), default=0)
        self._pacer = RoomPacer(
            {name: tracks[name] for name in video},
            self._send_frame,
            fps=initial_fps or 30.0,
        )
        self._dropped_audio = 0
        self._closed = False

    def start(self) -> None:
        """Start pacing the model's media into the room."""
        for name, out in self._audio.items():
            out.pump = self._loop.create_task(self._pump_audio(out), name=f"room-audio-{name}")
        self._pacer.start()

    @property
    def protocol_version(self) -> ProtocolVersion:
        """The codec a message to the room connection is encoded in, before it is dropped."""
        return ProtocolVersion.V1

    def send_message(self, payload: bytes | str) -> None:
        """Drop a data frame: participants receive messages on their own connections."""

    def send_control(self, payload: bytes | str) -> None:
        """Drop a control frame."""

    @property
    def max_message_size(self) -> int:
        """Accept any size: nothing sent to the room connection leaves the process."""
        return 0x7FFF_FFFF_FFFF_FFFF

    @property
    def message_chunking(self) -> bool:
        """Report no chunking; the room connection never requests a step."""
        return False

    async def send_step_result(self, payload: bytes) -> bool:
        """Drop a step result."""
        return True

    def send_media(self, chunk: MediaChunk) -> None:
        """Submit a media chunk to the pacer, which may wait for queue room."""
        self._pacer.submit(chunk)

    def input_track_freed(self, name: str) -> None:
        """Hand on that input track *name* is free, for a participant waiting on it."""
        if self._on_input_freed is not None and not self._closed:
            self._on_input_freed(name)

    def flush_media(self, *, hold: bool = False) -> None:
        """Drop the queued media, the audio waiting for the room included.

        Each audio source's own queue is cleared, so the sound stops at the
        cut rather than up to :data:`AUDIO_QUEUE_MS` later.
        """
        self._pacer.flush(hold=hold)
        self._loop.call_soon_threadsafe(self._clear_audio)

    def set_media_rate(self, fps: float) -> None:
        """Re-pace the queued media at *fps* immediately."""
        self._pacer.set_rate(fps)

    def set_media_depth(self, depth: int) -> None:
        """Bound how many frames may queue between the model and the room."""
        self._pacer.set_depth(depth)

    def resume_track(self, name: str) -> None:
        """Do nothing: the room's server manages who receives each track."""

    def pause_track(self, name: str) -> None:
        """Do nothing: the room's server manages who receives each track."""

    async def close(self) -> None:
        """Stop pacing and feeding the room, then leave it. Idempotent.

        Once the pacer has stopped, queued frames are dropped, and so is the
        audio waiting for each source, including what the source itself
        queues, up to :data:`AUDIO_QUEUE_MS`; at most the one audio frame a
        source is already taking can still reach the room. The room is left
        even when the pacer fails to stop in time or the audio cannot be
        cleared, so a connection that closes never stays in the room with a
        frozen picture.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self._pacer.stop)
        except RuntimeError:
            logger.warning("the room pacer did not stop in time", exc_info=True)
        finally:
            for out in self._audio.values():
                if out.pump is not None:
                    out.pump.cancel()
            self._clear_audio()
            await self._on_close()

    def _send_frame(self, bundle: MediaBundle) -> None:
        """Hand one paced frame to the room, on the pacer's thread."""
        for name, data in bundle.tracks.items():
            key = name if name in self._video or name in self._audio else data.info.name
            if data.info.kind is TrackKind.VIDEO:
                source = self._video.get(key)
                if source is not None:
                    source.capture_frame(_video_frame(data.data))
            else:
                out = self._audio.get(key)
                samples = to_int16_mono(data.data)
                if out is not None and samples.size:
                    frame = rtc.AudioFrame(samples.tobytes(), out.sample_rate, 1, samples.size)
                    self._loop.call_soon_threadsafe(self._queue_audio, out, frame)

    def _queue_audio(self, out: _AudioOut, frame: rtc.AudioFrame) -> None:
        """Queue one frame for its track's pump, dropping the oldest when full."""
        if out.pending.full():
            out.pending.get_nowait()
            self._dropped_audio += 1
        out.pending.put_nowait(frame)

    def _clear_audio(self) -> None:
        """Drop the audio waiting for each source and the audio each source queues.

        Each source is cleared on its own: one whose queue cannot be cleared is
        logged and does not keep the other tracks playing.
        """
        for name, out in self._audio.items():
            while not out.pending.empty():
                out.pending.get_nowait()
            try:
                out.source.clear_queue()
            except Exception:
                logger.warning("clearing the queue of audio track %s failed", name, exc_info=True)

    async def _pump_audio(self, out: _AudioOut) -> None:
        """Feed one audio source in order, waiting for its queue to have room.

        A frame the source refuses is dropped and the pump goes on, so one
        failure does not silence the track for the rest of the connection.
        The first failure is logged with its traceback, later ones quietly.
        """
        failures = 0
        while True:
            try:
                frame = await out.pending.get()
            except asyncio.CancelledError:
                return
            try:
                await out.source.capture_frame(frame)
            except asyncio.CancelledError:
                return
            except Exception:
                failures += 1
                self._dropped_audio += 1
                if failures == 1:
                    logger.exception("feeding an audio frame to the room failed; dropping it")
                else:
                    logger.debug("feeding an audio frame to the room failed; dropping it")


def _video_frame(data: np.ndarray) -> rtc.VideoFrame:
    """Wrap one ``(H, W, 3)`` RGB frame, or the last frame of a batch, for a VideoSource."""
    frame = data[-1] if data.ndim == 4 else data
    height, width = int(frame.shape[0]), int(frame.shape[1])
    return rtc.VideoFrame(
        width, height, rtc.VideoBufferType.RGB24, np.ascontiguousarray(frame).tobytes()
    )
