"""The local recorder.

A standalone runtime records the model's output to local disk and serves the
clips straight back over HTTP, with no object store in the loop. The recorder
taps the runtime's media fan-out through :meth:`Recorder.on_chunk`, encodes the
frames into fMP4 HLS segments under a per-recording directory, and answers the
clip-manifest math behind ``GET /clips``.

A clip request resolves immediately to a marker range and a path-only playlist
URL the client polls; the bytes for the boundary segment may still be in flight.
Separately, once that boundary segment has actually landed on disk the recorder
notifies an external consumer (the runner journals it on ``/events``), so a
director learns a clip is genuinely fetchable rather than merely requested.
"""

from __future__ import annotations

import contextlib
import math
import queue
import re
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import numpy.typing as npt

from reactor_runtime.core import MediaBundle, MediaChunk, RecordingConfig, TrackKind
from reactor_runtime.log import get_logger
from reactor_runtime.recording.chunk_encoder import ChunkEncoder
from reactor_runtime.recording.markers import MarkerBookkeeper

logger = get_logger(__name__)

# The fixed frame rate the recorded media is encoded at. A chunk's frames are
# resampled onto this grid from the chunk's own rate, so the recorded PTS derives
# from the model's declared cadence rather than the wall-clock arrival time — a
# model that runs slower or faster than real time still records at true duration.
RECORDING_FPS = 30

_INIT_FILENAME = "init.mp4"
# Written into a recording's directory once it is finished, so its final segment
# (which has no successor to prove it closed) is recognised as fetchable.
_COMPLETE_MARKER = ".complete"
_HLS_MEDIA_TYPE = "application/vnd.apple.mpegurl"
# A recording id is a lowercase UUID. Validating a URL path segment against this
# shape before joining it onto the recordings root closes the path-traversal
# vector on ``/clips`` and ``/clips/chunks/{id}/{file}``.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# init plus the zero-padded fMP4 segments the encoder produces; anything else is
# not a recording artifact.
_CHUNK_FILENAME_RE = re.compile(r"^(init\.mp4|chunk_\d{5}\.m4s)$")

_AudioArray = npt.NDArray[Any]
_FeedItem = tuple[npt.NDArray[Any], _AudioArray | None]


class RecorderError(Exception):
    """Base for a clip request the recorder cannot serve."""


class RecorderDisabledError(RecorderError):
    """Recording is off, was never started, or the encoder crashed."""


class NoMediaYetError(RecorderDisabledError):
    """A clip was requested before the first real frame was recorded."""


class ClipSessionGoneError(Exception):
    """The addressed recording is unknown or has aged out (an HTTP 410)."""


@dataclass(frozen=True)
class ClipResult:
    """The immediate, pollable outcome of a clip or recording request.

    The recorder returns this without waiting for the boundary segment to close,
    so ``end_marker`` may point inside a segment ffmpeg is still writing. The
    client polls ``playlist_url`` until the manifest endpoint answers ``200``.

    Attributes:
        session_id: The recording id the clip belongs to.
        kind: ``"snap"`` for a tail clip, ``"recording"`` for the whole session.
        start_marker: Clip start, in seconds on the recording timeline.
        end_marker: Clip end, in seconds on the recording timeline.
        now_marker: The timeline position when the request was resolved.
        predicted_ready_at_ms: Unix epoch in milliseconds when the boundary
            segment is expected to be servable.
        playlist_url: A path-only ``/clips?...`` URL the client absolutises.
    """

    session_id: str
    kind: str
    start_marker: float
    end_marker: float
    now_marker: float
    predicted_ready_at_ms: int
    playlist_url: str

    def to_dict(self) -> dict[str, Any]:
        """Return the clip as a plain dict for wire encoding and journalling."""
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "start_marker": self.start_marker,
            "end_marker": self.end_marker,
            "now_marker": self.now_marker,
            "predicted_ready_at_ms": self.predicted_ready_at_ms,
            "playlist_url": self.playlist_url,
        }


@dataclass(frozen=True)
class ClipManifest:
    """A ready HLS manifest body (an HTTP 200)."""

    body: str
    media_type: str = _HLS_MEDIA_TYPE


@dataclass(frozen=True)
class Pending:
    """The boundary segment has not landed yet (an HTTP 202 with a retry hint)."""

    retry_after: int = 2


@dataclass(frozen=True)
class Gone:
    """The addressed recording is unknown or has aged out (an HTTP 410)."""


ClipReadyCallback = Callable[[ClipResult], None]
"""Notified once a requested clip's boundary segment is on disk."""

ChunkReadyCallback = Callable[[str, int], None]
"""Notified with ``(recording_id, idx)`` once a recording segment has closed."""


class Recorder:
    """Records one session's output and serves its clips from local disk.

    Owned by the runner and constructed from the recording config. The runner
    calls :meth:`start` on the session-start boundary and :meth:`stop` on close;
    in between, the recorder taps the model's output buffer, encodes fMP4
    segments, and answers :meth:`request_clip` / :meth:`request_recording` for
    the wire and :meth:`manifest` / :meth:`chunk_path` for the ``/clips`` routes.

    Recording is best-effort: an encoder failure disables the recorder for the
    rest of the session and surfaces as a failed clip request, but never breaks
    the media path it taps.
    """

    def __init__(
        self,
        config: RecordingConfig,
        *,
        on_clip_ready: ClipReadyCallback | None = None,
        on_chunk_ready: ChunkReadyCallback | None = None,
    ) -> None:
        """Bind the recorder to its config and the readiness notifications.

        Args:
            config: The recorder's tunables, including the directory clips are
                written under.
            on_clip_ready: Called when a requested clip's boundary segment lands
                on disk, on a recorder-owned thread.
            on_chunk_ready: Called with ``(recording_id, idx)`` when a recording
                segment has closed, on a recorder-owned thread, so the recording
                can be mirrored as it is produced.
        """
        self._config = config
        self._on_clip_ready = on_clip_ready
        self._on_chunk_ready = on_chunk_ready
        # The recordings root is materialised on the first start, so a disabled
        # recorder (the common case) never creates a directory.
        self._root: Path | None = None

        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._markers: MarkerBookkeeper | None = None
        self._encoder: ChunkEncoder | None = None
        self._video_track: str | None = None
        self._audio_track: str | None = None
        self._audio_sample_rate = 48_000
        self._has_audio = False
        self._started = False
        self._disabled = False

        self._feed_queue: queue.Queue[_FeedItem | None] = queue.Queue(maxsize=4)
        self._feed_thread: threading.Thread | None = None
        self._feed_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()

        self._pending: list[tuple[int, ClipResult]] = []
        self._pending_lock = threading.Lock()

        # The highest media-segment index already announced as closed, and whether
        # the init segment has been, so each segment is announced exactly once.
        # Guarded so the watch thread and a concurrent stop never double-announce.
        self._announced_init = False
        self._announced_chunk_idx = -1
        self._chunk_lock = threading.Lock()

        self._dropped_frames = 0
        # Fractional grid frames carried between chunks so resampling a chunk's
        # own rate onto the fixed recording grid accumulates no rounding drift.
        self._grid_debt = 0.0
        self._audio_jitter_buf: list[_AudioArray] = []
        self._audio_buffered_samples = 0

    @property
    def enabled(self) -> bool:
        """Whether recording is turned on by config."""
        return self._config.enabled

    @property
    def disabled(self) -> bool:
        """Whether the recorder cannot currently serve a clip request."""
        return (
            self._disabled
            or not self._started
            or (self._encoder is not None and self._encoder.failed)
        )

    # -- lifecycle ------------------------------------------------------------

    def start(self, session_id: str) -> None:
        """Begin recording the session's output under *session_id*.

        The recording is stored and addressed under *session_id*: a director
        passes the platform's session id so a clip is fetched by the same id the
        platform stores it under, and a session started without one is given a
        freshly minted id so sequential recordings never share a directory.
        Opens the recording directory and starts the feed and watch workers;
        frames arrive through :meth:`on_chunk`, fed by the runner's media
        fan-out. A no-op when recording is disabled or already running.

        Args:
            session_id: The id this recording is stored and addressed under.
        """
        if not self._config.enabled or self._started:
            return
        self._reset_session_state()
        if self._root is None:
            self._root = (
                Path(self._config.recording_dir)
                if self._config.recording_dir
                else Path(tempfile.mkdtemp(prefix="reactor-recordings-"))
            )
            self._root.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._session_dir = self._root / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._markers = MarkerBookkeeper()
        self._feed_stop.clear()
        self._watch_stop.clear()
        self._feed_thread = threading.Thread(
            target=self._feed_loop, name="recording-feed", daemon=True
        )
        self._watch_thread = threading.Thread(
            target=self._watch_loop, name="recording-watch", daemon=True
        )
        self._feed_thread.start()
        self._watch_thread.start()
        self._started = True
        logger.info(
            "recorder started",
            recording_id=self._session_id,
            session_id=session_id,
            dir=str(self._session_dir),
        )

    def stop(self) -> None:
        """Stop recording, finalise the directory, and release the encoder.

        Blocks while the encoder shuts down and the workers join, so the runner
        runs it off the event loop. The recording directory is left in place so
        its clips stay fetchable after the session ends.
        """
        if not self._started:
            return
        self._disabled = True
        self._feed_stop.set()
        with contextlib.suppress(queue.Full):
            self._feed_queue.put_nowait(None)
        feed_thread = self._feed_thread
        self._feed_thread = None
        if self._encoder is not None:
            self._encoder.stop()
        if feed_thread is not None:
            feed_thread.join(timeout=2.0)
        # Mark the recording finished so its final segment is servable, then fire
        # any clip whose boundary has now landed before the watcher winds down.
        if self._session_dir is not None:
            try:
                (self._session_dir / _COMPLETE_MARKER).write_text("")
            except OSError:
                logger.exception("failed to write recording completion marker")
        self._fire_ready_chunks()
        self._fire_ready_clips()
        self._watch_stop.set()
        watch_thread = self._watch_thread
        self._watch_thread = None
        if watch_thread is not None:
            watch_thread.join(timeout=2.0)
        self._started = False
        logger.info(
            "recorder stopped",
            recording_id=self._session_id,
            dropped=self._dropped_frames,
        )

    def _reset_session_state(self) -> None:
        """Clear per-session state so a restart never inherits the last session."""
        self._encoder = None
        self._disabled = False
        self._video_track = None
        self._audio_track = None
        self._audio_sample_rate = 48_000
        self._has_audio = False
        self._dropped_frames = 0
        self._grid_debt = 0.0
        self._audio_jitter_buf = []
        self._audio_buffered_samples = 0
        with self._pending_lock:
            self._pending = []
        with self._chunk_lock:
            self._announced_init = False
            self._announced_chunk_idx = -1
        self._drain_feed_queue()

    # -- media fan-out tap ----------------------------------------------------

    def on_chunk(self, chunk: MediaChunk) -> None:
        """Feed one emitted media chunk to the recording; non-blocking.

        Called on the model thread by the runner's media fan-out. The chunk's
        frames are resampled from the chunk's own rate onto the fixed recording
        grid (:data:`RECORDING_FPS`): the chunk represents ``n_frames / fps``
        seconds of media, which is ``that * RECORDING_FPS`` grid frames, so a
        model producing at a rate other than the grid still records at true
        duration. Fractional grid frames carry across chunks so the resampling
        accumulates no drift. The timeline advances by the media actually fed,
        not by wall-clock, so a pause simply stops advancing rather than
        recording dead air. A frame that does not fit the feed queue is dropped
        to keep the model thread unblocked.
        """
        if self.disabled:
            return
        markers = self._markers
        if markers is None:
            return
        bundle = chunk.bundle
        if self._encoder is None:
            self._build_encoder(bundle)
            if self._encoder is None:
                return
        frames = self._video_frames(bundle)
        if not frames:
            return
        fps = chunk.fps if chunk.fps > 0 else float(RECORDING_FPS)
        if self._has_audio:
            self._buffer_audio(bundle)
        self._grid_debt += (len(frames) / fps) * RECORDING_FPS
        grid_frames = int(self._grid_debt)
        self._grid_debt -= grid_frames
        audio_target = round(self._audio_sample_rate / RECORDING_FPS) if self._has_audio else 0
        fed = 0
        for i in range(grid_frames):
            video_data = frames[i * len(frames) // grid_frames]
            audio_data = self._take_audio(audio_target) if self._has_audio else None
            try:
                self._feed_queue.put_nowait((video_data, audio_data))
            except queue.Full:
                self._dropped_frames += 1
                if self._dropped_frames == 1 or self._dropped_frames % 300 == 0:
                    logger.warning(
                        "recorder feed queue full; dropping a frame to keep the model unblocked",
                        dropped_total=self._dropped_frames,
                    )
                break
            fed += 1
        if fed:
            markers.advance(fed / RECORDING_FPS)

    def _video_frames(self, bundle: MediaBundle) -> list[npt.NDArray[Any]]:
        """Split the recorded video track into single ``(H, W, 3)`` frames.

        A batched track ``(N, H, W, 3)`` becomes ``N`` frames; an unbatched one
        is a single frame. Empty when the bundle carries no video.
        """
        track = bundle.get_track(self._video_track) if self._video_track is not None else None
        if track is None:
            videos = bundle.get_tracks_by_kind(TrackKind.VIDEO)
            track = videos[0] if videos else None
        if track is None:
            return []
        data = track.data
        if data.ndim == 4:
            return [data[i] for i in range(data.shape[0])]
        return [data]

    def _build_encoder(self, bundle: MediaBundle) -> None:
        """Resolve the tracks to record and spawn the encoder from a real frame."""
        video_name = self._config.video_track
        if video_name is None or bundle.get_track(video_name) is None:
            videos = bundle.get_tracks_by_kind(TrackKind.VIDEO)
            if not videos:
                logger.warning("recording enabled but the model emits no video; disabling recorder")
                self._disabled = True
                return
            video_name = videos[0].info.name
        self._video_track = video_name

        audio_name = self._config.audio_track
        audio_td = bundle.get_track(audio_name) if audio_name is not None else None
        if audio_name is None:
            audios = bundle.get_tracks_by_kind(TrackKind.AUDIO)
            audio_td = audios[0] if audios else None
            audio_name = audio_td.info.name if audio_td is not None else None
        self._audio_track = audio_name
        self._has_audio = audio_name is not None and audio_td is not None
        if self._has_audio and audio_td is not None and audio_td.info.rate > 0:
            self._audio_sample_rate = int(audio_td.info.rate)

        assert self._session_dir is not None
        self._encoder = ChunkEncoder(
            output_dir=self._session_dir,
            config=self._config,
            has_audio=self._has_audio,
            audio_sample_rate=self._audio_sample_rate,
            frame_rate=RECORDING_FPS,
        )

    def _feed_loop(self) -> None:
        """Drain the feed queue into the encoder, disabling on encoder failure.

        Each item is a video frame and the audio samples for its grid slot, both
        already sized to the recording grid, so the worker only writes bytes and
        the encoder's PTS derives from the fixed input frame rate.
        """
        while True:
            if self._feed_stop.is_set() and self._feed_queue.empty():
                return
            try:
                item = self._feed_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            encoder = self._encoder
            if encoder is None:
                continue
            video, audio = item
            try:
                encoder.feed_video(video)
                if self._has_audio and audio is not None:
                    encoder.feed_audio(audio)
            except Exception:
                logger.exception("recorder encoder feed failed; disabling recording")
                self._disabled = True
                self._drain_feed_queue()
                return

    def _buffer_audio(self, bundle: MediaBundle) -> None:
        """Append a chunk's audio to the jitter buffer, capped at one second.

        The whole chunk's audio is buffered once; :meth:`_take_audio` then pulls a
        grid slot's worth per recorded frame, so the audio DTS tracks the video
        PTS regardless of how the chunk's frames map onto the grid.
        """
        if self._audio_track is None:
            return
        track = bundle.get_track(self._audio_track)
        if track is None or track.data.size == 0:
            return
        flat = np.ascontiguousarray(track.data, dtype=np.int16).reshape(-1)
        self._audio_jitter_buf.append(flat)
        self._audio_buffered_samples += int(flat.size)
        cap = self._audio_sample_rate
        while self._audio_buffered_samples > cap:
            head = self._audio_jitter_buf[0]
            drop = self._audio_buffered_samples - cap
            if int(head.size) <= drop:
                self._audio_jitter_buf.pop(0)
                self._audio_buffered_samples -= int(head.size)
            else:
                self._audio_jitter_buf[0] = head[drop:]
                self._audio_buffered_samples -= drop

    def _take_audio(self, target: int) -> _AudioArray | None:
        """Pull exactly *target* samples from the jitter buffer, padding silence."""
        if target <= 0:
            return None
        out = np.zeros(target, dtype=np.int16)
        filled = 0
        while filled < target and self._audio_jitter_buf:
            head = self._audio_jitter_buf[0]
            take = min(int(head.size), target - filled)
            out[filled : filled + take] = head[:take]
            filled += take
            if take == int(head.size):
                self._audio_jitter_buf.pop(0)
            else:
                self._audio_jitter_buf[0] = head[take:]
            self._audio_buffered_samples -= take
        return out.reshape(1, -1)

    def _drain_feed_queue(self) -> None:
        """Discard every queued feed item."""
        while True:
            try:
                self._feed_queue.get_nowait()
            except queue.Empty:
                return

    # -- clip / recording requests --------------------------------------------

    def request_clip(self, duration_seconds: float) -> ClipResult:
        """Resolve a snap-clip of the last *duration_seconds* of output.

        Returns immediately with ``end_marker`` at the current timeline position,
        so the boundary segment may still be in flight; the client polls
        ``/clips`` until it lands.

        Raises:
            RecorderDisabledError: If recording is off or the encoder crashed.
            NoMediaYetError: If no real frame has been recorded yet.
            ValueError: If *duration_seconds* is not positive.
        """
        markers = self._require_started_markers()
        capped = min(float(duration_seconds), float(self._config.clip_max_seconds))
        if capped <= 0:
            raise ValueError("duration_seconds must be positive")
        start, end = markers.compute_clip_range(capped)
        return self._build_result("snap", start, end)

    def request_recording(self) -> ClipResult:
        """Resolve a request for the whole session so far.

        Same promise-then-poll semantics as :meth:`request_clip`.

        Raises:
            RecorderDisabledError: If recording is off or the encoder crashed.
            NoMediaYetError: If no real frame has been recorded yet.
        """
        markers = self._require_started_markers()
        start, end = markers.compute_recording_range()
        return self._build_result("recording", start, end)

    def _require_started_markers(self) -> MarkerBookkeeper:
        """Return the live marker bookkeeper, or raise why a clip cannot be served."""
        if self.disabled:
            raise RecorderDisabledError("recorder disabled or encoder crashed")
        markers = self._markers
        if markers is None or not markers.recording_started:
            raise NoMediaYetError("no media generated yet")
        return markers

    def _build_result(self, kind: str, start: float, end: float) -> ClipResult:
        """Assemble a :class:`ClipResult` and register its boundary for readiness."""
        markers = self._markers
        assert markers is not None
        assert self._session_id is not None
        now = markers.now_marker()
        cs = float(self._config.chunk_seconds)
        wait_s = max(0.0, (math.floor(now / cs) + 1) * cs - now) if cs > 0 else 0.0
        predicted_ready_at_ms = round((time.time() + wait_s) * 1000)
        query = urlencode(
            {"session_id": self._session_id, "start": f"{start:.3f}", "end": f"{end:.3f}"}
        )
        clip = ClipResult(
            session_id=self._session_id,
            kind=kind,
            start_marker=start,
            end_marker=end,
            now_marker=now,
            predicted_ready_at_ms=predicted_ready_at_ms,
            playlist_url=f"/clips?{query}",
        )
        with self._pending_lock:
            self._pending.append((_boundary_index(end, self._config.chunk_seconds), clip))
        return clip

    # -- clip-ready notification ----------------------------------------------

    def _watch_loop(self) -> None:
        """Poll for landed segments and notify ready clips and closed chunks."""
        while not self._watch_stop.is_set():
            self._fire_ready_chunks()
            self._fire_ready_clips()
            self._watch_stop.wait(0.1)

    def _fire_ready_clips(self) -> None:
        """Notify the consumer for every pending clip whose boundary has landed.

        Selection and removal happen under a single lock hold, so each pending
        clip is claimed by exactly one caller: ``stop()`` and the watch thread can
        run this concurrently without both firing the same clip (and journalling a
        duplicate ``clip_ready`` fact). Callbacks fire outside the lock, on only the
        entries this caller removed.
        """
        session_dir = self._session_dir
        if session_dir is None:
            return
        with self._pending_lock:
            ready = [
                entry for entry in self._pending if self._boundary_ready(session_dir, entry[0])
            ]
            for entry in ready:
                self._pending.remove(entry)
        callback = self._on_clip_ready
        if callback is None:
            return
        for _, clip in ready:
            try:
                callback(clip)
            except Exception:
                logger.exception("clip-ready callback raised")

    def _fire_ready_chunks(self) -> None:
        """Announce every recording segment that has closed since the last poll.

        Walks the closed media segments in order (and the init segment once it is
        readable), notifying the consumer once per segment so a recording can be
        mirrored as it is produced. A segment counts as closed the same way the
        clip manifest treats it: its successor exists, or the recording finished.
        """
        session_dir = self._session_dir
        recording_id = self._session_id
        callback = self._on_chunk_ready
        if session_dir is None or recording_id is None or callback is None:
            return
        ready: list[int] = []
        with self._chunk_lock:
            if not self._announced_init and self._init_ready(session_dir):
                self._announced_init = True
                ready.append(-1)
            while self._boundary_ready(session_dir, self._announced_chunk_idx + 1):
                self._announced_chunk_idx += 1
                ready.append(self._announced_chunk_idx)
        for idx in ready:
            try:
                callback(recording_id, idx)
            except Exception:
                logger.exception("chunk-ready callback raised")

    # -- HTTP serving ---------------------------------------------------------

    def manifest(self, session_id: str, start: float, end: float) -> ClipManifest | Pending | Gone:
        """Resolve a marker range into an HLS manifest, a pending hint, or gone.

        Args:
            session_id: The recording id the clip belongs to.
            start: Clip start in seconds on the recording timeline.
            end: Clip end in seconds on the recording timeline.

        Returns:
            A :class:`ClipManifest` once the boundary segment is on disk, a
            :class:`Pending` while it is still in flight, or :class:`Gone` when
            the recording is unknown or aged out.

        Raises:
            ValueError: If the marker range is malformed.
        """
        if self._root is None or not _SESSION_ID_RE.match(session_id):
            return Gone()
        if not (math.isfinite(start) and start >= 0):
            raise ValueError("start must be a finite non-negative number")
        if not (math.isfinite(end) and end > start):
            raise ValueError("end must be a finite number greater than start")
        cs = self._config.chunk_seconds
        if cs <= 0:
            return Gone()
        session_dir = self._root / session_id
        if not session_dir.is_dir():
            return Gone()
        chunk_start_idx = max(0, math.floor(start / cs))
        chunk_end_idx = max(chunk_start_idx, math.ceil(end / cs) - 1)
        if not self._boundary_ready(session_dir, chunk_end_idx):
            return Pending()
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            f"#EXT-X-TARGETDURATION:{cs}",
            "#EXT-X-PLAYLIST-TYPE:VOD",
            f'#EXT-X-MAP:URI="/clips/chunks/{session_id}/{_INIT_FILENAME}"',
        ]
        for idx in range(chunk_start_idx, chunk_end_idx + 1):
            lines.append(f"#EXTINF:{cs:.3f},")
            lines.append(f"/clips/chunks/{session_id}/chunk_{idx:05d}.m4s")
        lines.append("#EXT-X-ENDLIST")
        return ClipManifest(body="\n".join(lines) + "\n")

    def chunk_path(self, session_id: str, filename: str) -> Path | None:
        """Resolve a chunk URL to the file on disk that backs it.

        Args:
            session_id: The recording id from the URL path.
            filename: The segment file name from the URL path.

        Returns:
            The path to serve, or ``None`` when the file name is not a recording
            artifact or the segment does not exist.

        Raises:
            ClipSessionGoneError: If the recording id is malformed or unknown.
        """
        if self._root is None or not _SESSION_ID_RE.match(session_id):
            raise ClipSessionGoneError(session_id)
        session_dir = self._root / session_id
        if not session_dir.is_dir():
            raise ClipSessionGoneError(session_id)
        if not _CHUNK_FILENAME_RE.match(filename):
            return None
        path = session_dir / filename
        return path if path.is_file() else None

    def _boundary_ready(self, session_dir: Path, boundary_idx: int) -> bool:
        """Return whether the boundary segment is on disk and closed."""
        if not self._init_ready(session_dir):
            return False
        boundary = session_dir / f"chunk_{boundary_idx:05d}.m4s"
        if not boundary.is_file():
            return False
        # A segment is closed once its successor exists (ffmpeg has rolled over)
        # or the recording itself has finished.
        successor = session_dir / f"chunk_{boundary_idx + 1:05d}.m4s"
        return successor.is_file() or (session_dir / _COMPLETE_MARKER).is_file()

    @staticmethod
    def _init_ready(session_dir: Path) -> bool:
        """Return whether the init segment carries its codec headers yet."""
        init = session_dir / _INIT_FILENAME
        if not init.is_file():
            return False
        try:
            if init.stat().st_size > 0:
                return True
        except OSError:
            return False
        # ffmpeg creates init empty and flushes its headers with the first
        # segment, so the first chunk appearing is the signal init is usable.
        return (session_dir / "chunk_00000.m4s").is_file()


def _boundary_index(end: float, chunk_seconds: int) -> int:
    """Return the index of the segment that contains the clip's end marker."""
    if chunk_seconds <= 0:
        return 0
    return max(0, math.ceil(end / chunk_seconds) - 1)
