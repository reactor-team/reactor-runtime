"""Rate-controlled emission buffer.

A model produces frames whenever its compute finishes; a client needs them at a
smooth, steady rate. :class:`OutputBuffer` is the seam between the two: it takes
finished :class:`MediaBundle`s, splits any batched bundle into single frames, and
a dedicated thread drains them to registered callbacks at a fixed FPS, filling
gaps by repeating the last frame so the stream never stalls.

Three stages::

    submit(MediaBundle)  ->  Queue  ->  emission thread  ->  callbacks
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from reactor_runtime.core.values import (
    MediaBundle,
    TrackData,
    TrackInfo,
    TrackKind,
)

logger = logging.getLogger(__name__)

DEFAULT_FRAME_DIMENSIONS: tuple[int, int, int] = (720, 1280, 3)
"""Black-frame shape used until the first real video frame reveals the true size."""

EmissionCallback = Callable[[MediaBundle, bool, bool], None]
"""A per-tick observer ``(bundle, duplicate, is_fresh_black)``."""


def split_batch(bundle: MediaBundle) -> list[MediaBundle]:
    """Split a multi-frame bundle into one bundle per frame.

    A batched video track is ``(N, H, W, 3)``; an unbatched one is ``(H, W, 3)``
    and is repeated into every frame. Audio is divided proportionally across the
    frames. All batched video tracks must agree on ``N``.

    Args:
        bundle: The bundle to split.

    Returns:
        One single-frame bundle per batched frame, or ``[bundle]`` unchanged when
        there is nothing to split.

    Raises:
        ValueError: If two batched video tracks disagree on the batch size.
    """
    video_tracks = bundle.get_tracks_by_kind(TrackKind.VIDEO)
    if not video_tracks:
        return [bundle]

    batched = [(track, track.data.shape[0]) for track in video_tracks if track.data.ndim == 4]
    if not batched:
        return [bundle]

    n_frames = batched[0][1]
    if n_frames == 1:
        # Squeeze the batch dimension into a fresh bundle rather than editing the
        # caller's: the multi-frame path below also leaves the input untouched,
        # and a producer must be able to read back what it submitted.
        squeezed = dict(bundle.tracks)
        for track, _ in batched:
            squeezed[track.info.name] = TrackData(info=track.info, data=track.data[0])
        return [MediaBundle(tracks=squeezed)]

    for track, size in batched:
        if size != n_frames:
            raise ValueError(
                f"Video track '{track.info.name}' has batch size {size}, "
                f"expected {n_frames} (from '{batched[0][0].info.name}')"
            )

    video_splits: dict[str, list[Any]] = {}
    for track in video_tracks:
        if track.data.ndim == 4:
            video_splits[track.info.name] = list(track.data)
        else:
            video_splits[track.info.name] = [track.data] * n_frames

    audio_splits: dict[str, list[Any]] = {}
    for track in bundle.get_tracks_by_kind(TrackKind.AUDIO):
        audio = track.data
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        audio_splits[track.info.name] = np.array_split(audio, n_frames, axis=1)

    info_by_name = {track.info.name: track.info for track in bundle.get_tracks()}
    result: list[MediaBundle] = []
    for index in range(n_frames):
        tracks: dict[str, TrackData] = {}
        for name, frames in video_splits.items():
            tracks[name] = TrackData(info=info_by_name[name], data=frames[index])
        for name, chunks in audio_splits.items():
            tracks[name] = TrackData(info=info_by_name[name], data=chunks[index])
        result.append(MediaBundle(tracks=tracks))
    return result


class _FlushMarker:
    """Queue sentinel that asks the emission thread to reset at a session boundary."""


_FLUSH_MARKER = _FlushMarker()


class OutputBuffer:
    """Drains submitted media to callbacks at a steady FPS.

    The producer calls :meth:`submit`; a dedicated thread, started by
    :meth:`start_emission`, delivers one frame per tick to every registered
    callback. When no new frame is ready the last one is repeated (a gap-fill
    duplicate) so the stream stays live.

    Args:
        output_tracks: The model's outbound track topology, used to synthesise
            black frames before the first real frame arrives.
        queue_depth: How many frames may sit queued before :meth:`submit` blocks
            to throttle the producer to the emission rate.
    """

    def __init__(self, output_tracks: dict[str, TrackInfo], queue_depth: int = 10) -> None:
        self._video_tracks = {
            name: info for name, info in output_tracks.items() if info.kind is TrackKind.VIDEO
        }

        self._callbacks: list[EmissionCallback] = []
        self._callbacks_lock = threading.Lock()

        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_depth)

        self._fps: float = 0.0
        self._interval: float = 0.0

        self._thread: threading.Thread | None = None
        # The single source of truth for whether emission runs: set = stopped.
        # Starts set so a submit before start_emission is a no-op rather than a
        # block against a thread that will never drain.
        self._stop = threading.Event()
        self._stop.set()
        self._lifecycle_lock = threading.Lock()

        self._last_emitted: MediaBundle | None = None
        self._frame_dims: tuple[int, int, int] | None = None

    def add_callback(self, callback: EmissionCallback) -> None:
        """Register a per-tick callback, idempotently and in registration order."""
        with self._callbacks_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: EmissionCallback) -> None:
        """Deregister a callback; a no-op if it was never registered."""
        with self._callbacks_lock, contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    def set_fps(self, fps: float) -> None:
        """Set the emission rate.

        Args:
            fps: Frames per second; must be positive.

        Raises:
            ValueError: If *fps* is not positive.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._fps = fps
        self._interval = 1.0 / fps

    @property
    def fps(self) -> float:
        """The current emission rate."""
        return self._fps

    def submit(self, bundle: MediaBundle, *, drop: bool = False) -> int:
        """Split *bundle* into single frames and enqueue them.

        Blocks the caller while the queue is full, which throttles the producer to
        the emission rate, unless *drop* is set.

        Args:
            bundle: A finished media bundle, possibly batched.
            drop: Discard frames that do not fit instead of blocking.

        Returns:
            The number of frames actually enqueued.
        """
        enqueued = 0
        for frame in split_batch(bundle):
            if self._enqueue(frame, drop=drop):
                enqueued += 1
        return enqueued

    def _enqueue(self, bundle: MediaBundle, *, drop: bool) -> bool:
        """Enqueue one single-frame bundle, throttling unless *drop* is set."""
        if drop:
            try:
                self._queue.put_nowait(bundle)
                return True
            except queue.Full:
                return False
        while not self._stop.is_set():
            try:
                self._queue.put(bundle, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _drain_queue(self) -> None:
        """Discard every queued item."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _dispatch(self, bundle: MediaBundle, duplicate: bool, *, is_fresh_black: bool) -> None:
        """Fire every callback in order, isolating one's failure from the rest."""
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(bundle, duplicate, is_fresh_black)
            except Exception:
                logger.exception("OutputBuffer callback %r raised; continuing", callback)

    def _emission_loop(self) -> None:
        """Deliver one frame per tick at a steady cadence until stopped."""
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                interval = self._interval
                self._emit_one_tick()
                next_tick += interval
                now = time.perf_counter()
                # If a stall (GC pause, slow callback) put the schedule more than a
                # full interval behind, resume at the next clean boundary rather
                # than rapid-firing catch-up frames.
                if next_tick < now - interval:
                    next_tick = now + interval
                self._sleep_until(next_tick)
        except Exception:
            logger.exception("Emission loop crashed")
            self._stop.set()

    def _emit_one_tick(self) -> None:
        """Emit the next queued frame, the flush black, or a gap-fill duplicate."""
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            item = None

        if isinstance(item, _FlushMarker):
            # Session-boundary black: tagged so the wire forwards it (otherwise the
            # client freezes on the last frame) while the recorder ignores it.
            self._last_emitted = None
            self._dispatch(self._black_bundle(), True, is_fresh_black=True)
        elif isinstance(item, MediaBundle):
            video = item.get_tracks_by_kind(TrackKind.VIDEO)
            if video and video[0].data.ndim == 3:
                shape = video[0].data.shape
                self._frame_dims = (shape[0], shape[1], shape[2])
            self._last_emitted = item
            self._dispatch(item, False, is_fresh_black=False)
        elif self._last_emitted is not None:
            self._dispatch(self._video_only(self._last_emitted), True, is_fresh_black=False)
        else:
            self._dispatch(self._black_bundle(), True, is_fresh_black=False)

    _SLEEP_CHUNK = 0.005

    def _sleep_until(self, deadline: float) -> None:
        """Sleep until *deadline*, waking immediately on stop.

        Waits in chunks down to ``_SLEEP_CHUNK`` then spin-waits the final stretch,
        since OS sleep granularity overshoots short sleeps by several milliseconds.
        """
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or self._stop.is_set():
                return
            if remaining > self._SLEEP_CHUNK and self._stop.wait(timeout=self._SLEEP_CHUNK):
                return

    @staticmethod
    def _video_only(bundle: MediaBundle) -> MediaBundle:
        """Return a copy of *bundle* carrying only its video tracks.

        A gap-fill repeats the last video frame but not its audio, which would
        otherwise be replayed and heard as a stutter.
        """
        return MediaBundle(
            tracks={
                name: track
                for name, track in bundle.tracks.items()
                if track.info.kind is TrackKind.VIDEO
            }
        )

    def _black_bundle(self) -> MediaBundle:
        """Build a black frame for each outbound video track."""
        dims = self._frame_dims or DEFAULT_FRAME_DIMENSIONS
        black = np.zeros(dims, dtype=np.uint8)
        return MediaBundle(
            tracks={
                name: TrackData(info=info, data=black) for name, info in self._video_tracks.items()
            }
        )

    def start_emission(self) -> None:
        """Start the emission thread.

        Raises:
            RuntimeError: If no callback is registered or the FPS is unset.
        """
        if not self._callbacks:
            raise RuntimeError("cannot start emission: register a callback first")
        if self._fps <= 0:
            raise RuntimeError("cannot start emission: set the FPS first")
        with self._lifecycle_lock:
            if not self._stop.is_set():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._emission_loop, name="emission")
            self._thread.start()

    def stop_emission(self) -> None:
        """Stop the emission thread and wait for it to finish.

        Raises:
            RuntimeError: If the thread does not stop within two seconds.
        """
        with self._lifecycle_lock:
            if self._stop.is_set():
                return
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("emission thread did not stop within 2s")

    def clear(self) -> None:
        """Drop queued frames and reset gap-fill state for a new session.

        Resets ``_last_emitted`` / ``_frame_dims`` directly, so call it only while
        emission is stopped — between sessions. The in-flight reset, safe while
        the emission thread is running, is :meth:`flush`, which routes the reset
        through a sentinel the thread consumes.
        """
        self._last_emitted = None
        self._frame_dims = None
        self._drain_queue()

    def flush(self) -> None:
        """Drop queued frames and ask the emission thread for a boundary reset.

        The thread resets ``_last_emitted`` when it dequeues the sentinel, then the
        next tick emits a fresh black tagged ``is_fresh_black`` — forwarded by the
        wire, ignored by the recorder.
        """
        self._drain_queue()
        try:
            self._queue.put_nowait(_FLUSH_MARKER)
        except queue.Full:
            # The producer is the only other writer and it just drained, so this
            # is unreachable in practice.
            logger.warning("OutputBuffer.flush: queue full after drain; reset sentinel dropped")
