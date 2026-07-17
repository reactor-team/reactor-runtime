# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Per-connection media pacer.

A model emits finished media in bursts, one :class:`MediaChunk` per inference,
at whatever rate its compute finishes; a client needs frames at a smooth, steady
cadence. :class:`MediaPacer` is the seam between the two on one WebRTC
connection: it takes the connection's share of each chunk, splits any batch into
single frames, and a dedicated thread drains them to the wire at the chunk's
declared rate, repeating the last frame to fill a gap so the stream never stalls.

One pacer lives per connection and dies with it, so there is no cross-session
state to reset — a fresh connection starts with a fresh pacer. This is the seam
a native GStreamer jitter buffer replaces: everything above it hands the pacer a
chunk and never sees a frame.

Three stages::

    submit(MediaChunk)  ->  Queue  ->  pacing thread  ->  on_frame(MediaBundle)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from reactor_runtime.core.values import (
    MediaBundle,
    MediaChunk,
    TrackData,
    TrackInfo,
    TrackKind,
)

logger = logging.getLogger(__name__)

DEFAULT_FRAME_DIMENSIONS: tuple[int, int, int] = (720, 1280, 3)
"""Black-frame shape used until the first real video frame reveals the true size."""

FrameSink = Callable[[MediaBundle], None]
"""Receives one single-frame bundle per tick, at the pacer's cadence."""


class MediaPacer:
    """Drains submitted media to a wire at a steady FPS.

    The producer calls :meth:`submit`; a dedicated thread, started by
    :meth:`start`, delivers one frame per tick to the frame sink. When no new
    frame is ready the last one is repeated (a gap-fill) so the wire stays live;
    before the first real frame a black frame stands in.

    Args:
        video_tracks: The connection's outbound video tracks, used to synthesise
            black frames before the first real frame arrives.
        on_frame: The sink one single-frame bundle is handed to each tick.
        queue_depth: How many frames may sit queued before :meth:`submit` drops,
            so a fast producer never blocks the thread that submits it.
        fps: The initial pacing rate, used until the first chunk sets its own.
    """

    def __init__(
        self,
        video_tracks: dict[str, TrackInfo],
        on_frame: FrameSink,
        *,
        queue_depth: int = 10,
        fps: float = 30.0,
    ) -> None:
        self._video_tracks = {
            name: info for name, info in video_tracks.items() if info.kind is TrackKind.VIDEO
        }
        self._on_frame = on_frame

        self._queue: queue.Queue[MediaBundle] = queue.Queue(maxsize=queue_depth)

        self._interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self._thread: threading.Thread | None = None
        # The single source of truth for whether pacing runs: set = stopped.
        self._stop = threading.Event()
        self._stop.set()
        self._lifecycle_lock = threading.Lock()

        self._last_emitted: MediaBundle | None = None
        self._frame_dims: tuple[int, int, int] | None = None

    def submit(self, chunk: MediaChunk) -> int:
        """Split *chunk* into single frames, adopt its rate, and enqueue them.

        The pacing rate is updated to the chunk's ``fps`` so a dynamic-rate model
        paces at the throughput it is actually producing. Frames that do not fit
        the queue are dropped rather than blocking the producer, which runs on
        the model thread and must never stall on one slow connection.

        Args:
            chunk: A finished media chunk, possibly batched.

        Returns:
            The number of frames actually enqueued.
        """
        if chunk.fps > 0:
            self._interval = 1.0 / chunk.fps
        enqueued = 0
        for frame in chunk.frames():
            try:
                self._queue.put_nowait(frame)
                enqueued += 1
            except queue.Full:
                pass
        return enqueued

    def start(self) -> None:
        """Start the pacing thread, idempotently."""
        with self._lifecycle_lock:
            if not self._stop.is_set():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._pacing_loop, name="media-pacer")
            self._thread.start()

    def stop(self) -> None:
        """Stop the pacing thread and wait for it to finish.

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
                raise RuntimeError("media pacer thread did not stop within 2s")

    def _pacing_loop(self) -> None:
        """Deliver one frame per tick at a steady cadence until stopped."""
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                interval = self._interval
                self._emit_one_tick()
                next_tick += interval
                now = time.perf_counter()
                # If a stall (GC pause, slow sink) put the schedule more than a
                # full interval behind, resume at the next clean boundary rather
                # than rapid-firing catch-up frames.
                if next_tick < now - interval:
                    next_tick = now + interval
                self._sleep_until(next_tick)
        except Exception:
            logger.exception("Media pacer loop crashed")
            self._stop.set()

    def _emit_one_tick(self) -> None:
        """Emit the next queued frame, or a gap-fill duplicate, or black."""
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            item = None

        if item is not None:
            video = item.get_tracks_by_kind(TrackKind.VIDEO)
            if video and video[0].data.ndim == 3:
                shape = video[0].data.shape
                self._frame_dims = (shape[0], shape[1], shape[2])
            self._last_emitted = item
            self._dispatch(item)
        elif self._last_emitted is not None:
            self._dispatch(self._video_only(self._last_emitted))
        else:
            self._dispatch(self._black_bundle())

    def _dispatch(self, bundle: MediaBundle) -> None:
        """Hand one frame to the sink, isolating its failure from the loop."""
        try:
            self._on_frame(bundle)
        except Exception:
            logger.exception("Media pacer frame sink raised; continuing")

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
