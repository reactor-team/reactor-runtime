"""The media pacer of a session's room connection.

A model emits finished media in bursts, one :class:`MediaChunk` per step; a
room needs frames at a steady cadence. :class:`RoomPacer` takes each chunk,
splits it into single frames, and a dedicated thread hands them to the room
connection at the chunk's declared rate, video and audio of one frame together.

It differs from the WebRTC pacer where a room differs from a peer. A room's
tracks are published at the join, before the model has rendered anything, and
a subscriber shows nothing until the first frame, so no black frame is sent
at the start: the first frame in the room is the model's. A subscriber keeps
the last frame it decoded through a gap, so nothing is repeated to fill one.
A plain flush still cuts to one black frame, which is what a flush means to
the model; a held flush sends nothing, so the last frame stays on screen.

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

from reactor_runtime.core.values import MediaBundle, MediaChunk, TrackData, TrackInfo, TrackKind

logger = logging.getLogger(__name__)

FrameSink = Callable[[MediaBundle], None]
"""Receives one single-frame bundle per tick, at the pacer's cadence."""

_DEFAULT_FPS = 30.0


class RoomPacer:
    """Drains submitted media to a room connection at a steady rate.

    The producer calls :meth:`submit`; a thread started by :meth:`start`
    hands one frame per tick to the sink. When no frame is ready nothing is
    sent.

    Args:
        video_tracks: The output video tracks, used for the black frame a
            plain flush sends.
        on_frame: The sink each single-frame bundle is handed to.
        queue_depth: How many frames may sit queued between the model and the
            room, never applied below one chunk, so a model that batches
            always fits a whole chunk.
        fps: The pacing rate until the first chunk sets its own.
    """

    def __init__(
        self,
        video_tracks: dict[str, TrackInfo],
        on_frame: FrameSink,
        *,
        queue_depth: int = 10,
        fps: float = _DEFAULT_FPS,
    ) -> None:
        self._video_tracks = {
            name: info for name, info in video_tracks.items() if info.kind is TrackKind.VIDEO
        }
        self._on_frame = on_frame
        self._queue: queue.Queue[MediaBundle] = queue.Queue()
        self._depth = queue_depth
        # Signalled after each dequeue, so a backpressure submit sleeps until
        # room opens; also the lock every queue change and the flags share.
        self._room = threading.Condition()
        # Bumped by flush(); a submit in flight abandons the rest of its chunk.
        self._epoch = 0
        self._interval = 1.0 / fps if fps > 0 else 1.0 / _DEFAULT_FPS
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stop.set()
        self._closed = False
        # Set by a plain flush, cleared when the next tick sends the black frame.
        self._black_pending = False
        self._frame_dims: tuple[int, int, int] | None = None
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        """How many frames the queue bound has discarded, cumulatively."""
        return self._dropped_frames

    def submit(self, chunk: MediaChunk) -> int:
        """Split *chunk* into single frames, adopt its rate, and enqueue them.

        A chunk that asks for backpressure (``chunk.wait``) waits while the
        queue is full, throttling the producer to the playout rate. Otherwise
        frames beyond the bound are dropped from the tail and counted.

        Returns:
            The number of frames enqueued.
        """
        if chunk.fps > 0:
            self._interval = 1.0 / chunk.fps
        epoch = self._epoch
        enqueued = 0
        aborted = False
        for frame in chunk.frames():
            capacity = max(self._depth, chunk.n_frames)
            with self._room:
                if self._closed or self._epoch != epoch:
                    aborted = True
                    break
                if chunk.wait:
                    while (
                        not self._stop.is_set()
                        and self._epoch == epoch
                        and self._queue.qsize() >= capacity
                    ):
                        self._room.wait(timeout=0.1)
                    if self._stop.is_set() or self._epoch != epoch:
                        aborted = True
                        break
                elif self._queue.qsize() >= capacity:
                    break
                self._queue.put_nowait(frame)
            enqueued += 1
        if enqueued < chunk.n_frames and not chunk.wait and not aborted:
            dropped = chunk.n_frames - enqueued
            self._dropped_frames += dropped
            logger.warning(
                "Room pacer queue full; dropped %d of %d frames", dropped, chunk.n_frames
            )
        return enqueued

    def set_rate(self, fps: float) -> None:
        """Set the playout rate, re-pacing already-queued frames immediately."""
        if fps > 0:
            self._interval = 1.0 / fps

    def set_depth(self, depth: int) -> None:
        """Set the queue bound, waking any producer blocked on the old one."""
        if depth <= 0:
            return
        with self._room:
            self._depth = depth
            self._room.notify_all()

    def flush(self, *, hold: bool = False) -> None:
        """Drop queued frames, then cut to black or hold the last frame.

        Any producer blocked in a backpressure submit abandons the rest of its
        chunk, so none of the flushed content plays afterwards. Without
        *hold* the next tick sends one black frame on each video track.
        """
        with self._room:
            self._epoch += 1
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._black_pending = not hold
            self._room.notify_all()

    def start(self) -> None:
        """Start the pacing thread, idempotently. A stopped pacer stays stopped."""
        with self._room:
            if not self._stop.is_set() or self._closed:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._pacing_loop, name="room-pacer")
            self._thread.start()

    def stop(self) -> None:
        """Stop the pacing thread, wait for it, and release held media.

        Stopping is final. Queued frames are dropped once the thread has
        stopped, because each is a view into the model's whole output array.

        Raises:
            RuntimeError: If the thread does not stop within two seconds.
        """
        with self._room:
            self._stop.set()
            self._closed = True
            thread = self._thread
            self._thread = None
            self._room.notify_all()
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("room pacer thread did not stop within 2s")
        with self._room:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def _pacing_loop(self) -> None:
        """Deliver one frame per tick at a steady cadence until stopped."""
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                interval = self._interval
                self._emit_one_tick()
                next_tick += interval
                now = time.perf_counter()
                # A stall more than an interval behind resumes at the next clean
                # boundary rather than rapid-firing catch-up frames.
                if next_tick < now - interval:
                    next_tick = now + interval
                remaining = next_tick - time.perf_counter()
                if remaining > 0 and self._stop.wait(timeout=remaining):
                    return
        except Exception:
            logger.exception("Room pacer loop crashed")
            self._stop.set()

    def _emit_one_tick(self) -> None:
        """Send the next queued frame, the black frame a flush left, or nothing."""
        with self._room:
            try:
                item: MediaBundle | None = self._queue.get_nowait()
            except queue.Empty:
                item = None
            black = item is None and self._black_pending
            if item is not None or black:
                self._black_pending = False
            self._room.notify_all()
        if item is not None:
            video = item.get_tracks_by_kind(TrackKind.VIDEO)
            if video and video[0].data.ndim == 3:
                shape = video[0].data.shape
                self._frame_dims = (shape[0], shape[1], shape[2])
            self._dispatch(item)
        elif black and self._frame_dims is not None:
            self._dispatch(self._black_bundle(self._frame_dims))

    def _dispatch(self, bundle: MediaBundle) -> None:
        """Hand one frame to the sink, isolating its failure from the loop."""
        try:
            self._on_frame(bundle)
        except Exception:
            logger.exception("Room pacer frame sink raised; continuing")

    def _black_bundle(self, dims: tuple[int, int, int]) -> MediaBundle:
        black = np.zeros(dims, dtype=np.uint8)
        return MediaBundle(
            tracks={
                name: TrackData(info=info, data=black) for name, info in self._video_tracks.items()
            }
        )
