"""Per-connection media pacer.

A model emits finished media in bursts, one :class:`MediaChunk` per inference,
at whatever rate its compute finishes; a client needs frames at a smooth, steady
cadence. :class:`MediaPacer` is the seam between the two on one WebRTC
connection: it takes the connection's share of each chunk, splits any batch into
single frames, and a dedicated thread drains them to the wire at the chunk's
declared rate. When the queue runs dry nothing is sent — the client keeps the
frame it has — and a single black frame marks each boundary (connection start,
or a flush) so the wire never replays stale content.

One pacer lives per connection and dies with it, so there is no cross-session
state to reset — a fresh connection starts with a fresh pacer. Everything above it hands the
pacer a chunk and never sees a frame.

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
    frame is ready nothing is sent — the client keeps its current frame;
    a single black frame stands in at each boundary (connection start, or a
    flush) before real media arrives.

    Args:
        video_tracks: The connection's outbound video tracks, used to synthesise
            black frames before the first real frame arrives.
        on_frame: The sink one single-frame bundle is handed to each tick.
        queue_depth: How many frames may sit queued between the model and the
            wire — the buffered-latency bound. Never applied below one chunk,
            so a model that batches always fits a whole chunk.
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

        # The queue is unbounded in itself; the depth is the bound, checked on
        # submit. The effective capacity never sits below one chunk, so a
        # batching model always fits a whole chunk regardless of the depth.
        self._queue: queue.Queue[MediaBundle] = queue.Queue()
        self._depth = queue_depth
        # Signalled by the pacing thread after each dequeue so a blocking
        # submit (chunk.wait) can sleep until room opens instead of polling.
        self._room = threading.Condition()
        # Bumped by flush(); a submit in flight notices and abandons the
        # rest of its chunk, so flushed content cannot trickle back in.
        self._epoch = 0

        self._interval = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self._thread: threading.Thread | None = None
        # The single source of truth for whether pacing runs: set = stopped.
        self._stop = threading.Event()
        self._stop.set()
        self._lifecycle_lock = threading.Lock()

        # One black frame marks a boundary (start of the connection, or a
        # flush); after it the pacer stays silent until real media arrives,
        # so the wire carries only frames the model actually produced.
        self._boundary_black_pending = True
        self._frame_dims: tuple[int, int, int] | None = None

    def submit(self, chunk: MediaChunk) -> int:
        """Split *chunk* into single frames, adopt its rate, and enqueue them.

        The pacing rate is updated to the chunk's ``fps`` so a dynamic-rate model
        paces at the throughput it is actually producing. The queue bound is the
        configured depth, never applied below one chunk, so a batching model
        always fits a whole chunk.

        When the chunk asks for backpressure (``chunk.wait``), a full queue makes
        this call wait until the pacing thread drains room — throttling the
        producer to the playout rate. Otherwise frames beyond the capacity are
        dropped from the tail, so a producer that prefers skipping to waiting
        never stalls.

        Args:
            chunk: A finished media chunk, possibly batched.

        Returns:
            The number of frames actually enqueued.
        """
        if chunk.fps > 0:
            self._interval = 1.0 / chunk.fps
        epoch = self._epoch
        enqueued = 0
        aborted = False
        for frame in chunk.frames():
            # One authoritative bound: the configured depth, never below one
            # chunk so a batching model's whole chunk always fits. Re-read
            # each iteration so a concurrent set_depth applies mid-chunk.
            capacity = max(self._depth, chunk.n_frames)
            if chunk.wait:
                with self._room:
                    while (
                        not self._stop.is_set()
                        and self._epoch == epoch
                        and self._queue.qsize() >= capacity
                    ):
                        self._room.wait(timeout=0.1)
                if self._stop.is_set() or self._epoch != epoch:
                    aborted = True
                    break
            else:
                # A flush mid-chunk abandons the rest of it on this path too;
                # the flushed run's tail must not trickle in after the cut.
                if self._epoch != epoch:
                    aborted = True
                    break
                if self._queue.qsize() >= capacity:
                    break
            self._queue.put_nowait(frame)
            enqueued += 1
        if enqueued < chunk.n_frames and not chunk.wait and not aborted:
            logger.warning(
                "Media pacer queue full; dropped %d of %d frames",
                chunk.n_frames - enqueued,
                chunk.n_frames,
            )
        return enqueued

    def set_rate(self, fps: float) -> None:
        """Set the playout rate, re-pacing already-queued frames immediately.

        A subsequent chunk's own ``fps`` tag supersedes it, so this is the
        between-emits control rather than a pin.
        """
        if fps > 0:
            self._interval = 1.0 / fps

    def set_depth(self, depth: int) -> None:
        """Set the queue bound, waking any producer blocked on the old one."""
        if depth <= 0:
            return
        with self._room:
            self._depth = depth
            self._room.notify_all()

    def flush(self) -> None:
        """Drop queued frames and cut playout to black.

        The queue is drained, the boundary black re-armed, and any producer
        blocked in a backpressure submit abandons the rest of its chunk — so
        the next tick emits a black frame and none of the flushed content
        plays afterwards.
        """
        with self._room:
            self._epoch += 1
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._boundary_black_pending = True
            self._room.notify_all()

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
        # Wake any producer blocked in a backpressure submit, so teardown
        # cannot deadlock behind a queue nobody will drain.
        with self._room:
            self._room.notify_all()
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
        """Emit the next queued frame, or the one boundary black, or nothing.

        An empty queue sends nothing: the client keeps showing the frame it
        already has, and no bandwidth is spent repeating it. The single black
        frame marks a boundary (connection start or flush) so the client
        transitions off stale content exactly once.
        """
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            item = None
        else:
            with self._room:
                self._room.notify_all()

        if item is not None:
            video = item.get_tracks_by_kind(TrackKind.VIDEO)
            if video and video[0].data.ndim == 3:
                shape = video[0].data.shape
                self._frame_dims = (shape[0], shape[1], shape[2])
            self._boundary_black_pending = False
            self._dispatch(item)
        elif self._boundary_black_pending:
            self._boundary_black_pending = False
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

    def _black_bundle(self) -> MediaBundle:
        """Build a black frame for each outbound video track."""
        dims = self._frame_dims or DEFAULT_FRAME_DIMENSIONS
        black = np.zeros(dims, dtype=np.uint8)
        return MediaBundle(
            tracks={
                name: TrackData(info=info, data=black) for name, info in self._video_tracks.items()
            }
        )
