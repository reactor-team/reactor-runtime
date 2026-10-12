"""Thread-safe ring buffer backing one inbound media track.

The runtime pushes decoded frames in from its own thread; the model reads them
out from the model loop. :class:`InputBuffer` is the hand-off between the two — a
bounded, lock-guarded deque that wakes a blocked reader when a frame arrives or
the track closes. Model authors reach it through the :class:`MediaInput` track handle
(``await self.media.camera.read()``), never directly.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from enum import Enum, auto
from typing import NamedTuple

from reactor_runtime.core.values import ConnId, InputFrame

DEFAULT_BUFFER_CAPACITY = 128
"""How many frames a buffer holds before the oldest is evicted."""


class ReadMode(Enum):
    """Which frames a read returns.

    Attributes:
        FIFO: The oldest frames, consumed in arrival order.
        LATEST: The newest frames, discarding any older backlog.
    """

    FIFO = auto()
    LATEST = auto()


class BufferClosed(Exception):  # noqa: N818 — the established name a model author catches
    """Raised by a read once the track has closed and is drained."""


class _Entry(NamedTuple):
    """One buffered frame, with what the buffer keeps beside it."""

    session: int
    frame: InputFrame
    pushed_at: float
    """When the frame was pushed, on ``perf_counter``."""
    source: ConnId | None
    """The connection that sent the frame, or ``None``."""


BufferSnapshot = tuple[tuple[_Entry, ...], int]
"""What :meth:`InputBuffer.snapshot` returns and :meth:`InputBuffer.restore` takes."""


class InputBuffer:
    """A bounded, thread-safe frame buffer for one inbound track.

    "Input" is from the model's point of view: frames flow from the client into
    the model. The buffer is bounded — when full, the oldest frame is evicted so
    the model always sees the most recent data.

    The bound is a number of frames, or, for an audio track that declares
    ``buffer_seconds``, a number of samples. A frame bound holds a length of
    audio that depends on the transport's frame size; a sample bound holds the
    same length on every transport.

    Each frame is pushed for a session, by number. A new session can begin
    before the model has finished the last one, so a :meth:`reset` for the
    session that ended keeps any frame already pushed for a later one, and a
    push only counts the frames it evicts from its own session.

    A sample bound also caps the number of frames, so frames that carry no
    samples, such as a byte stream's end frames, cannot grow it without limit.

    Args:
        maxlen: How many frames the buffer holds before evicting the oldest.
            Ignored when *max_samples* is set.
        max_samples: How many audio samples the buffer holds before evicting
            the oldest frames, or ``None`` to bound it by *maxlen* instead.
        max_frames: Under a sample bound, how many frames the buffer holds
            whatever their samples. Defaults to *max_samples*.
    """

    def __init__(
        self,
        maxlen: int = DEFAULT_BUFFER_CAPACITY,
        *,
        max_samples: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        # One mutex shared between the plain lock and the condition: simple
        # acquire/release reads take the lock, wait/notify paths take the
        # condition, and a single underlying mutex rules out lock-ordering bugs.
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # Both bounds are enforced by push(), so the deque itself is unbounded.
        self._buffer: deque[_Entry] = deque()
        self._maxlen = maxlen
        self._max_samples = max_samples
        self._max_frames = max_frames if max_frames is not None else (max_samples or maxlen)
        self._samples = 0
        # The sender of each frame a read took, and how long it waited in the
        # buffer in ms, until the step loop takes them (take_read_waits). A
        # model with its own run() may never take them, so they are bounded
        # like the frames: the oldest go first. The default step loop takes
        # them after every read, so it never reaches the bound.
        self._read_waits: deque[tuple[ConnId | None, float]] = deque(maxlen=maxlen)
        self._total_received = 0
        self._closed = False

    async def read(
        self,
        n: int = 1,
        timeout: float | None = None,  # noqa: ASYNC109 (a threaded wait, not loop cancellation)
        mode: ReadMode = ReadMode.LATEST,
    ) -> list[InputFrame]:
        """Return *n* frames, waiting on the model loop until they arrive.

        Runs the blocking wait on a worker thread so the model's event loop keeps
        dispatching handlers while the read is pending.

        Cancellation: a started worker thread cannot be cancelled, so an
        indefinite read (``timeout=None``) parks until a frame arrives or the
        track is closed. The runtime closes a track on teardown and disconnect,
        which wakes a parked read with :class:`BufferClosed` and frees the worker;
        :meth:`reset` re-opens only after that close. A read is therefore released
        by closing the track, not by cancelling the awaiting task — pass *timeout*
        when a bounded wait is wanted.

        Args:
            n: How many frames to return.
            timeout: Seconds to wait, or ``None`` to wait indefinitely.
            mode: Whether to return the newest or the oldest frames.

        Returns:
            Exactly *n* frames.

        Raises:
            BufferClosed: The track closed before *n* frames were available.
            TimeoutError: *timeout* elapsed first.
            ValueError: *n* exceeds the buffer's frame bound. Under a
                ``buffer_seconds`` bound, *n* frames can also fail to fit by
                their samples; such a read waits, so pass *timeout* to bound it.
        """
        return await asyncio.to_thread(self._read, n, timeout, mode)

    def _read(self, n: int, timeout: float | None, mode: ReadMode) -> list[InputFrame]:
        frame_bound = self._maxlen if self._max_samples is None else self._max_frames
        if n > frame_bound:
            raise ValueError(f"requested {n} frames but the buffer holds at most {frame_bound}")
        with self._condition:
            if not self._condition.wait_for(
                lambda: len(self._buffer) >= n or self._closed, timeout=timeout
            ):
                raise TimeoutError(f"timed out waiting for {n} frame(s) after {timeout}s")
            if self._closed and len(self._buffer) < n:
                raise BufferClosed("buffer closed")
            if mode is ReadMode.LATEST:
                return self._take_latest(n)
            return self._take_fifo(n)

    def _take_latest(self, n: int) -> list[InputFrame]:
        """Return the *n* newest frames and drop the rest. Caller holds the lock."""
        taken = list(self._buffer)[len(self._buffer) - n :]
        self._note_waits(taken)
        self._buffer.clear()
        self._samples = 0
        return [entry.frame for entry in taken]

    def _take_fifo(self, n: int) -> list[InputFrame]:
        """Pop the *n* oldest frames. Caller holds the lock."""
        taken = [self._buffer.popleft() for _ in range(n)]
        self._note_waits(taken)
        if self._max_samples is not None:
            self._samples -= sum(_sample_count(entry.frame) for entry in taken)
        return [entry.frame for entry in taken]

    def _note_waits(self, taken: list[_Entry]) -> None:
        """Note how long the frames *taken* waited. Caller holds the lock."""
        now = time.perf_counter()
        self._read_waits.extend((entry.source, (now - entry.pushed_at) * 1000.0) for entry in taken)

    def take_read_waits(self) -> list[tuple[ConnId | None, float]]:
        """Return each frame read since the last take, as its sender and its wait in ms.

        The sender is the connection that pushed the frame, or ``None`` when
        the push named none. The waits start again.
        """
        with self._lock:
            waits = list(self._read_waits)
            self._read_waits.clear()
        return waits

    def try_read(self, n: int = 1, mode: ReadMode = ReadMode.LATEST) -> list[InputFrame] | None:
        """Return *n* frames without blocking, or ``None`` when too few are ready.

        Args:
            n: How many frames to return.
            mode: Whether to return the newest or the oldest frames.

        Returns:
            *n* frames, or ``None`` if fewer are available (the buffer is left
            untouched).

        Raises:
            BufferClosed: The track has closed.
        """
        with self._lock:
            if self._closed:
                raise BufferClosed("buffer closed")
            if len(self._buffer) < n:
                return None
            if mode is ReadMode.LATEST:
                return self._take_latest(n)
            return self._take_fifo(n)

    @property
    def available(self) -> int:
        """How many frames are buffered right now."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_received(self) -> int:
        """How many frames have arrived since the last reset."""
        with self._lock:
            return self._total_received

    @property
    def closed(self) -> bool:
        """Whether the track has closed."""
        with self._lock:
            return self._closed

    def clear(self) -> None:
        """Drop every buffered frame without closing the track."""
        with self._lock:
            self._buffer.clear()
            self._samples = 0

    def close(self) -> None:
        """Close the track, waking any blocked reader to raise :class:`BufferClosed`."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reset(self, through_session: int | None = None) -> None:
        """Re-open the track for a new session, dropping the ended sessions' frames.

        Follows :meth:`close`: closing releases any parked reader, and reset then
        re-opens the drained track. Re-opening without closing first would leave a
        parked indefinite read waiting against the fresh buffer.

        Args:
            through_session: The session that ended. Frames pushed for it, or
                for an earlier one, are dropped; frames already pushed for a
                later session are kept, with their count. ``None`` drops every
                frame.
        """
        with self._condition:
            self._closed = False
            if through_session is None:
                self._buffer.clear()
            else:
                self._buffer = deque(e for e in self._buffer if e.session > through_session)
            self._samples = sum(_sample_count(entry.frame) for entry in self._buffer)
            self._read_waits.clear()
            self._total_received = len(self._buffer)

    def snapshot(self) -> BufferSnapshot:
        """Return the buffered frames and the received count, for :meth:`restore`."""
        with self._lock:
            return tuple(self._buffer), self._total_received

    def restore(self, snapshot: BufferSnapshot) -> None:
        """Put back the frames and the received count a :meth:`snapshot` returned."""
        entries, total_received = snapshot
        with self._condition:
            self._buffer.clear()
            self._buffer.extend(entries)
            self._samples = sum(_sample_count(entry.frame) for entry in self._buffer)
            self._total_received = total_received

    def push(self, frame: InputFrame, source: ConnId | None = None, session: int = 0) -> int:
        """Append a frame from the runtime thread, waking a blocked reader.

        Frames pushed after :meth:`close` are dropped until :meth:`reset`, and
        are not counted as evicted: the track is not being read.

        Args:
            frame: The frame to buffer.
            source: The connection that sent the frame, so the frame's wait in
                the buffer is reported to that connection only.
            session: The session the frame belongs to, by number.

        Returns:
            How many frames of the same session the bound evicted to make room.
            Frames of an earlier session are evicted too, but not counted: they
            belonged to a session that has ended. Under a sample bound the
            newest frame is always kept, even when it alone holds more samples
            than the bound.
        """
        with self._condition:
            if self._closed:
                return 0
            self._buffer.append(_Entry(session, frame, time.perf_counter(), source))
            if self._max_samples is not None:
                self._samples += _sample_count(frame)
            evicted = 0
            while len(self._buffer) > 1 and self._over_bound():
                gone = self._buffer.popleft()
                if self._max_samples is not None:
                    self._samples -= _sample_count(gone.frame)
                if gone.session == session:
                    evicted += 1
            self._total_received += 1
            self._condition.notify_all()
            return evicted

    def _over_bound(self) -> bool:
        """Whether the buffer holds more than its bound. Caller holds the lock."""
        if self._max_samples is not None:
            return self._samples > self._max_samples or len(self._buffer) > self._max_frames
        return len(self._buffer) > self._maxlen


def _sample_count(frame: InputFrame) -> int:
    """How many samples an ``(1, M)`` audio frame carries."""
    return int(frame.data.shape[-1])
