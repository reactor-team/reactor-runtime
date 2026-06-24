"""Thread-safe ring buffer backing one inbound media track.

The runtime pushes decoded frames in from its own thread; the model reads them
out from the model loop. :class:`InputBuffer` is the hand-off between the two — a
bounded, lock-guarded deque that wakes a blocked reader when a frame arrives or
the track closes. Model authors reach it through the :class:`Input` track handle
(``await self.input.camera.read()``), never directly.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from enum import Enum, auto

from reactor_runtime.core.values import InputFrame

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


class InputBuffer:
    """A bounded, thread-safe frame buffer for one inbound track.

    "Input" is from the model's point of view: frames flow from the client into
    the model. The buffer is a bounded deque — when full, the oldest frame is
    evicted so the model always sees the most recent data.

    Args:
        maxlen: How many frames the buffer holds before evicting the oldest.
    """

    def __init__(self, maxlen: int = DEFAULT_BUFFER_CAPACITY) -> None:
        # One mutex shared between the plain lock and the condition: simple
        # acquire/release reads take the lock, wait/notify paths take the
        # condition, and a single underlying mutex rules out lock-ordering bugs.
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._buffer: deque[InputFrame] = deque(maxlen=maxlen)
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
            ValueError: *n* exceeds the buffer capacity.
        """
        return await asyncio.to_thread(self._read, n, timeout, mode)

    def _read(self, n: int, timeout: float | None, mode: ReadMode) -> list[InputFrame]:
        if self._buffer.maxlen is not None and n > self._buffer.maxlen:
            raise ValueError(
                f"requested {n} frames but the buffer holds at most {self._buffer.maxlen}"
            )
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
        start = len(self._buffer) - n
        result = [self._buffer[start + i] for i in range(n)]
        self._buffer.clear()
        return result

    def _take_fifo(self, n: int) -> list[InputFrame]:
        """Pop the *n* oldest frames. Caller holds the lock."""
        return [self._buffer.popleft() for _ in range(n)]

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

    def close(self) -> None:
        """Close the track, waking any blocked reader to raise :class:`BufferClosed`."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reset(self) -> None:
        """Re-open a closed track for a new session, dropping frames and counters.

        Follows :meth:`close`: closing releases any parked reader, and reset then
        re-opens the drained track. Re-opening without closing first would leave a
        parked indefinite read waiting against the fresh buffer.
        """
        with self._condition:
            self._closed = False
            self._buffer.clear()
            self._total_received = 0

    def push(self, frame: InputFrame) -> None:
        """Append a frame from the runtime thread, waking a blocked reader.

        Frames pushed after :meth:`close` are dropped until :meth:`reset`.
        """
        with self._condition:
            if self._closed:
                return
            self._buffer.append(frame)
            self._total_received += 1
            self._condition.notify_all()
