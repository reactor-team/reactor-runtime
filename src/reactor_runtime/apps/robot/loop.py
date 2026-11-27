"""The pieces of a robot policy's streamed loop: camera readers and the policy pacer.

In a streamed session every camera has its own reader, a task that takes each
frame off the camera's input track as it arrives and hands it on, so frames
reach the :class:`~reactor_runtime.apps.robot.sync.InputSynchronizer` while
``generate()`` runs. The policy loop asks for an observation at the pace a
:class:`PolicyPacer` sets.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from reactor_runtime.core.values import InputFrame
from reactor_runtime.interface.internal.input_buffer import BufferClosed, InputBuffer, ReadMode

READ_POLL_S = 0.1
"""How long one blocking read of a camera waits before its reader looks again."""


async def read_camera(
    name: str,
    buffer: InputBuffer,
    on_frame: Callable[[str, InputFrame], None],
) -> None:
    """Hand every frame of one camera to *on_frame*, oldest first, until cancelled.

    Each frame is handed on once, in arrival order, on the event loop. The
    reader waits for a frame on a worker thread, at most :data:`READ_POLL_S`
    at a time, so a cancelled reader frees its thread within that time; a
    frame that thread takes after the cancellation is dropped. The reader
    returns once the track has closed.

    Args:
        name: The camera's track name, passed to *on_frame*.
        buffer: The camera's input track.
        on_frame: Called with *name* and each frame.
    """
    while True:
        try:
            frames = await buffer.read(1, timeout=READ_POLL_S, mode=ReadMode.FIFO)
        except TimeoutError:
            continue
        except BufferClosed:
            return
        for frame in frames:
            on_frame(name, frame)


class PolicyPacer:
    """Space the steps of a policy loop at most one per period, on a fixed grid.

    With a rate, step starts fall on a grid of ``1 / fps`` seconds, so the
    rate does not drift with the time each step takes. A step that starts a
    whole period or more past its slot, after a slow ``generate()`` or a wait
    for an observation, starts a new grid from that moment: missed slots are
    skipped, never made up in a burst. Without a rate, steps run back to back.

    Args:
        fps: Steps per second, or ``None`` for back to back.

    Raises:
        ValueError: *fps* is not greater than zero.
    """

    def __init__(self, fps: float | None) -> None:
        """Create a pacer whose first step may start at once."""
        if fps is not None and not fps > 0:
            raise ValueError(f"a policy runs at a rate greater than zero, got fps={fps}")
        self._period = None if fps is None else 1.0 / fps
        self._due: float | None = None

    @property
    def period(self) -> float | None:
        """Seconds between two step starts, or ``None`` for back to back."""
        return self._period

    async def wait(self) -> None:
        """Wait until the next step may start."""
        if self._due is None:
            return
        delay = self._due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def started(self) -> None:
        """Note that a step starts now, which sets when the next one may."""
        if self._period is None:
            return
        now = time.monotonic()
        if self._due is None or now - self._due >= self._period:
            self._due = now + self._period
        else:
            self._due += self._period
