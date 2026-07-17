"""The engine half of the model layer — :class:`ReactorCore`.

The machinery a model author never touches: the model's own thread and asyncio
loop, its media buffers, the two typed inbound queues, and the outbound slots.
``ReactorCore`` owns the *how*; :class:`ReactorModel` supplies the *what* —
handler semantics — by overriding the loop hooks. Everything that reaches the
model from the outside arrives through a handful of thread-safe entrypoints,
which is what keeps the bridge above it thin.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar, get_type_hints

from reactor_runtime.core.model import Command, ReactorEvent
from reactor_runtime.core.values import (
    ConnId,
    InputFrame,
    MediaBundle,
    MediaChunk,
    TrackData,
)
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.interface.tracks import Input, Output

logger = logging.getLogger(__name__)

RequestId = str
"""A client-originated request id, carried end-to-end so a reply can correlate."""

_Holder = TypeVar("_Holder")

BroadcastSink = Callable[[ModelMessage], None]

AddressedSink = Callable[[ConnId, "ModelMessage | None", RequestId | None], None]
"""Delivers a reply to one connection, correlated by its request id.

A ``None`` message is the bodyless acknowledgement of a command whose handler
completed without returning one, so the client's awaited command resolves.
"""

MediaSink = Callable[[MediaChunk], None]
"""Receives each finished media chunk the model emits, unpaced, on the model thread."""

FailureSink = Callable[[BaseException], None]
"""Receives the exception that ended :meth:`ReactorCore.run`, on the model thread."""


@dataclass(frozen=True)
class CommandEnvelope:
    """A validated command plus the addressing a reply needs.

    Attributes:
        command: The validated command to dispatch.
        conn_id: The connection that sent it, when known.
        request_id: The request id to correlate a reply against, when present.
    """

    command: Command
    conn_id: ConnId | None
    request_id: RequestId | None


class ReactorCore:
    """The model's loop, buffers, queues, and outbound slots.

    Subclassed by :class:`ReactorModel`, which fills the loop hooks with the two
    dispatchers. On its own, ``ReactorCore`` accepts inbound traffic onto the two
    queues and routes media into the input buffers; nothing drains the queues
    until a subclass supplies the drain loops via :meth:`_background_coros`.
    """

    fps: ClassVar[int] = 30

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._loop_task: asyncio.Task[None] | None = None

        self._command_q: asyncio.Queue[CommandEnvelope] | None = None
        self._reactor_q: asyncio.Queue[ReactorEvent] | None = None

        self._out_broadcast: BroadcastSink | None = None
        self._out_addressed: AddressedSink | None = None
        self._out_media: MediaSink | None = None
        self._on_failure: FailureSink | None = None

        self._input_buffers: dict[str, InputBuffer] = {}
        self._wire_input_buffers()

    # -- author hooks ---------------------------------------------------------

    def load(self, config_path: Path | None) -> None:
        """Load weights and allocate resources, once, before any client connects.

        Args:
            config_path: Path to the model's config file (from ``runtime.config``
                in ``reactor.yaml``), or ``None`` when none is configured. The
                runtime does not parse it; the model reads and interprets it
                however it wants (for example ``yaml.safe_load`` the contents).
        """

    async def run(self) -> None:
        """Drive the model. Overridden by :class:`ReactorModel`."""
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        """Hand a finished output downstream as a media chunk.

        Converts the typed *output* into a neutral bundle and hands it to the
        bound media sink as a :class:`MediaChunk`, tagged with the rate its
        frames should play out at: the measured throughput when *compute_time* is
        given, else the model's declared :attr:`fps`. Emission does not pace or
        block — a consumer (a transport connection, a recorder) owns pacing and
        any overflow handling downstream.

        Args:
            output: The produced output, one payload per declared track.
            compute_time: Wall-clock seconds spent producing it, if measured.
            drop: Retained for source compatibility; overflow is now handled by
                each downstream consumer, so this has no effect here.
        """
        bundle = self._to_bundle(output)
        n_frames = bundle.frame_count
        if compute_time is not None and compute_time > 0:
            fps = n_frames / compute_time
        else:
            fps = float(self.fps)
        if self._out_media is not None:
            self._out_media(MediaChunk(bundle=bundle, fps=fps, n_frames=n_frames))
        await asyncio.sleep(0)

    async def send(self, message: ModelMessage) -> None:
        """Broadcast a typed message to every connected client."""
        if self._out_broadcast is not None:
            self._out_broadcast(message)

    # -- outbound binding (called once by the bridge) -------------------------

    def bind_output(
        self, *, broadcast: BroadcastSink, addressed: AddressedSink, media: MediaSink
    ) -> None:
        """Bind the outbound sinks. Called once before the loop starts."""
        self._out_broadcast = broadcast
        self._out_addressed = addressed
        self._out_media = media

    def bind_failure(self, callback: FailureSink) -> None:
        """Bind the sink that receives an unrecoverable crash of :meth:`run`.

        The callback fires at most once, on the model thread, after the loop's
        background tasks have been cancelled. A caller that lives on another
        loop marshals from within the callback.
        """
        self._on_failure = callback

    # -- thread-safe ingress (called by the bridge from the runtime thread) ---

    def submit_command(
        self, command: Command, conn_id: ConnId | None, request_id: RequestId | None
    ) -> None:
        """Enqueue a validated command onto the model loop."""
        self._enqueue(self._command_q, CommandEnvelope(command, conn_id, request_id))

    def post_reactor_event(self, event: ReactorEvent) -> None:
        """Enqueue a reactor-authoritative event onto the model loop."""
        self._enqueue(self._reactor_q, event)

    def push_media(self, track: str, frame: InputFrame) -> None:
        """Route an inbound frame into its track's buffer."""
        if self._loop.is_closed():
            return
        buffer = self._input_buffers.get(track)
        if buffer is None:
            logger.debug("media for unknown input track %r", track)
            return
        buffer.push(frame)

    def _enqueue(self, queue: asyncio.Queue[Any] | None, item: Any) -> None:
        """Thread-safe put onto a loop-bound queue, dropped before the loop runs."""
        if queue is None or self._loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(queue.put_nowait, item)

    # -- thread lifecycle (start_thread is the only thing that spawns) --------

    def start_thread(self) -> None:
        """Spawn the model thread running its own asyncio loop."""
        self._thread = threading.Thread(target=self._run, name="model", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Cancel the model loop from another thread."""
        if self._loop_task is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._loop_task.cancel)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._lifecycle())
        finally:
            self._loop.close()

    async def _lifecycle(self) -> None:
        """Bootstrap loop-bound state, start the drain loops, then run the model.

        A ``run()`` that raises anything but cancellation is an unrecoverable
        crash: the model loop is gone and there is nothing left to serve. The
        crash is reported through the bound failure sink once the background
        tasks are cancelled, so the owner can end the session rather than keep
        serving a dead loop. Cancellation is the normal stop and reports nothing.
        """
        self._command_q = asyncio.Queue()
        self._reactor_q = asyncio.Queue()
        self._loop_task = asyncio.current_task()
        self._on_loop_ready()
        tasks = [asyncio.create_task(coro) for coro in self._background_coros()]
        crash: BaseException | None = None
        try:
            await self.run()
        except asyncio.CancelledError:
            logger.info("model loop cancelled")
        except Exception as exc:
            crash = exc
            logger.exception("model run() crashed")
        finally:
            for task in tasks:
                task.cancel()
        if crash is not None and self._on_failure is not None:
            self._on_failure(crash)

    # -- subclass loop hooks --------------------------------------------------

    def _on_loop_ready(self) -> None:
        """Create loop-bound state. Overridden by subclasses."""

    def _background_coros(self) -> list[Coroutine[Any, Any, None]]:
        """Return the coroutines to run alongside ``run()``. Overridden by subclasses."""
        return []

    # -- internals ------------------------------------------------------------

    def _to_bundle(self, output: Output) -> MediaBundle:
        """Convert a typed output into a neutral media bundle."""
        tracks = {
            name: TrackData(info=info, data=getattr(output, name))
            for name, info in type(output).__tracks__.items()
        }
        return MediaBundle(tracks=tracks)

    def _wire_input_buffers(self) -> None:
        """Build a buffer per inbound track and bind the readable input handle."""
        holder = self._find_holder(Input)
        if holder is None:
            return
        attr_name, input_cls = holder
        buffers = {name: InputBuffer() for name in input_cls.__tracks__}
        self._input_buffers.update(buffers)
        setattr(self, attr_name, input_cls(**buffers))

    @classmethod
    def _find_holder(cls, base: type[_Holder]) -> tuple[str, type[_Holder]] | None:
        """Find the attribute annotated as a concrete subclass of *base*."""
        try:
            hints = get_type_hints(cls)
        except Exception:
            return None
        for attr_name, hint in hints.items():
            if isinstance(hint, type) and issubclass(hint, base) and hint is not base:
                return attr_name, hint
        return None
