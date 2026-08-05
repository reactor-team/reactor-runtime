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
    CommandFailure,
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

AddressedSink = Callable[[ConnId, "ModelMessage | CommandFailure | None", RequestId | None], None]
"""Delivers a reply to one connection, correlated by its request id.

A :class:`ModelMessage` is the handler's typed answer. A :class:`CommandFailure`
is the reason it could not produce one. A ``None`` message is the bodyless
acknowledgement of a command whose handler completed without returning one. All
three resolve the client's awaited command, so it never waits on a reply that
does not come.
"""

MediaSink = Callable[[MediaChunk], None]
"""Receives each finished media chunk the model emits, unpaced, on the model thread."""

FailureSink = Callable[[BaseException], None]
"""Receives the exception that ended :meth:`ReactorCore.run`, on the model thread."""


@dataclass(frozen=True)
class MediaOps:
    """Operations the runtime exposes over its downstream media consumers.

    Bound alongside the sinks; the model reaches them through its
    :attr:`ReactorCore.output` handle. Like the sinks, each operation is a
    one-way call carrying only plain values, so the boundary between the
    model and the runtime stays a set of serialisable messages. Each fans
    out to every live connection and is remembered for connections that
    open later.

    Attributes:
        flush: Drop queued media and cut playout to black.
        set_rate: Re-pace queued media at a new frames-per-second immediately.
        set_depth: Bound how many frames may queue between model and wire.
    """

    flush: Callable[[], None]
    set_rate: Callable[[float], None]
    set_depth: Callable[[int], None]


class OutputStream:
    """The model's handle onto its outbound media stream, bound as ``self.output``.

    Emission and playout control in one place: :meth:`emit` hands a finished
    output downstream, :attr:`fps` is the live playout rate, and :meth:`flush`
    drops everything queued and cuts playout to black.

    There is no single buffer behind this handle — queued media lives
    downstream, in each connection's own pacer — so every operation fans out
    to all live connections and applies to connections that join later.
    Operations invoked before the runtime binds its consumers (for example
    inside ``load()``) are remembered and applied at bind time.
    """

    def __init__(self, core: ReactorCore) -> None:
        self._core = core

    @property
    def fps(self) -> float:
        """The playout rate, in frames per second.

        Reading it returns the rate unmeasured chunks are tagged with — the
        model's declared :attr:`ReactorCore.fps` until it is reassigned here.
        Assigning re-paces already-queued frames on every connection
        immediately and tags subsequent emits with the new rate. A later
        chunk emitted with a measured ``compute_time`` supersedes it, so this
        is the between-emits control rather than a pin — on a pipeline that
        declares no ``fps`` (driven with measured throughput every yield)
        the assignment lasts one chunk.
        """
        return float(self._core.fps)

    @fps.setter
    def fps(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._core.fps = fps
        ops = self._core._media_ops
        if ops is not None:
            ops.set_rate(float(fps))

    async def emit(
        self,
        output: Output,
        *,
        compute_time: float | None = None,
        drop: bool = False,
    ) -> None:
        """Hand a finished output downstream as a media chunk.

        Converts the typed *output* into a neutral bundle and hands it to the
        bound media sink as a :class:`MediaChunk`, tagged with the rate its
        frames should play out at: the measured throughput when *compute_time*
        is given, else :attr:`fps`.

        By default this call waits while a downstream consumer's bounded
        queue is full, throttling a producer that outruns the playout rate.
        The wait runs on a worker thread, so the model loop keeps dispatching
        commands and lifecycle events. Pass ``drop=True`` for a producer that
        prefers skipping frames to waiting — the overflow is discarded
        downstream and this call returns as soon as the chunk is handed over.

        Args:
            output: The produced output, one payload per declared track.
            compute_time: Wall-clock seconds spent producing it, if measured.
            drop: Discard overflow downstream instead of waiting for room.
        """
        core = self._core
        bundle = core._to_bundle(output)
        n_frames = bundle.frame_count
        if compute_time is not None and compute_time > 0:
            fps = n_frames / compute_time
        else:
            fps = float(core.fps)
        if core._out_media is not None:
            chunk = MediaChunk(bundle=bundle, fps=fps, n_frames=n_frames, wait=not drop)
            await asyncio.to_thread(core._out_media, chunk)
        await asyncio.sleep(0)

    def flush(self) -> None:
        """Drop queued media on every connection and cut playout to black.

        Call it when generation resets or restarts, so nothing of the old
        content plays after the cut. Releases a producer blocked in
        :meth:`emit` and discards the rest of its chunk. The session
        recording is not flushed — a playout cut is not an archive boundary.
        """
        ops = self._core._media_ops
        if ops is not None:
            ops.flush()


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

    fps: float = 30.0
    """The declared playout rate, tagging unmeasured emits. Adjustable at any
    time through :attr:`OutputStream.fps` on the :attr:`output` handle."""

    buffer_size: ClassVar[int | None] = None
    """How many frames may queue between the model and each wire — the
    buffered-latency bound a model declares. ``None`` accepts the runtime
    default; a declared value must be positive (rejected at startup
    otherwise), and the bound is never applied below one emitted chunk."""

    output: OutputStream
    """The model's handle onto its outbound media stream."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._loop_task: asyncio.Task[None] | None = None

        self._command_q: asyncio.Queue[CommandEnvelope] | None = None
        self._reactor_q: asyncio.Queue[ReactorEvent] | None = None

        self._out_broadcast: BroadcastSink | None = None
        self._out_addressed: AddressedSink | None = None
        self._out_media: MediaSink | None = None
        self._media_ops: MediaOps | None = None
        self._on_failure: FailureSink | None = None
        self.output = OutputStream(self)

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
        """Hand a finished output downstream — an alias of ``self.output.emit``.

        See :meth:`OutputStream.emit` for the contract. By default the call
        waits while downstream is full, throttling the model to the playout
        rate; ``drop=True`` discards the overflow instead.

        Args:
            output: The produced output, one payload per declared track.
            compute_time: Wall-clock seconds spent producing it, if measured.
            drop: Discard overflow downstream instead of waiting for room.
        """
        await self.output.emit(output, compute_time=compute_time, drop=drop)

    async def send(self, message: ModelMessage) -> None:
        """Broadcast a typed message to every connected client."""
        if self._out_broadcast is not None:
            self._out_broadcast(message)

    # -- outbound binding (called once by the bridge) -------------------------

    def bind_output(
        self,
        *,
        broadcast: BroadcastSink,
        addressed: AddressedSink,
        media: MediaSink,
        media_ops: MediaOps | None = None,
    ) -> None:
        """Bind the outbound sinks. Called once before the loop starts.

        Pushes the playout settings declared or set before binding — the
        model's ``buffer_size`` class attribute, and a rate assigned to
        ``self.output.fps`` during ``load()`` — down to the media consumers.

        Raises:
            ValueError: If the model declares a non-positive ``buffer_size``.
                Binding runs before the loop starts, so this surfaces as a
                startup failure rather than a silently inherited default.
        """
        if self.buffer_size is not None and self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        self._out_broadcast = broadcast
        self._out_addressed = addressed
        self._out_media = media
        self._media_ops = media_ops
        if media_ops is not None:
            if self.buffer_size is not None:
                media_ops.set_depth(self.buffer_size)
            if "fps" in self.__dict__:
                media_ops.set_rate(float(self.fps))

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
