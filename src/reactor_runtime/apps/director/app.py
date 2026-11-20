"""The director base class, :class:`ReactorDirectorApp`.

A model that builds a whole shot of video in one call writes ``load()`` and
``generate()``. This class does the rest: it turns ``generate()``'s parameters
into the ``enqueue`` command, keeps the generation queue, the playout queue and
the history, builds shots in the background, plays them on the output tracks,
lets one shot continue another, and reports every shot as a step.

It is a :class:`ReactorApp` that replaces only the loop. It overrides the
streamed half of the loop and the step a client requests, never ``run()``, so
the same class serves streamed sessions, sessions started with a starting input
and ``steps``, and stepped sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from reactor_runtime.apps.director.contract import DirectorSpec, build_spec, describe
from reactor_runtime.apps.director.media import (
    decode_file,
    frame_count,
    last_frame,
    png_bytes,
    read_file_video,
    slice_output,
)
from reactor_runtime.apps.director.shots import (
    History,
    ShotEntry,
    ShotQueue,
    ShotStatus,
    dependents,
)
from reactor_runtime.core.fields import InputField
from reactor_runtime.core.local_file import LocalFile
from reactor_runtime.core.model import ClientConnected, ReactorEvent, SessionEnded, SessionStarted
from reactor_runtime.core.values import RequestedStep
from reactor_runtime.interface.app.reactor_app import ReactorApp
from reactor_runtime.interface.app.step_completed import StepCompleted
from reactor_runtime.interface.events.decorators import (
    EVENT_ATTR,
    EventHandler,
    event,
    make_command,
)
from reactor_runtime.interface.events.errors import CommandError
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.reactor_core import StepEnvelope
from reactor_runtime.interface.tracks import Output
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

EMIT_FRAMES = 3
"""Frames per emitted slice. Small slices keep playout even and a stop quick."""

_TAIL_SLICES = 2
"""Slices a finished shot waits before it cuts to black, so its last frames play out."""

_SETTINGS_COMMANDS = ("set_autoplay", "set_build_paused", "set_hold_last_frame")


def _is_fatal(error: BaseException) -> bool:
    """Return whether *error* means the model itself is gone, not only this shot."""
    from reactor_runtime.distributed.errors import RankDesync, WorkerCrashed, WorkerTimeout
    from reactor_runtime.model_server.errors import ModelServerLost, StepUnanswered

    return isinstance(
        error, WorkerCrashed | WorkerTimeout | RankDesync | ModelServerLost | StepUnanswered
    )


@dataclass
class _Clock:
    """The playout clock, carried across shots that continue each other."""

    start: float | None = None
    frames: int = 0


class ReactorDirectorApp(ReactorApp):
    """Base class for a model that builds whole shots and plays them when a client asks.

    Write ``load()`` and ``generate()``. The parameters of ``generate()`` are
    the inputs a client sends with ``enqueue``, declared and validated as the
    parameters of an ``@event`` handler are; its return value is the shot::

        class Reel(ReactorDirectorApp):
            fps = 24

            def load(self, config_path):
                self.engine = ReelModel(config_path)

            def generate(
                self,
                prompt: str = InputField(moderate=True),
                continue_from: Shot | None = None,
            ) -> ReelOutput:
                start = continue_from.last_frame if continue_from else None
                video, audio = self.engine.generate(prompt, start)
                return ReelOutput(main_video=video, main_audio=audio)

    **Inputs.** Each parameter becomes a field of ``enqueue``. The class adds
    two of its own: ``metadata``, a string echoed back on every message about
    the shot, and ``position``, where the shot enters the generation queue.
    A parameter may not take either name, nor ``client``, and must be one a
    client can name: no positional-only or variadic parameters.

    **Continuing a shot.** A parameter typed :class:`Shot`, or ``Shot | None``
    defaulting to ``None``, takes a shot id on the wire. ``enqueue`` refuses an
    id that is not queued, built, or in the history. A shot builds only once
    every shot it names is built, and receives each as a :class:`Shot`; it
    fails when one of them fails. With autoplay on, a shot whose
    ``continue_from`` names the shot that just played plays straight on from
    it, with no cut and no black frame.

    **What ``generate()`` returns.**

    * An :class:`Output` is played on its tracks at :attr:`fps` and saved as
      the step's ``output.mp4``.
    * A :class:`LocalFile` is kept byte for byte as the step's output and is
      not decoded for it. Set :attr:`plays` to also play it: the class decodes
      it onto those tracks when its turn comes. The file belongs to the class
      once returned, which deletes it when no step and no playout still need
      it.

    Anything else fails the shot. A plain ``def generate()`` runs on a thread
    of its own, so commands and playout go on while it works; an
    ``async def generate()`` runs on the model's event loop.

    **Failure.** An exception from ``generate()`` fails that shot: clients get
    ``shot_failed`` with the reason, the shot reports a failed step, the shots
    that depend on it fail too, and the session goes on. An error that means
    the model process is gone (a lost model server, a crashed worker) ends the
    model loop, as a raise out of ``process_output()`` does.

    **Steps.** In a streamed session every shot that leaves the generation
    queue by building, or by failing, reports one step: its media or file,
    ``last_frame.png`` when it has video, and ``shot.json``, its description.
    A shot removed with ``cancel`` or ``reset``, or dropped when the session
    ends, reports none. In a stepped session each request builds the
    front-most shot that can build, after the request's commands, and returns
    it as the step's result, with its description as the result's data. The
    shot goes to the history, not to the playout queue. A request with no
    shot that can build is refused.

    **The client contract** is the same for every director app; only the
    fields of ``enqueue`` differ. Commands: ``enqueue``, ``cancel``,
    ``discard``, ``move``, ``play``, ``stop``, ``set_autoplay``,
    ``set_hold_last_frame``, ``set_build_paused``, ``get_state`` and
    ``reset``. Every change broadcasts ``director_state``, which a client also
    receives when it joins; ``shot_generated``, ``shot_failed``,
    ``shot_started``, ``shot_finished`` and ``shot_stopped`` mark each shot's
    moves.

    Building runs only while a session is live and a client is connected, the
    runtime's own client included, and ``build_paused`` is false. One shot
    builds at a time, and only while the playout queue has room for it. A
    build already running when its shot is cancelled, or its session ends,
    runs on; its result is dropped.

    The class owns ``run()``, the step hooks, and the session's lifecycle
    bookkeeping; a model declares no ``state:`` and writes no ``run()``,
    ``process_input()`` or ``process_output()``. ``@event`` handlers and
    lifecycle hooks of its own still work.

    Class attributes:
        fps: The rate shots play at, in frames per second.
        max_queued: The most shots waiting to build, 20 by default.
        max_ready: The most built shots waiting to play, 10 by default. Each
            is held in memory.
        history: The most built shots kept so later shots can continue them,
            32 by default. A shot a queued shot depends on is kept past it.
        plays: The :class:`Output` a returned :class:`LocalFile` plays on, or
            ``None`` to keep files without playing them. It declares at most
            one video and one audio track.

    Raises:
        ValueError: When a subclass is defined with a limit that is not a
            positive integer.
        TypeError: When a subclass is defined with a ``generate()`` whose
            parameters a client cannot set, or with a ``plays`` that is not an
            :class:`Output` with tracks.
    """

    max_queued: ClassVar[int] = 20
    max_ready: ClassVar[int] = 10
    history: ClassVar[int] = 32
    plays: ClassVar[type[Output] | None] = None

    __director__: ClassVar[DirectorSpec | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        for name in ("max_queued", "max_ready", "history"):
            value = getattr(cls, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{cls.__name__}.{name} must be a positive integer, got {value!r}")
        plays = cls.plays
        if plays is not None and not (
            isinstance(plays, type) and issubclass(plays, Output) and plays.__tracks__
        ):
            raise TypeError(f"{cls.__name__}.plays must be an Output subclass with tracks")
        if cls.generate is not ReactorDirectorApp.generate:
            spec = build_spec(cls, cls.generate)
            cls.__director__ = spec
            _stamp(
                cls,
                "enqueue",
                spec.enqueue,
                ReactorDirectorApp._queue_shot,
                spec.messages.queued,
                "Queue one shot. It builds when its turn comes, then waits in the playout queue "
                "until it plays. Replies with the shot. Fails with `queue_full` when the "
                "generation queue is full, and `unknown_shot` when a shot id it names is not "
                "queued, built, or in the history.",
            )
            _stamp(
                cls,
                "get_state",
                make_command("get_state", []),
                ReactorDirectorApp._get_state,
                spec.messages.state,
                "Reply with `director_state`, everything the session holds.",
            )
        super().__init_subclass__(**kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Build one shot from the inputs a client queued it with.

        Override it with one named parameter per input; the class reads the
        signature when the subclass is defined. Return an :class:`Output` or a
        :class:`LocalFile`, or raise to fail the shot.
        """
        raise NotImplementedError(f"{type(self).__name__} must define generate()")

    def __init__(self) -> None:
        super().__init__()
        spec = type(self).__director__
        if spec is None:
            raise TypeError(f"{type(self).__name__} must define generate()")
        self._spec = spec
        self._generation = ShotQueue(self.max_queued)
        self._playout = ShotQueue(self.max_ready)
        self._history = History(self.history)
        self._playing: ShotEntry | None = None
        self._play_started = False
        self._stop_requested = False
        self._forget_playing = False
        self._autoplay = False
        self._hold_last_frame = False
        self._build_paused = False
        # Bumped at every session start and end; a build or a playout that
        # began under another value belongs to a session that is over.
        self._epoch = 0
        self._build_task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None

    # -- commands -------------------------------------------------------------

    async def _queue_shot(self, **values: Any) -> ModelMessage:
        if self._generation.full:
            raise CommandError(
                "queue_full",
                f"The generation queue is full ({self.max_queued} shots); cancel one or wait.",
            )
        metadata = values.pop("metadata")
        position = values.pop("position")
        sources: dict[str, str] = {}
        for name in self._spec.shot_parameters:
            shot_id = values.get(name)
            if not shot_id:
                values[name] = None
                continue
            if self._generation.get(shot_id) is None and self._built(shot_id) is None:
                raise CommandError(
                    "unknown_shot",
                    f"No shot with id {shot_id!r} is queued, built, or in the history.",
                )
            sources[name] = shot_id
        continues = sources.get(self._spec.continues) if self._spec.continues else None
        entry = ShotEntry(
            id=str(uuid.uuid4()),
            inputs=values,
            sources=sources,
            continues=continues,
            metadata=metadata,
        )
        self._generation.add(entry, position)
        await self._publish()
        return self._shot_message(self._spec.messages.queued, entry)

    async def _get_state(self) -> ModelMessage:
        return self._state_message()

    @event(
        name="cancel",
        description=(
            "Remove a shot that is not built yet. A build already running for it runs on and "
            "its result is dropped. Fails with `unknown_shot`, `shot_built` for a built shot "
            "(discard it instead), and `shot_referenced` while queued shots depend on it."
        ),
    )
    async def cancel(
        self, shot_id: str = InputField(description="The id of the shot to remove.")
    ) -> None:
        """Remove a queued shot."""
        entry = self._generation.get(shot_id)
        if entry is None:
            if self._built(shot_id) is not None:
                raise CommandError("shot_built", "That shot is built; discard it instead.")
            raise CommandError("unknown_shot", f"No queued shot has id {shot_id!r}.")
        if any(other.depends_on({shot_id}) for other in self._generation if other is not entry):
            raise CommandError(
                "shot_referenced", "Queued shots depend on that shot; cancel them first."
            )
        self._generation.remove(entry)
        await self._publish()

    @event(
        name="discard",
        description=(
            "Remove a built shot from the playout queue before it plays. It stays in the "
            "history, so later shots can still continue it. Fails with `unknown_shot`, and "
            "`shot_not_ready` for a shot that is not built yet (cancel it instead)."
        ),
    )
    async def discard(
        self, shot_id: str = InputField(description="The id of the built shot to remove.")
    ) -> None:
        """Remove a shot from the playout queue."""
        entry = self._playout.get(shot_id)
        if entry is None:
            if self._generation.get(shot_id) is not None:
                raise CommandError(
                    "shot_not_ready", "That shot is not built yet; cancel it instead."
                )
            raise CommandError("unknown_shot", f"No shot waiting to play has id {shot_id!r}.")
        self._playout.remove(entry)
        self._retire(entry)
        await self._publish()

    @event(
        name="move",
        description=(
            "Move a shot within the queue that holds it: the generation queue for a shot "
            "not built yet, the playout queue for a built one. 0 is the front; a position "
            "past the back is the back. Fails with `unknown_shot`."
        ),
    )
    async def move(
        self,
        shot_id: str = InputField(description="The id of the shot to move."),
        position: int = InputField(ge=0, description="Where it goes; 0 is the front."),
    ) -> None:
        """Reorder a shot within its queue."""
        for queue in (self._generation, self._playout):
            entry = queue.get(shot_id)
            if entry is not None:
                queue.move(entry, position)
                await self._publish()
                return
        raise CommandError("unknown_shot", f"No queued shot has id {shot_id!r}.")

    @event(
        name="play",
        description=(
            "Play a built shot: the one named, or the front of the playout queue. Fails with "
            "`already_playing`, `no_ready_shot` when nothing is built, `shot_not_ready` for a "
            "shot still building, `unknown_shot`, and `not_streaming` in a stepped session, "
            "which plays nothing."
        ),
    )
    async def play(
        self,
        shot_id: str = InputField(
            default="", description="The id of the shot to play. Empty plays the front one."
        ),
    ) -> None:
        """Start playing a built shot."""
        if self._stepped:
            raise CommandError(
                "not_streaming", "A stepped session plays nothing; each step returns its shot."
            )
        if self._playing is not None:
            raise CommandError("already_playing", "A shot is playing; stop it first.")
        if shot_id:
            entry = self._playout.get(shot_id)
            if entry is None:
                if self._generation.get(shot_id) is not None:
                    raise CommandError("shot_not_ready", "That shot is not built yet.")
                raise CommandError("unknown_shot", f"No shot waiting to play has id {shot_id!r}.")
        else:
            entry = self._playout.head()
            if entry is None:
                raise CommandError("no_ready_shot", "No built shot is waiting to play.")
        self._start_playing(entry)
        await self._publish()

    @event(
        name="stop",
        description=(
            "Cut the shot that is playing. With autoplay on, the next built shot starts, so "
            "this skips. Fails with `not_playing`."
        ),
    )
    async def stop(self) -> None:
        """Cut the shot that is playing."""
        if self._playing is None:
            raise CommandError("not_playing", "No shot is playing.")
        self._stop_requested = True
        self._notify()

    @event(
        name="set_autoplay",
        description="Play built shots in order on their own, without a play per shot.",
    )
    async def set_autoplay(
        self, autoplay: bool = InputField(default=False, description="True plays on its own.")
    ) -> None:
        """Turn autoplay on or off."""
        self._autoplay = autoplay
        await self._publish()

    @event(
        name="set_hold_last_frame",
        description=(
            "Between shots, and when a shot is stopped, hold its last frame instead of "
            "cutting to black."
        ),
    )
    async def set_hold_last_frame(
        self,
        hold_last_frame: bool = InputField(default=False, description="True holds the last frame."),
    ) -> None:
        """Choose between black and the last frame between shots."""
        self._hold_last_frame = hold_last_frame
        await self._publish()

    @event(
        name="set_build_paused",
        description="Stop starting new builds. A build already running finishes.",
    )
    async def set_build_paused(
        self, build_paused: bool = InputField(default=False, description="True pauses building.")
    ) -> None:
        """Pause or resume building."""
        self._build_paused = build_paused
        await self._publish()

    @event(
        name="reset",
        description=(
            "Drop every shot, queued, built and in the history, cut the shot that is playing, "
            "and put every setting back to its default."
        ),
    )
    async def reset(self) -> None:
        """Clear the session back to its defaults."""
        self._drop_every_shot()
        self._autoplay = self._hold_last_frame = self._build_paused = False
        self.output.flush()
        await self._publish()

    # -- the session ----------------------------------------------------------

    def _on_loop_ready(self) -> None:
        super()._on_loop_ready()
        self._streamed_wake = asyncio.Event()
        self._play_wake = asyncio.Event()

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        await super()._dispatch_reactor_event(event)
        if isinstance(event, SessionStarted | SessionEnded):
            self._epoch += 1
            self._drop_every_shot()
            self._autoplay = self._hold_last_frame = self._build_paused = False
        elif isinstance(event, ClientConnected) and self._session_active and not event.system:
            self._reply(event.conn_id, self._state_message(), None)
        self._notify()

    def _drop_every_shot(self) -> None:
        """Drop every shot the class holds and cut the one playing."""
        dropped = self._generation.clear() + self._playout.clear() + self._history.clear()
        if self._build_task is not None and self._spec.is_async:
            self._build_task.cancel()
        playing = self._playing
        if playing is not None and self._play_started:
            # The player cuts it, reports it stopped, and keeps nothing of it.
            self._stop_requested = True
            self._forget_playing = True
        elif playing is not None:
            self._playing = None
            dropped.append(playing)
        for entry in dropped:
            self._release(entry)

    # -- the streamed loop ----------------------------------------------------

    async def _run_streamed(self) -> None:
        """Build shots in the background and play them, until the gate drops."""
        drops = self._gate_drops
        player = asyncio.create_task(self._play_shots(drops), name="director-player")
        player.add_done_callback(lambda _: self._streamed_wake.set())
        try:
            while self._live.is_set() and self._gate_drops == drops and not player.done():
                self._start_build()
                await self._streamed_wake.wait()
                self._streamed_wake.clear()
            if player.done() and not player.cancelled():
                player.result()
        finally:
            if not player.done():
                player.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await player

    def _start_build(self) -> None:
        """Start the next build, when one can start."""
        if self._build_task is not None or self._build_paused:
            return
        if len(self._playout) >= self.max_ready:
            return
        entry = self._generation.next_to_build(self._sources_built)
        if entry is None:
            return
        entry.status = ShotStatus.BUILDING
        self._build_task = asyncio.create_task(
            self._build(entry, self._epoch), name="director-build"
        )

    async def _build(self, entry: ShotEntry, epoch: int) -> None:
        """Build *entry*, put it where it goes, and report it."""
        try:
            await self._publish()
            started = time.monotonic()
            result, error = await self._call_generate(entry)
            elapsed = time.monotonic() - started
            if epoch != self._epoch or entry not in self._generation:
                self._drop_result(result)
                return
            if error is not None and _is_fatal(error):
                self.crash(error)
                return
            reports = await self._land(entry, result, error, elapsed, playable=True)
            for report in reports:
                await self._complete(report)
            if entry.status is ShotStatus.BUILT:
                self._release(entry)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("the director's build failed outside generate()")
            self.crash(error)
        finally:
            if self._build_task is asyncio.current_task():
                self._build_task = None
            self._notify()

    async def _complete(self, report: StepCompleted) -> None:
        """Report a shot's step, or a failed one when its file cannot be kept."""
        try:
            await self.complete_step(report)
        except (OSError, ValueError) as error:
            logger.warning("a shot's file could not be kept with its step", error=str(error))
            await self.complete_step(
                StepCompleted(
                    error=f"the shot's file could not be kept: {error}", elapsed=report.elapsed
                )
            )

    # -- playout --------------------------------------------------------------

    async def _play_shots(self, drops: int) -> None:
        """Play shots while the gate stays open, one at a time."""
        while self._live.is_set() and self._gate_drops == drops:
            entry = self._next_to_play()
            if entry is None:
                await self._play_wake.wait()
                self._play_wake.clear()
                continue
            await self._play(entry, drops)

    def _next_to_play(self) -> ShotEntry | None:
        if self._playing is not None:
            return None if self._play_started else self._playing
        if self._autoplay:
            head = self._playout.head()
            if head is not None:
                self._start_playing(head)
                return head
        return None

    def _start_playing(self, entry: ShotEntry) -> None:
        self._playout.remove(entry)
        entry.status = ShotStatus.PLAYING
        self._playing = entry
        self._notify()

    async def _play(self, entry: ShotEntry, drops: int) -> None:
        """Play *entry*, and every shot autoplay continues from it, then cut or hold."""
        epoch = self._epoch
        clock = _Clock()
        self._play_started = True
        self._stop_requested = False
        messages = self._spec.messages
        try:
            while True:
                await self.send(self._shot_message(messages.started, entry))
                await self._publish()
                output = entry.output
                if output is None:
                    assert entry.file is not None
                    assert self.plays is not None
                    output = await asyncio.to_thread(decode_file, entry.file, self.plays, self.fps)
                    entry.frames = entry.frames or frame_count(output, self.fps)
                ending = await self._emit_shot(output, epoch, drops, clock)
                if epoch != self._epoch:
                    if self._playing is entry:
                        self._playing = None
                    self._release(entry)
                    self.output.flush()
                    return
                after = self._chained_after(entry) if ending == "finished" else None
                self._playing = None
                if self._forget_playing:
                    self._release(entry)
                else:
                    self._retire(entry)
                if ending == "finished":
                    await self.send(self._shot_message(messages.finished, entry))
                elif ending == "stopped":
                    await self.send(self._shot_message(messages.stopped, entry))
                if after is not None:
                    self._start_playing(after)
                    entry = after
                    continue
                if ending == "finished" and not self._hold_last_frame:
                    await asyncio.sleep(_TAIL_SLICES * EMIT_FRAMES / self.fps)
                self.output.flush(hold=self._hold_last_frame and ending != "gone")
                await self._publish()
                return
        except asyncio.CancelledError:
            if self._playing is entry and epoch == self._epoch:
                self._playing = None
                self._retire(entry)
            self.output.flush()
            raise
        finally:
            self._play_started = False
            self._stop_requested = False
            self._forget_playing = False

    def _chained_after(self, finished: ShotEntry) -> ShotEntry | None:
        """The front built shot, when autoplay plays it straight on from *finished*."""
        head = self._playout.head()
        if self._autoplay and head is not None and head.continues == finished.id:
            return head
        return None

    async def _emit_shot(self, output: Output, epoch: int, drops: int, clock: _Clock) -> str:
        """Emit *output* in small slices on a fixed clock.

        The clock never bursts to catch up: when the wire held a slice back, it
        starts again from now. It is carried across chained shots, so a shot
        that continues another keeps the same cadence.

        Returns:
            ``"finished"`` when every frame went out, ``"stopped"`` when ``stop``
            or ``reset`` cut it, ``"gone"`` when the gate dropped or the
            session changed.
        """
        fps = float(self.fps)
        total = frame_count(output, fps)
        loop = asyncio.get_running_loop()
        for lo in range(0, total, EMIT_FRAMES):
            if epoch != self._epoch or not self._live.is_set() or self._gate_drops != drops:
                return "gone"
            if self._stop_requested:
                return "stopped"
            hi = min(lo + EMIT_FRAMES, total)
            now = loop.time()
            if clock.start is None:
                clock.start = now
            position = clock.frames / fps
            clock.start = max(clock.start, now - position)
            delay = clock.start + position - now
            if delay > 0:
                await asyncio.sleep(delay)
            clock.frames += hi - lo
            await self.emit(slice_output(output, lo, hi, fps))
        return "finished"

    # -- the stepped session --------------------------------------------------

    async def _run_requested_step(self, request: StepEnvelope) -> None:
        """Build the front-most shot that can build and answer the request with it."""
        async with self._step_lock:
            session = self._sessions_started
            entry = self._generation.next_to_build(self._sources_built)
            if entry is None:
                head = self._generation.head()
                reason = (
                    "no shot is queued"
                    if head is None
                    else f"shot {head.id} waits for a shot that is not built"
                )
                self._refuse_step(request, session, reason)
                return
            self._steps_accepted += 1
            accepted = RequestedStep(
                request.conn_id,
                request.request_id,
                self._steps_accepted,
                max_message_size=request.max_message_size,
                stream_generation=request.stream_generation,
            )
            entry.status = ShotStatus.BUILDING
            await self._publish()
            started = time.monotonic()
            result, error = await self._call_generate(entry)
            if error is not None and _is_fatal(error):
                raise error
            reports = await self._land(
                entry, result, error, time.monotonic() - started, playable=False
            )
        report, files = await self._take_returned_files(reports[0])
        self._report_step(report, session, request=accepted, files=files)
        self._release(entry)

    # -- building -------------------------------------------------------------

    def _generate_kwargs(self, entry: ShotEntry) -> dict[str, Any]:
        kwargs = dict(entry.inputs)
        for name in self._spec.shot_parameters:
            shot_id = entry.inputs.get(name)
            built = self._built(shot_id) if shot_id else None
            kwargs[name] = built.shot() if built is not None else None
        return kwargs

    async def _call_generate(self, entry: ShotEntry) -> tuple[Any, Exception | None]:
        """Call ``generate()`` for *entry* and return its result or the exception it raised."""
        kwargs = self._generate_kwargs(entry)
        try:
            if self._spec.is_async:
                result = await self.generate(**kwargs)
            else:
                call = functools.partial(self.generate, **kwargs)
                result = await asyncio.get_running_loop().run_in_executor(
                    self._build_executor(), call
                )
        except Exception as error:
            return None, error
        return result, None

    def _build_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="director-build")
        return self._executor

    async def _land(
        self,
        entry: ShotEntry,
        result: Any,
        error: Exception | None,
        elapsed: float,
        *,
        playable: bool,
    ) -> list[StepCompleted]:
        """Put a built or failed shot where it goes, tell clients, and return its steps.

        The first report is *entry*'s; a failed shot is followed by one for
        each queued shot that depended on it.
        """
        self._generation.remove(entry)
        if error is None:
            try:
                await self._take_result(entry, result)
            except TypeError as wrong:
                error = wrong
        if error is not None:
            return await self._fail_shot(entry, f"{type(error).__name__}: {error}", elapsed)
        if playable and self._can_play(entry):
            entry.status = ShotStatus.READY
            self._playout.add(entry)
        else:
            entry.status = ShotStatus.BUILT
            self._history.put(entry)
            self._trim_history()
        await self.send(self._shot_message(self._spec.messages.generated, entry))
        await self._publish()
        return [await self._step_report(entry, elapsed)]

    async def _take_result(self, entry: ShotEntry, result: Any) -> None:
        if isinstance(result, Output):
            entry.output = result
            entry.frames = frame_count(result, float(self.fps))
            entry.last_frame = last_frame(result)
        elif isinstance(result, LocalFile):
            entry.file = result.path
            entry.last_frame, entry.frames = await asyncio.to_thread(read_file_video, result.path)
        else:
            raise TypeError(
                f"generate() returned {type(result).__name__}; return an Output or a LocalFile"
            )

    async def _fail_shot(
        self, entry: ShotEntry, reason: str, elapsed: float
    ) -> list[StepCompleted]:
        entry.status = ShotStatus.FAILED
        entry.error = reason
        failed = [entry]
        for dependent in dependents(self._generation, {entry.id}):
            self._generation.remove(dependent)
            dependent.status = ShotStatus.FAILED
            dependent.error = f"a shot it depends on failed ({entry.id})"
            failed.append(dependent)
        for shot in failed:
            await self.send(self._shot_message(self._spec.messages.failed, shot))
        await self._publish()
        reports = [StepCompleted(error=reason, elapsed=elapsed)]
        reports.extend(StepCompleted(error=shot.error) for shot in failed[1:])
        return reports

    async def _step_report(self, entry: ShotEntry, elapsed: float) -> StepCompleted:
        description = self._describe(entry)
        files: dict[str, bytes | LocalFile] = {"shot.json": json.dumps(description).encode()}
        if entry.last_frame is not None:
            files["last_frame.png"] = await asyncio.to_thread(png_bytes, entry.last_frame)
        output = entry.output
        if entry.file is not None:
            suffix = entry.file.suffix or ".mp4"
            files[f"output{suffix}"] = LocalFile(entry.file)
            output = None
        return StepCompleted(output=output, files=files, data=description, elapsed=elapsed)

    def _drop_result(self, result: Any) -> None:
        """Delete the file of a result nobody waits for any more."""
        if isinstance(result, LocalFile):
            result.path.unlink(missing_ok=True)

    # -- bookkeeping ----------------------------------------------------------

    def _can_play(self, entry: ShotEntry) -> bool:
        return entry.output is not None or (entry.file is not None and self.plays is not None)

    def _built(self, shot_id: str) -> ShotEntry | None:
        """Return the built shot with *shot_id*, wherever it is, or ``None``."""
        if self._playing is not None and self._playing.id == shot_id:
            return self._playing
        return self._playout.get(shot_id) or self._history.get(shot_id)

    def _sources_built(self, entry: ShotEntry) -> bool:
        return all(self._built(shot_id) is not None for shot_id in entry.sources.values())

    def _retire(self, entry: ShotEntry) -> None:
        """Move a shot that will not play again to the history, keeping only what continues it."""
        entry.status = ShotStatus.BUILT
        self._release(entry)
        self._history.put(entry)
        self._trim_history()

    def _release(self, entry: ShotEntry) -> None:
        """Drop a shot's media and delete its file; its last frame and inputs stay."""
        entry.output = None
        if entry.file is not None:
            entry.file.unlink(missing_ok=True)
            entry.file = None

    def _trim_history(self) -> None:
        protected = {shot_id for entry in self._generation for shot_id in entry.sources.values()}
        for evicted in self._history.trim(protected):
            self._release(evicted)

    def _describe(self, entry: ShotEntry) -> dict[str, Any]:
        return describe(entry, float(self.fps))

    def _shot_message(self, message: type[ModelMessage], entry: ShotEntry) -> ModelMessage:
        return _message(message, shot=self._describe(entry))

    def _valid_commands(self) -> list[str]:
        commands = {"get_state", "reset", *_SETTINGS_COMMANDS}
        if not self._generation.full:
            commands.add("enqueue")
        if len(self._generation):
            commands.update({"cancel", "move"})
        if len(self._playout):
            commands.update({"discard", "move"})
        if self._playing is not None:
            commands.add("stop")
        elif len(self._playout) and not self._stepped:
            commands.add("play")
        return sorted(commands)

    def _state_message(self) -> ModelMessage:
        return _message(
            self._spec.messages.state,
            autoplay=self._autoplay,
            hold_last_frame=self._hold_last_frame,
            build_paused=self._build_paused,
            playing=None if self._playing is None else self._describe(self._playing),
            generation=[self._describe(entry) for entry in self._generation],
            playout=[self._describe(entry) for entry in self._playout],
            history=[self._describe(entry) for entry in self._history],
            capacity={
                "generation": self.max_queued,
                "playout": self.max_ready,
                "history": self.history,
            },
            valid_commands=self._valid_commands(),
        )

    async def _publish(self) -> None:
        """Broadcast the session's state and wake the loops to act on it."""
        await self.send(self._state_message())
        self._notify()

    def _notify(self) -> None:
        for name in ("_streamed_wake", "_play_wake"):
            wake = getattr(self, name, None)
            if wake is not None:
                wake.set()


def _message(message: type[ModelMessage], **fields: Any) -> ModelMessage:
    """Build one of the message types the class generated, whose fields only it knows."""
    return cast("Callable[..., ModelMessage]", message)(**fields)


def _stamp(
    cls: type,
    name: str,
    command: Any,
    implementation: Callable[..., Any],
    response: type[ModelMessage],
    description: str,
) -> None:
    """Stamp a command handler that replies with *response* onto *cls*.

    The handler carries the metadata an ``@event`` decorator produces, so the
    contract reads it as an ordinary command. Each class gets its own handler,
    so the reply it publishes is that class's message type.
    """

    async def handler(self: Any, **kwargs: Any) -> Any:
        return await implementation(self, **kwargs)

    handler.__name__ = name
    handler.__qualname__ = f"{cls.__qualname__}.{name}"
    handler.__annotations__ = {"return": response}
    setattr(
        handler,
        EVENT_ATTR,
        EventHandler(
            name=name, description=description, command=command, is_async=True, reserved=()
        ),
    )
    setattr(cls, name, handler)
