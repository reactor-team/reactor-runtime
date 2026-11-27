"""The base class of a robot policy, :class:`ReactorRobotApp`."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import Any

from reactor_runtime.apps.robot.loop import PolicyPacer, read_camera
from reactor_runtime.apps.robot.messages import ActionsFailed, ActionsMessage
from reactor_runtime.apps.robot.observe import (
    INVALID_OBSERVE,
    ObserveBuffer,
    ObserveError,
    StampedState,
    parse_observe,
)
from reactor_runtime.apps.robot.sync import InputSynchronizer
from reactor_runtime.apps.robot.types import Actions, Observation
from reactor_runtime.core.fields import InputField
from reactor_runtime.core.model import (
    ClientDisconnected,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import InputFrame, TrackKind
from reactor_runtime.interface.app import ReactorApp
from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.app.reactor_app import _REFUSED_SLEEP, _step_report
from reactor_runtime.interface.app.step_completed import StepCompleted
from reactor_runtime.interface.events.decorators import event
from reactor_runtime.interface.events.errors import ApplicationError, CommandError
from reactor_runtime.interface.internal.reactor_core import STEP_SESSION
from reactor_runtime.interface.tracks import MediaInput, Output
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

STATE_TRACK = "state"
"""The synchronizer track the robot state of each ``observe`` message lands on."""

NO_OBSERVATION = "no observation"
"""The reason the default ``process_input()`` refuses a step when no capture is complete."""


class ReactorRobotApp(ReactorApp):
    """Base class a robot policy subclasses.

    A subclass is a :class:`~reactor_runtime.ReactorApp` and is declared the
    same way: its media inputs, its ``state:`` class, its ``@event`` handlers
    and its lifecycle hooks build the same contract. Every video input is a
    camera. A policy writes ``generate(obs)``, which takes an
    :class:`Observation` and returns its result, and runs one step per
    observation.

    The robot sends its state as the built-in ``observe`` command,
    ``{capture_time_us, state}``: the state of one capture, stamped with the
    capture's time on the robot's clock, the same stamp the capture's camera
    frames carry. The newest
    :data:`~reactor_runtime.apps.robot.observe.OBSERVE_BUFFER_SIZE` states are
    kept in stamp order for the session; they are dropped, with their counts,
    when a session starts or ends, and the states alone are dropped when the
    last client leaves. A malformed message is refused with the
    :data:`~reactor_runtime.apps.robot.observe.INVALID_OBSERVE` code and a
    reason the client can read, and counted. A subclass that declares its own
    ``observe`` command, or a camera named ``state``, fails when the class is
    defined.

    In a streamed session, each camera has a reader task that takes every
    frame off its track as it arrives, and every ``observe`` state goes with
    them into an :class:`~reactor_runtime.apps.robot.sync.InputSynchronizer`,
    one track per camera plus one for the state, joined by exact capture
    stamp. A frame that carries no capture stamp cannot be joined and is
    dropped and counted. The policy loop then runs one step at a time:
    ``process_input()``, ``generate()`` and ``process_output()``, under the
    step lock as on a ``ReactorApp``, and the output and report of each step
    as on a ``ReactorApp``. ``generate()`` runs on a worker thread, so the
    readers keep taking frames meanwhile; command handlers and lifecycle
    hooks still wait for the step to end.

    The default ``process_input()`` takes the newest complete capture not
    used yet and builds its :class:`Observation`; with none, it refuses the
    step with :data:`NO_OBSERVATION` and the loop asks again a few
    milliseconds later, without calling the model. A capture is used at most
    once and never after a newer one: a model slower than the robot skips the
    captures that completed while it ran instead of queueing them, and the
    synchronizer counts them.

    Class attributes:
        fps: The rate the policy loop runs at, in steps per second, when the
            class declares it: steps start on a fixed grid of ``1 / fps``
            seconds and a step that overruns its slot skips the slots it
            missed. Left out, steps run back to back, each as soon as an
            observation is ready. As on a ``ReactorApp``, a pinned ``fps``
            is also the rate emitted media plays out at.

    After ``process_output()`` returns, a step whose ``generate()`` returned
    :class:`Actions` sends an ``actions`` message
    (:class:`~reactor_runtime.apps.robot.messages.ActionsMessage`) to every
    connection, stamped with the step and the ``capture_time_us`` of its
    observation, before the step's media and report. This delivery is the
    class's own: ``process_output()`` cannot replace or suppress it, only add
    to it. Actions whose ``extra`` cannot be written as JSON are not sent; the
    step is answered with ``actions_failed`` instead and reported with the
    error. Every step that ran is reported with :meth:`complete_step`, as on a
    ``ReactorApp``.

    When ``generate()`` raises, the default ``process_output()`` sends
    ``actions_failed``
    (:class:`~reactor_runtime.apps.robot.messages.ActionsFailed`) with the
    step, the capture stamp and the error, reports the step with that error,
    and the loop goes on with the next observation.

    A stepped session runs the three hooks once per request, as on a
    ``ReactorApp``. Its frames do not go through the synchronizer, so the
    default ``process_input()`` refuses its steps with :data:`NO_OBSERVATION`;
    and as its steps have no capture, nothing is delivered and an error from
    ``generate()`` is re-raised, as on a ``ReactorApp``.
    """

    _observed: ObserveBuffer
    _sync: InputSynchronizer
    _cameras: tuple[str, ...]
    _unstamped: int
    _policy_steps: int
    _capture: _StepCapture | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if STATE_TRACK in _camera_names(cls):
            raise TypeError(
                f"{cls.__qualname__} declares a camera named {STATE_TRACK!r}, the name the "
                "robot state takes in each observation; rename the camera"
            )

    def _on_loop_ready(self) -> None:
        """Create the loop-bound state: the observe buffer and the synchronizer."""
        super()._on_loop_ready()
        self._observed = ObserveBuffer()
        self._cameras = _camera_names(type(self))
        self._sync = InputSynchronizer(tracks=self._cameras, data_tracks=(STATE_TRACK,))
        self._unstamped = 0
        self._policy_steps = 0
        self._capture = None

    async def process_input(self) -> Any:
        """Return the observation of the newest complete capture not used yet.

        The observation holds each camera's frame, as an ``(H, W, 3)`` ``uint8``
        array, and the state of the capture's ``observe`` message. Its
        ``step`` is the number of steps the session ran before it, and its
        ``skew_us`` is zero for every camera, as the join is exact. Override
        to shape the observation; call ``super()`` to take the capture.

        Returns:
            The :class:`Observation`. An override returns whatever its
            ``generate()`` takes.

        Raises:
            ApplicationError: :data:`NO_OBSERVATION`, when no capture newer
                than the last one used has every camera's frame and its state.
                The step is refused and nothing is taken.
        """
        taken = self._sync.take()
        if taken is None:
            raise ApplicationError(NO_OBSERVATION)
        frames: dict[str, _Received] = dict(taken.frames)
        state: _Received = taken.data[STATE_TRACK]
        self._capture = _StepCapture(
            step=self._policy_steps,
            capture_time_us=taken.capture_time_us,
            received_at=min(part.at for part in (state, *frames.values())),
        )
        return Observation(
            step=self._policy_steps,
            capture_time_us=taken.capture_time_us,
            frames={name: part.item.data for name, part in frames.items()},
            state=state.item,
            skew_us=dict.fromkeys(frames, 0),
        )

    async def process_output(self, outcome: StepOutcome, /) -> Output | StepCompleted | None:
        """Turn what ``generate()`` did into what the client receives.

        :class:`Actions` emit no media; the class sends them as the
        ``actions`` message once this method returns, whatever it returns.
        When ``generate()`` raised, the error is logged, ``actions_failed``
        is sent to every connection with the step, the capture stamp and the
        error as ``"<type>: <message>"``, and the step is reported with that
        error; the policy loop goes on with the next observation. Any other
        result is read as on a ``ReactorApp``.

        Override to add effects or recover differently; an override that
        handles ``outcome.error`` itself sends its own messages, and one that
        re-raises ends the model loop, as on a ``ReactorApp``.

        Returns:
            ``None`` for :class:`Actions`, a :class:`StepCompleted` holding the
            error for a failed ``generate()``, otherwise what
            :meth:`ReactorApp.process_output` returns.

        Raises:
            Exception: The error ``generate()`` raised, when the step has no
                capture to answer, as in a stepped session.
        """
        if outcome.error is None and isinstance(outcome.result, Actions):
            return None
        capture = self._capture
        if outcome.error is not None and capture is not None:
            reason = f"{type(outcome.error).__name__}: {outcome.error}"
            logger.warning(
                "generate failed",
                step=capture.step,
                capture_time_us=capture.capture_time_us,
                exc_info=outcome.error,
            )
            await self.send(
                ActionsFailed(
                    step=capture.step,
                    capture_time_us=capture.capture_time_us,
                    reason=reason,
                )
            )
            return StepCompleted(error=reason)
        return await super().process_output(outcome)

    @event(
        name="observe",
        description=(
            "Send the robot state of one capture, stamped with the capture's time on "
            "the robot's monotonic clock."
        ),
    )
    def _observe(
        self,
        capture_time_us: Any = InputField(
            default=None,
            description=(
                "The capture stamp, an integer number of microseconds on the robot's "
                "monotonic clock, the same stamp the capture's camera frames carry."
            ),
        ),
        state: Any = InputField(
            default=None,
            description="The robot state at the capture, as a JSON object.",
        ),
    ) -> None:
        """Keep the state of one capture, or refuse a malformed message.

        Raises:
            CommandError: ``invalid_observe``, when the stamp is missing, not an
                integer, or negative, or the state is missing or not a JSON
                object. The message is counted and nothing is kept.
        """
        try:
            stamp, robot_state = parse_observe(capture_time_us, state)
        except ObserveError as refused:
            self._observed.refuse()
            logger.debug("observe refused", reason=str(refused))
            raise CommandError(INVALID_OBSERVE, str(refused)) from None
        entry = StampedState(stamp, robot_state, time.monotonic())
        self._observed.push(entry)
        self._push_state(entry)

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Run the lifecycle as a ``ReactorApp`` does, then drop the inputs it ends."""
        await super()._dispatch_reactor_event(event)
        if isinstance(event, SessionStarted | SessionEnded):
            self._observed.reset()
            self._sync.reset()
            if isinstance(event, SessionStarted):
                self._policy_steps = 0
        elif isinstance(event, ClientDisconnected) and event.total == 0:
            self._observed.clear()
            self._sync.reset()

    async def _run_streamed(self) -> None:
        """Feed the synchronizer from a reader per camera and run the policy loop.

        Starts from the states the session already holds and no frame, and
        drops everything the synchronizer holds once the gate drops.

        Raises:
            Exception: Whatever :meth:`process_output` raised, which by default
                is the error :meth:`generate` raised, or a camera reader's
                failure. It ends the model loop, as on a ``ReactorApp``.
        """
        drops = self._gate_drops
        self._sync.reset()
        for entry in self._observed.entries():
            self._push_state(entry)
        readers = [
            asyncio.create_task(read_camera(name, self._input_buffers[name], self._take_frame))
            for name in self._cameras
        ]
        pacer = PolicyPacer(float(self.fps) if self._fps_pinned else None)
        try:
            while self._live.is_set() and self._gate_drops == drops:
                _raise_failed(readers)
                await pacer.wait()
                if not (self._live.is_set() and self._gate_drops == drops):
                    break
                if not await self._policy_step(pacer):
                    await asyncio.sleep(_REFUSED_SLEEP)
                else:
                    await asyncio.sleep(0)
        finally:
            for reader in readers:
                reader.cancel()
            if readers:
                await asyncio.wait(readers)
            self._sync.reset()
        _raise_failed(readers)

    async def _policy_step(self, pacer: PolicyPacer) -> bool:
        """Run one step of the policy loop; return whether it ran or was refused."""
        async with self._step_lock:
            session = self._sessions_started
            input_started = time.perf_counter()
            self._capture = None
            try:
                try:
                    observation = await self.process_input()
                except ApplicationError as refused:
                    self._report_input_waits(session, report=False)
                    reason = f"{type(refused).__name__}: {refused}"
                    if reason != self._last_refusal:
                        logger.debug(
                            "step refused", reason=str(refused), kind=type(refused).__name__
                        )
                        self._last_refusal = reason
                    return False
                pacer.started()
                self._last_refusal = None
                if isinstance(observation, Observation) and self._capture is None:
                    self._capture = _StepCapture(
                        observation.step, observation.capture_time_us, time.monotonic()
                    )
                self._policy_steps += 1
                input_ms = (time.perf_counter() - input_started) * 1000.0
                self._report_input_waits(session)
                outcome: StepOutcome = await asyncio.to_thread(self._generate, observation)
                output_started = time.perf_counter()
                returned = await self.process_output(outcome)
                stages = self._step_stages(
                    input_ms, outcome.elapsed, (time.perf_counter() - output_started) * 1000.0
                )
                undelivered = await self._deliver_actions(outcome)
            finally:
                self._capture = None

        report = _step_report(returned, outcome)
        if undelivered is not None:
            report = replace(report, error=undelivered)
        rate = None
        if report.output is not None:
            pace = None if self._fps_pinned else outcome.elapsed
            token = STEP_SESSION.set(session)
            try:
                await self.emit(report.output, compute_time=pace)
            finally:
                STEP_SESSION.reset(token)
            rate = self._playout_rate(self._to_bundle(report.output).frame_count, pace)
        report, files = await self._take_returned_files(report)
        self._report_step(report, session, rate, stages, files=files)
        return True

    async def _deliver_actions(self, outcome: StepOutcome) -> str | None:
        """Send the step's actions to every connection; return why they were not sent, if not.

        Sends nothing for a step whose ``generate()`` did not return
        :class:`Actions`, or whose capture is unknown.
        """
        actions = outcome.result
        capture = self._capture
        if outcome.error is not None or not isinstance(actions, Actions) or capture is None:
            if isinstance(actions, Actions) and capture is None:
                logger.warning("actions not sent: the step's input is not an Observation")
            return None
        extra = dict(actions.extra)
        try:
            json.dumps(extra)
        except (TypeError, ValueError) as error:
            reason = f"Actions.extra is not JSON: {error}"
            logger.warning("actions not sent", step=capture.step, reason=reason)
            await self.send(
                ActionsFailed(
                    step=capture.step, capture_time_us=capture.capture_time_us, reason=reason
                )
            )
            return reason
        await self.send(
            ActionsMessage(
                step=capture.step,
                capture_time_us=capture.capture_time_us,
                chunk=actions.chunk.tolist(),
                dt_ms=actions.dt_ms,
                names=None if actions.names is None else list(actions.names),
                extra=extra,
                generate_ms=outcome.elapsed * 1000.0,
                observation_age_ms=(time.monotonic() - capture.received_at) * 1000.0,
            )
        )
        return None

    def _push_state(self, entry: StampedState) -> None:
        """Hand one stamped robot state to the synchronizer, with when it arrived."""
        self._sync.push(
            STATE_TRACK, entry.capture_time_us, _Received(entry.state, entry.received_at)
        )

    def _take_frame(self, camera: str, frame: InputFrame) -> None:
        """Hand one camera frame to the synchronizer, or count it when it has no stamp."""
        if frame.capture_time_us is None:
            self._unstamped += 1
            return
        self._sync.push(camera, frame.capture_time_us, _Received(frame, time.monotonic()))


@dataclass(frozen=True, slots=True)
class _Received:
    """One frame or state in the synchronizer, with when the runtime took it (monotonic seconds)."""

    item: Any
    at: float


@dataclass(frozen=True, slots=True)
class _StepCapture:
    """The capture the running step observed: its step, its stamp and when it first arrived."""

    step: int
    capture_time_us: int
    received_at: float


def _camera_names(cls: type[ReactorApp]) -> tuple[str, ...]:
    """Return the video inputs *cls* declares, in declaration order."""
    holder = cls._find_holder(MediaInput)
    if holder is None:
        return ()
    _, inputs = holder
    return tuple(name for name, info in inputs.__tracks__.items() if info.kind is TrackKind.VIDEO)


def _raise_failed(readers: list[asyncio.Task[None]]) -> None:
    """Raise the error a camera reader ended with, if one did."""
    for reader in readers:
        if reader.done() and not reader.cancelled():
            error = reader.exception()
            if error is not None:
                raise error
