"""The base class of a robot policy, :class:`ReactorRobotApp`."""

from __future__ import annotations

import time
from typing import Any

from reactor_runtime.apps.robot.observe import (
    INVALID_OBSERVE,
    ObserveBuffer,
    ObserveError,
    StampedState,
    parse_observe,
)
from reactor_runtime.core.fields import InputField
from reactor_runtime.core.model import (
    ClientDisconnected,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.interface.app import ReactorApp
from reactor_runtime.interface.events.decorators import event
from reactor_runtime.interface.events.errors import CommandError
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


class ReactorRobotApp(ReactorApp):
    """Base class a robot policy subclasses.

    A subclass is a :class:`~reactor_runtime.ReactorApp` and is declared the
    same way: its media inputs, its ``state:`` class, its ``@event`` handlers
    and its lifecycle hooks build the same contract, and the default step loop
    drives ``process_input()``, ``generate()`` and ``process_output()`` exactly
    as it does for a ``ReactorApp``, in streamed and stepped sessions alike.

    A policy's ``generate()`` takes an :class:`Observation` and returns
    :class:`Actions`. ``process_input()`` builds that observation and, as on a
    ``ReactorApp``, is the subclass's to write; what it returns is what
    ``generate()`` gets.

    Every subclass also takes the robot's state as the built-in ``observe``
    command, ``{capture_time_us, state}``: the state of one capture, stamped
    with the capture's time on the robot's clock. The newest
    :data:`~reactor_runtime.apps.robot.observe.OBSERVE_BUFFER_SIZE` states are
    kept in stamp order for the session; they are dropped, with their counts,
    when a session starts or ends, and the states alone are dropped when the
    last client leaves. A malformed message is refused with the
    :data:`~reactor_runtime.apps.robot.observe.INVALID_OBSERVE` code and a
    reason the client can read, and counted. A subclass that declares its own
    ``observe`` command fails when the class is defined.
    """

    _observed: ObserveBuffer

    def _on_loop_ready(self) -> None:
        """Create the loop-bound state, the session's observe buffer included."""
        super()._on_loop_ready()
        self._observed = ObserveBuffer()

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
        self._observed.push(StampedState(stamp, robot_state, time.monotonic()))

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Run the lifecycle as a ``ReactorApp`` does, then drop the states it ends."""
        await super()._dispatch_reactor_event(event)
        if isinstance(event, SessionStarted | SessionEnded):
            self._observed.reset()
        elif isinstance(event, ClientDisconnected) and event.total == 0:
            self._observed.clear()
