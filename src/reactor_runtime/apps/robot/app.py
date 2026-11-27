"""The base class of a robot policy, :class:`ReactorRobotApp`."""

from __future__ import annotations

from reactor_runtime.interface.app import ReactorApp


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
    """
