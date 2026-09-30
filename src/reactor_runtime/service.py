"""The service — the runtime's lifecycle root.

A tiny in-process supervision tree that owns the running of the process: the sole
arbiter of start, drain, and stop ordering, the one signal owner, and the one
readiness source. Components declare their dependencies; the service brings them
up edge-first — the reverse of dependency order — so the HTTP surface that fronts
the runner is accepting before the model loads and the runtime is observable from
boot. It runs one blocking main loop until shutdown is requested, then drains and
stops in that same edge-first order, so intake closes first and the model thread
(the runner) is the last thing released.

The service also guarantees that a requested shutdown ends the process. The
orderly wind-down awaits component code and, past it, the interpreter joins
every thread still alive — a native peer being released, an encoder, a worker
the model itself started. Any of those can block without bound, and a process
that reports unhealthy yet stays up is restarted only when an external liveness
probe gives up on it, minutes later. A deadline armed at the shutdown request
forces the exit instead, after writing every thread's stack so the hang is
attributable.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import os
import signal
import sys
import threading

from reactor_runtime.core import Health, ServiceComponent
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

# Exit status of a process brought down by an unrecoverable failure — a model
# whose run loop crashed — as opposed to a stop it was asked for or a model
# that refused to load, both of which exit clean.
FAILURE_EXIT_CODE = 1

# How long a forced exit waits for its structured log record to be written
# before leaving anyway. The handlers lock, and the thread that is stuck may be
# holding that lock; the raw stderr line and the stack dump are already out.
_EXIT_RECORD_GRACE = 1.0


def _exit(code: int) -> None:
    """Leave the process now, skipping interpreter cleanup.

    ``os._exit`` runs no ``atexit`` hooks and joins no threads: it is the one
    exit a stuck thread cannot hold up. Wrapped so a test can intercept it.
    """
    os._exit(code)


class Service:
    """The control block supervising the runtime's components.

    Components are hooked on with :meth:`add`; none manages its own place in the
    lifecycle. :meth:`run` starts them edge-first (the reverse of dependency
    order), blocks on a single shutdown event, and on shutdown drains then stops
    them in that same edge-first order — so the model thread is released last.

    A shutdown request arms an exit deadline. The orderly wind-down that
    finishes inside it disarms it; one that does not is cut short by a forced
    exit carrying the shutdown's exit status.
    """

    def __init__(self, *, exit_timeout: float = 10.0, grace_period: float = 30.0) -> None:
        """Start with no components and an unset shutdown signal.

        Args:
            exit_timeout: Seconds the wind-down itself may take before the
                process is forced to exit.
            grace_period: Seconds a draining session is given to end. A stop
                that is asked for (a signal) may still have a session to drain,
                so its deadline is this on top of ``exit_timeout``; a failure
                has no session left and gets ``exit_timeout`` alone.
        """
        self._components: dict[str, ServiceComponent] = {}
        self._shutdown = asyncio.Event()
        self._exit_timeout = exit_timeout
        self._grace_period = grace_period
        self._failure = False
        self._deadline: threading.Timer | None = None
        self._deadline_lock = threading.Lock()

    def add(self, component: ServiceComponent) -> None:
        """Register a component under its name.

        Args:
            component: The component to supervise.

        Raises:
            ValueError: If a component with the same name is already registered.
        """
        if component.name in self._components:
            raise ValueError(f"duplicate component name '{component.name}'")
        self._components[component.name] = component

    @property
    def exit_code(self) -> int:
        """The status the process should exit with once :meth:`run` returns.

        ``0`` for a stop that was asked for or a model that refused to load;
        :data:`FAILURE_EXIT_CODE` once a shutdown was requested for a failure.
        """
        return FAILURE_EXIT_CODE if self._failure else 0

    async def run(self) -> None:
        """Start every component edge-first, block until shutdown, then drain and stop.

        Components come up in reverse dependency order — the HTTP edge before the
        runner it fronts — so the runtime's surface is observable from boot while
        the model is still loading. The one signal handler and the one main loop
        live here. On shutdown — requested by a signal or :meth:`request_shutdown`
        — each started component is drained and then stopped in that same
        edge-first order, so intake closes first and the model thread (the runner)
        is the last thing released. Cleanup runs for whatever started, even if a
        later start fails. A wind-down that completes disarms the exit deadline
        the request armed; the caller then leaves the process with
        :attr:`exit_code`, under a deadline of its own from
        :meth:`bound_process_exit`.
        """
        order = list(reversed(self._topological_order()))
        started: list[ServiceComponent] = []
        try:
            for component in order:
                logger.info("starting component", component=component.name)
                await component.start()
                started.append(component)
            self._install_signal_handlers()
            logger.info("runtime started", components=[component.name for component in started])
            await self._shutdown.wait()
            logger.info("shutdown requested; draining")
        finally:
            # Shutdown is best-effort: a component that fails to drain or stop
            # must not abort the wind-down of the rest, or this supervision tree
            # would leak exactly the resources it exists to release. `started` is
            # in edge-first order, so draining/stopping it forward closes the
            # edge first and releases the core (the runner) last.
            for component in started:
                logger.info("draining component", component=component.name)
                try:
                    await component.drain()
                except Exception:
                    logger.exception("component failed to drain", component=component.name)
            for component in started:
                logger.info("stopping component", component=component.name)
                try:
                    await component.stop()
                except Exception:
                    logger.exception("component failed to stop", component=component.name)
            logger.info("runtime stopped", exit_code=self.exit_code)
            self._disarm_deadline()

    def request_shutdown(self, *, failure: bool = False) -> None:
        """Signal the main loop to begin draining and stopping, and arm the exit deadline.

        Args:
            failure: Whether the process is going down because something broke
                (a crashed model loop) rather than because it was asked to. A
                failure sets the exit status to :data:`FAILURE_EXIT_CODE` and
                gives the wind-down ``exit_timeout`` alone: there is no session
                left to drain. A stop that was asked for keeps the session's
                grace period on top of it.

        The first request arms the deadline; later ones only raise the exit
        status, so a crash that lands during a signal-driven drain is still
        reported as a failure.
        """
        if failure:
            self._failure = True
        self._shutdown.set()
        self._arm_deadline()

    def bound_process_exit(self) -> None:
        """Arm the exit deadline for the process teardown that follows :meth:`run`.

        The wind-down :meth:`run` supervises is only part of leaving: after it,
        the event loop joins its worker threads and the interpreter joins every
        non-daemon thread still alive, and a thread the runtime does not own — a
        native peer being released, a worker the model started — can hold either
        up without bound. The caller that is about to leave the process arms
        this so that teardown is bounded too; a thread that outlives it is
        dumped and the process is forced out with :attr:`exit_code`.
        """
        self._arm_deadline()

    def health(self) -> Health:
        """Aggregate every component's health into one process readiness."""
        return Health.aggregate(component.health() for component in self._components.values())

    def _budget(self) -> float:
        """Seconds the current phase of leaving may take before the exit is forced."""
        if self._failure:
            return self._exit_timeout
        return self._grace_period + self._exit_timeout

    def _arm_deadline(self) -> None:
        """Start the exit deadline; a deadline already running is kept."""
        with self._deadline_lock:
            if self._deadline is not None:
                return
            timer = threading.Timer(self._budget(), self._force_exit)
            timer.daemon = True
            timer.name = "exit-deadline"
            self._deadline = timer
            timer.start()

    def _disarm_deadline(self) -> None:
        """Cancel the exit deadline after a wind-down that completed in time."""
        with self._deadline_lock:
            if self._deadline is not None:
                self._deadline.cancel()
                self._deadline = None

    def _force_exit(self) -> None:
        """Leave the process from the deadline thread, with every stack on record.

        Runs off the event loop, which may itself be the thing that is stuck,
        and nothing on the way to ``_exit`` may wait on a lock another thread
        could hold. The reason and the stacks go straight to the stderr file
        descriptor — ``os.write`` and ``faulthandler`` take no Python-level
        lock — so the record of why and where the process was stuck is written
        whatever else is wedged. The structured log record is written too, for
        the log stream that filters on it, but on a helper thread joined for a
        bounded time: the logging handlers lock, and the wedged thread may be
        the one holding that lock. The exit itself is unconditional.
        """
        code = self.exit_code
        threads = [thread.name for thread in threading.enumerate()]
        reason = f"shutdown deadline exceeded; forcing exit (exit_code={code}, threads={threads})\n"
        with contextlib.suppress(Exception):
            os.write(sys.stderr.fileno(), reason.encode(errors="replace"))
        with contextlib.suppress(Exception):
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

        def record() -> None:
            logger.error(
                "shutdown deadline exceeded; forcing exit", exit_code=code, threads=threads
            )
            for stream in (sys.stdout, sys.stderr):
                stream.flush()

        with contextlib.suppress(Exception):
            recorder = threading.Thread(target=record, name="exit-record", daemon=True)
            recorder.start()
            recorder.join(timeout=_EXIT_RECORD_GRACE)
        _exit(code)

    def _install_signal_handlers(self) -> None:
        """Route SIGTERM and SIGINT to a shutdown request, where supported."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_shutdown)

    def _topological_order(self) -> list[ServiceComponent]:
        """Order components so each starts after the components it depends on.

        Returns:
            The components in dependency order.

        Raises:
            ValueError: If a dependency names an unknown component, or the
                dependencies form a cycle.
        """
        ordered: list[ServiceComponent] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"dependency cycle through '{name}'")
            component = self._components.get(name)
            if component is None:
                raise ValueError(f"unknown component dependency '{name}'")
            visiting.add(name)
            for dependency in component.depends_on:
                visit(dependency)
            visiting.discard(name)
            done.add(name)
            ordered.append(component)

        for name in self._components:
            visit(name)
        return ordered
