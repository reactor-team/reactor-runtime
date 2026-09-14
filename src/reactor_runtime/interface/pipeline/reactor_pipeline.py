"""Generator-driven authoring base — :class:`ReactorPipeline`.

A higher-level model base built on :class:`ReactorApp`. Instead of writing a
manual ``run()`` loop, an author implements an ``inference()`` generator and
declares a typed :class:`InputState`; the base drives the generator across
connection cycles and adapts the emission rate to the model's own pace.

It owns two things on top of :class:`ReactorApp`:

- The ``run()`` driver: gate on a live session with a client present, advance
  the generator one ``yield`` at a time, emit each :class:`Output`, skip a turn
  on :data:`Idle`, and tear the generator down cleanly when the client leaves.
- A generator lock: ``@event`` and lifecycle handlers run only between the
  generator's ``yield`` points, so ``self.state`` is consistent within a single
  inference turn rather than mutating mid-computation.

The typed state and its ``set_<field>`` commands come from :class:`ReactorApp`.
A pipeline must declare ``state: MyState``; the base leaves it optional.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from reactor_runtime.interface.app.reactor_app import ReactorApp, _fps_is_author_pinned
from reactor_runtime.interface.internal.input_buffer import BufferClosed
from reactor_runtime.interface.pipeline.idle import Idle
from reactor_runtime.interface.tracks import Output
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

# A short pause on an idle turn or a generator restart yields the loop so the
# dispatch loops deliver commands, without busy-spinning.
_IDLE_SLEEP = 0.005
_RESTART_SLEEP = 0.005


class _GeneratorEnded(Exception):  # noqa: N818 — a control-flow signal, not an error a caller handles
    """Internal signal that the inference generator finished and should restart.

    ``StopIteration`` cannot cross an ``async def`` boundary (PEP 479 turns it
    into ``RuntimeError``), so a finished generator is reported with this
    instead and the driver restarts it.
    """


class ReactorPipeline(ReactorApp):
    """Generator-driven model with typed, client-mutable state.

    Subclass and provide:

    - ``state: MyState`` — a class annotation naming an :class:`InputState`
      subclass. ``self.state`` holds the live instance during a session.
    - ``inference()`` — a generator (``def`` or ``async def``) that reads
      ``self.state``, optionally consumes input tracks, and yields an
      :class:`Output` per produced frame. Yield :data:`Idle` (or ``None``) to
      skip a turn.

    ``self.state`` is session-scoped: a fresh instance is built when a session
    starts — before the ``@session_started`` hook runs, so once-per-session
    initialization can write to it — and cleared when the session ends, after
    the ``@session_ended`` hook. A client leaving and rejoining within one
    session sees the same state; the next session starts from field defaults.
    Public state fields become ``set_<field>`` commands automatically;
    everything :class:`ReactorApp` offers — ``@event``, the lifecycle hooks,
    ``emit``, ``send`` — still applies.

    Generation runs only while the session is live and a client is connected.
    Ending the session (or the last client leaving) stops the generator at the
    next turn boundary; a fresh session with a connected client starts it again.

    When the subclass declares no ``fps``, the emission rate adapts to the
    measured inference time; declaring ``fps`` pins it to a fixed rate.
    """

    def __init__(self) -> None:
        super().__init__()
        # An abstract intermediate without a state annotation is left alone at
        # class creation; the requirement is enforced at instantiation.
        if type(self).__app_state__ is None:
            raise TypeError(
                f"{type(self).__name__} must declare 'state: MyState' where MyState "
                "is an InputState subclass."
            )

    # -- the driver -----------------------------------------------------------

    async def run(self) -> None:
        """Drive ``inference()`` across session cycles.

        Each cycle gets a new generator and lasts while the session is live and
        a client is connected. When that gate drops — the session ends or the
        last client leaves — the generator is closed, the input buffers reset,
        and the driver waits for the next runnable window. ``self.state`` is
        session-scoped: the reactor loop builds it when the session starts and
        clears it when the session ends, so it survives a client leaving and
        rejoining within one session. The inference callable is resolved from
        ``self.inference`` once, so ``load`` may replace it with an
        instance-bound generator; whether it is async is inferred from that
        callable.

        An uncaught exception in ``inference()`` — including a yield that is not
        an :class:`Output`, :data:`Idle`, or ``None`` — is fatal: it propagates
        out of this driver and permanently ends the model loop, not just the
        current session. The generator is still closed on the way out, and a
        failure in that cleanup is logged and dropped, so the exception the
        runtime reports is the one the model raised first.

        A cleanup that fails after a clean session end has no earlier exception
        to defer to, so it propagates and ends the model loop too. The input
        buffers reset on every path out, including that one.
        """
        lock = self._step_lock
        inference_fn = self.inference
        is_async = inspect.isasyncgenfunction(inference_fn)
        dynamic_fps = not _fps_is_author_pinned(type(self))

        while True:
            await self._live.wait()

            gen = inference_fn()
            ended_cleanly = False
            try:
                while self._live.is_set():
                    try:
                        async with lock:
                            output, compute_time = await self._advance(gen, is_async)
                    except _GeneratorEnded:
                        gen = inference_fn()
                        await asyncio.sleep(_RESTART_SLEEP)
                        continue
                    except BufferClosed:
                        break

                    if output is None:
                        await asyncio.sleep(_IDLE_SLEEP)
                        continue

                    if dynamic_fps:
                        await self.emit(output, compute_time=compute_time)
                    else:
                        await self.emit(output)
                ended_cleanly = True
            finally:
                try:
                    await self._close_generator(gen, is_async, ended_cleanly=ended_cleanly)
                finally:
                    for buffer in self._input_buffers.values():
                        buffer.reset()

    async def _advance(self, gen: Any, is_async: bool) -> tuple[Output | None, float]:
        """Advance the generator by one yield, timing the work it took.

        Returns the yielded :class:`Output` and the seconds spent producing it,
        or ``None`` with a zero time for a skipped turn (an :data:`Idle` or
        ``None`` yield).

        Raises:
            _GeneratorEnded: The generator finished and should be restarted.
            BufferClosed: An input read found its track closed mid-session.
            TypeError: The generator yielded something other than an
                :class:`Output`, :data:`Idle`, or ``None``.
        """
        start = time.perf_counter()
        try:
            result = await gen.__anext__() if is_async else next(gen)
        except (StopIteration, StopAsyncIteration) as end:
            raise _GeneratorEnded from end
        compute_time = time.perf_counter() - start
        if result is None or result is Idle:
            return None, 0.0
        if not isinstance(result, Output):
            raise TypeError(
                f"{type(self).__name__}.inference() yielded "
                f"{type(result).__name__}; yield an Output instance or Idle."
            )
        return result, compute_time

    async def _close_generator(self, gen: Any, is_async: bool, *, ended_cleanly: bool) -> None:
        """Close the generator on teardown, reporting a fatal cleanup failure.

        Closing the generator runs the model's own cleanup block. When the loop
        already carries an exception of its own, a failure here is logged and
        dropped, so the original exception stays the one the runtime reports.
        When the loop ended cleanly, that failure is the only one there is, so
        it propagates and ends the model loop.

        Args:
            gen: The inference generator to close.
            is_async: Whether *gen* is an async generator.
            ended_cleanly: Whether the session loop finished without raising.

        Raises:
            Exception: Whatever the model's cleanup raised, when *ended_cleanly*
                is true.
        """
        try:
            if is_async:
                await gen.aclose()
            else:
                gen.close()
        except Exception:
            if ended_cleanly:
                raise
            logger.exception("inference generator raised while closing")

    # -- author hook ----------------------------------------------------------

    def inference(self) -> Any:
        """Produce frames — override this as a generator.

        Implement as a sync ``def`` or ``async def`` generator. Each turn reads
        ``self.state`` for the current parameters, optionally reads input
        tracks, runs a forward pass, and yields an :class:`Output`. Yield
        :data:`Idle` or ``None`` to skip a turn. The generator starts when a
        client connects and is restarted if it finishes while the client stays.

        A ``finally:`` block in the generator runs when the session ends, so it
        is the place to release what the session held. Raising from it ends the
        model loop permanently — not just the current session — so raise only for
        a fault that the next session cannot survive, and handle anything
        recoverable in place.

        Example::

            def inference(self):
                while not self.state._started:
                    yield Idle
                while True:
                    frame = self.forward(prompt=self.state.prompt)
                    yield MyOutput(main_video=frame)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement inference()")
