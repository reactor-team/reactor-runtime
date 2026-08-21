"""The default generation loop — :class:`StepDriver`.

Internal machinery, not part of the authoring surface. The default
``ReactorModel.run()`` creates one of these and hands it the loop; an author who
overrides ``run()`` never has one, so nothing here is half-inherited into a
model that owns its own loop.

The driver reaches the application only through the surface an author writes —
``map_step``, ``model.generate``, ``to_output``, ``send``, ``emit``,
``on_step`` — so this file doubles as the authoring contract, spelled out as
the code that consumes it.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.input_buffer import BufferClosed
from reactor_runtime.interface.internal.reactor_core import fps_is_author_pinned
from reactor_runtime.interface.model.stepping import NotReady, StepStats
from reactor_runtime.interface.tracks import Output
from reactor_runtime.log import get_logger

if TYPE_CHECKING:
    from reactor_runtime.interface.model.reactor_model import ReactorModel

logger = get_logger(__name__)

_IDLE_SLEEP = 0.005
"""How long a declined step holds the stream before the loop tries again."""

Product = Output | ModelMessage
"""What one step can put on the wire: media on a track, or a typed message."""


class StepDriver:
    """Drive a model's declarative step layer while a client is connected.

    One step is :meth:`advance`, and :meth:`run` is the loop around it. Neither
    is reachable from a model class: the driver is created inside the default
    ``run()`` and dropped with it.
    """

    def __init__(self, app: ReactorModel) -> None:
        """Bind the application and resolve what the loop needs once.

        Args:
            app: The model whose declarative layer this drives.

        Raises:
            TypeError: If no model is bound to drive.
        """
        if getattr(app, "model", None) is None:
            raise TypeError(
                f"{type(app).__name__} has no model to drive. Either declare "
                f"'model: YourModel' and assign self.model in load(), or override "
                f"run() to own the generation loop."
            )
        self._app = app
        self._step = 0
        # Both resolved once: the pinned-rate decision is a fact about the class,
        # and stats are injected only into a to_output that asks for them.
        self._fps_pinned = fps_is_author_pinned(type(app))
        self._wants_stats = "stats" in inspect.signature(app.to_output).parameters
        # The last state this driver reported. None means nothing to compare
        # against yet, so the first look establishes the baseline and reports
        # nothing: a client that has just connected was told the state already.
        self._state: dict[str, Any] | None = None

    async def run(self) -> None:
        """Step continuously while a session is live and a client is connected.

        Parks between generation cycles, so an idle model does no work. A
        declined step holds the stream briefly and tries again; a closed input
        track ends the cycle and waits for the next one, leaving the input
        buffers empty. The step tally carries across cycles: it counts what the
        runtime has driven, so nothing a session or a model does resets it.
        """
        app = self._app
        while True:
            await app.connected.wait()
            try:
                while app.connected.is_set():
                    try:
                        await self.advance()
                    except NotReady as exc:
                        logger.debug("step declined", reason=str(exc))
                        await asyncio.sleep(_IDLE_SLEEP)
                    else:
                        # A step that emits nothing never awaits the transport,
                        # so yield explicitly to keep commands being dispatched.
                        await asyncio.sleep(0)
            except BufferClosed:
                logger.info("an input track closed; ending this generation cycle")
            finally:
                for buffer in app._input_buffers.values():
                    buffer.reset()
            # A cycle that ends the moment it starts must not spin: the gate is
            # already set, so waiting on it again would not yield.
            await asyncio.sleep(_IDLE_SLEEP)

    async def advance(self) -> None:
        """Run exactly one step: map, generate, place the products on the wire.

        A step belongs to the session that was live when it began. Awaiting the
        transport can outlast that session — a client leaving or a session ending
        clears the gate while a chunk is still going out — so the gate is checked
        again before the application is asked to do anything further. A step the
        session outlived is abandoned: its products went nowhere, so ``on_step``
        does not run and the state is left alone.

        The state is examined twice: once before the step, where it holds
        everything the commands dispatched since the last step have written, and
        once after, in case the step itself wrote to it. Either look reports a
        change to the application.

        Raises:
            NotReady: The mapping declined the step, the model produced nothing,
                or no client is connected to step for.
            BufferClosed: An input track closed, which ends the cycle rather
                than skipping a step.
        """
        app = self._app
        await self._report_state_change()
        if not app.connected.is_set():
            raise NotReady("no client is connected")

        # A model with no declared inbound tracks has no input holder at all.
        inputs = app.map_step(app.state, getattr(app, "input", None))

        started = time.perf_counter()
        produced = app.model.generate(**inputs)
        stats = StepStats(step=self._step, compute_time=time.perf_counter() - started)
        if produced is None:
            raise NotReady("the model produced nothing")
        self._step += 1

        messages, media = _partition(_as_products(self._to_products(produced, stats)))
        # Messages first: an action or a state change must not wait behind the
        # transport's media backpressure.
        for message in messages:
            await app.send(message)
        for output in media:
            if self._fps_pinned:
                await app.emit(output)
            else:
                await app.emit(output, compute_time=stats.compute_time)
        if not app.connected.is_set():
            logger.debug("step abandoned; the session ended while its products went out")
            return
        app.on_step(stats)
        await self._report_state_change()

    async def _report_state_change(self) -> None:
        """Call ``on_state_changed`` when the state differs from the last look.

        Establishing the first baseline reports nothing: a client that has just
        connected was told the state already.
        """
        app = self._app
        state = app.state
        snapshot = None if state is None else dict(vars(state))
        changed = (
            self._state is not None and snapshot is not None and _differs(self._state, snapshot)
        )
        self._state = snapshot
        if changed:
            await app.on_state_changed(state)

    def _to_products(self, produced: Any, stats: StepStats) -> Any:
        """Call ``to_output`` with what the model produced, spread by shape.

        A mapping spreads into keyword arguments, mirroring the way ``map_step``
        spreads into ``generate``; anything else arrives as one positional
        argument. ``stats`` rides along only when the signature declared it.
        """
        extra: dict[str, Any] = {"stats": stats} if self._wants_stats else {}
        if isinstance(produced, dict):
            return self._app.to_output(**produced, **extra)
        return self._app.to_output(produced, **extra)


def _differs(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Return whether two state snapshots hold different values.

    Identity is checked first, so a field holding an array or an opaque object
    that was not reassigned costs nothing to compare. A value that cannot answer
    ``!=`` with a plain bool — an array is the usual case — counts as changed
    rather than raising, because a mapping that keeps scratch state in a field is
    entitled to put anything there.
    """
    if before.keys() != after.keys():
        return True
    for name, value in after.items():
        previous = before[name]
        if previous is value:
            continue
        try:
            if bool(previous != value):
                return True
        except Exception:
            return True
    return False


def _as_products(products: Any) -> tuple[Product, ...]:
    """Normalise what ``to_output`` returned into the products to publish.

    Args:
        products: An :class:`Output`, a :class:`ModelMessage`, a sequence of
            either, or ``None`` for a step that publishes nothing.

    Returns:
        The products, in the order the application returned them.

    Raises:
        TypeError: If the return is not one of those shapes.
    """
    if products is None:
        return ()
    if isinstance(products, (Output, ModelMessage)):
        return (products,)
    if isinstance(products, Sequence) and not isinstance(products, (str, bytes)):
        items = tuple(products)
        for item in items:
            if not isinstance(item, (Output, ModelMessage)):
                raise TypeError(_products_error(item))
        return items
    raise TypeError(_products_error(products))


def _partition(products: tuple[Product, ...]) -> tuple[list[ModelMessage], list[Output]]:
    """Split products into the messages to send and the one output to emit.

    Raises:
        TypeError: If a step produced more than one :class:`Output`. Playout is
            paced per step, so a step that emits several times has to own its
            own loop instead.
    """
    messages = [item for item in products if isinstance(item, ModelMessage)]
    media = [item for item in products if isinstance(item, Output)]
    if len(media) > 1:
        raise TypeError(
            f"to_output() returned {len(media)} outputs, and one step emits at most one. "
            f"Combine them into a single Output with a payload per track, or override run() "
            f"to emit on your own schedule."
        )
    return messages, media


def _products_error(value: Any) -> str:
    """Build the message for a ``to_output`` return the driver cannot publish."""
    return (
        f"to_output() returned {type(value).__name__}; return an Output, a ModelMessage, "
        f"a sequence of either, or None for a step that publishes nothing."
    )
