"""Session and transport adapter for Reactor Realtime Engine."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.pipeline.reactor_pipeline import ReactorPipeline
from reactor_runtime.interface.tracks import Output


@dataclass(frozen=True)
class _EngineAPI:
    """Names imported lazily from the optional engine package."""

    action: Callable[..., Any]
    attach: Callable[..., Any]
    detach: Callable[..., Any]
    inbox: Callable[..., Any]
    outbox: Callable[..., Any]
    reset: Callable[..., Any]
    set_active: Callable[..., Any]
    run_engine: Callable[..., Coroutine[Any, Any, None]]


class RealtimePipeline(ReactorPipeline):
    """Adapt a queue-driven realtime engine to one runtime session.

    Subclasses declare an :class:`~reactor_runtime.InputState`, implement
    :meth:`build_engine`, and usually override :meth:`on_output` to translate
    the engine's opaque chunks into runtime outputs or messages.

    The adapter owns the boundary rather than model execution: it snapshots
    session state into engine ``Action`` messages, maps session and connection
    lifecycle events onto the engine inbox, and drains generated chunks into
    the runtime transport. An attached session is inactive while no client is
    connected, preserving engine state without spending inference work.

    A runtime process currently hosts one session, so this embedded adapter is
    a B=1 integration. Cross-session batching requires a shared engine host; it
    is not implied by an engine whose manifest advertises multiple sessions.
    """

    _engine_api: _EngineAPI | None
    _engine: Any
    _engine_inbox: Any
    _engine_outbox: Any
    _engine_stop: asyncio.Event | None
    _engine_session_id: str | None
    _engine_session_epoch: int
    _engine_active: bool | None
    _last_conditioning: Mapping[str, Any] | None
    _conditioning_interval_s: float | None

    def __init__(self) -> None:
        super().__init__()
        self._engine_api = None
        self._engine = None
        self._engine_inbox = None
        self._engine_outbox = None
        self._engine_stop = None
        self._engine_session_id = None
        self._engine_session_epoch = 0
        self._engine_active = None
        self._last_conditioning = None
        self._conditioning_interval_s = None

    def load(self, config_path: Path | None) -> None:
        """Build the engine and configure output pacing from its manifest."""
        self._engine_api = _load_engine_api()
        self._engine = self.build_engine(config_path)
        manifest = self._engine.manifest()
        if manifest.rate_hz <= 0:
            raise ValueError(f"engine rate_hz must be positive, got {manifest.rate_hz}")
        if manifest.frames_per_chunk <= 0:
            raise ValueError(
                f"engine frames_per_chunk must be positive, got {manifest.frames_per_chunk}"
            )
        self._conditioning_interval_s = 1.0 / float(manifest.rate_hz)
        self.output.fps = float(manifest.rate_hz * manifest.frames_per_chunk)

    def build_engine(self, config_path: Path | None) -> Any:
        """Load weights and return a ``RealtimeInterface`` implementation.

        Args:
            config_path: Model configuration supplied by ``reactor.yaml``, if
                configured.

        Returns:
            An engine implementing the public realtime-engine contract.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement build_engine()")

    def attach_context(self) -> Mapping[str, Any]:
        """Return the self-contained context used to seed a new engine session.

        The default contains the pipeline's public input-state fields. Override
        this for seeds, initial images, or other model-specific session data.
        Values are passed through without a device-to-host copy and therefore
        must remain stable after this method returns.
        """
        return self._public_state()

    def state_to_conditioning(self) -> Mapping[str, Any]:
        """Return the current client-controlled conditioning snapshot.

        The default contains the pipeline's public input-state fields. Models
        with media or derived control inputs should override this method and
        return the complete action-class conditioning for one engine tick.
        Values are passed through without a device-to-host copy and therefore
        must remain stable after this method returns.
        """
        return self._public_state()

    async def on_output(self, output: Any) -> None:
        """Route one opaque engine output into the runtime.

        Runtime :class:`Output` and :class:`ModelMessage` values work without
        an override. A model returning another chunk type must translate it in
        its own implementation.
        """
        if isinstance(output, Output):
            await self.emit(output)
        elif isinstance(output, ModelMessage):
            await self.send(output)
        else:
            raise TypeError(
                f"{type(self).__name__}.on_output() received {type(output).__name__}; "
                "override on_output() to translate this engine chunk."
            )

    def reset_engine(self, context: Mapping[str, Any] | None = None) -> None:
        """Reset the attached engine session while preserving its active state.

        This method is safe to call from an ``@event`` handler. It also clears
        buffered input and downstream media so pre-reset data cannot leak into
        the new engine epoch.

        Args:
            context: Replacement attach context. The current
                :meth:`attach_context` is used when omitted.
        """
        if self._engine_session_id is None or self._engine_api is None:
            raise RuntimeError("reset_engine() requires an active runtime session")
        snapshot = _snapshot(context if context is not None else self.attach_context())
        self.output.flush()
        self._reset_inputs()
        self._last_conditioning = None
        self._put_engine(self._engine_api.reset(self._engine_session_id, context=snapshot))

    def _on_loop_ready(self) -> None:
        super()._on_loop_ready()
        api = self._engine_api
        if api is None or self._engine is None or self._conditioning_interval_s is None:
            raise RuntimeError("RealtimePipeline.load() must finish before the model loop starts")
        self._engine_inbox = api.inbox()
        self._engine_outbox = api.outbox(maxsize=1)
        self._engine_stop = asyncio.Event()

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        # Stop scheduling before a potentially slow disconnect/session-ended
        # hook, while leaving the attachment intact for that hook to inspect.
        if isinstance(event, SessionEnded) or (
            isinstance(event, ClientDisconnected) and event.total == 0
        ):
            self._set_engine_active(False)

        await super()._dispatch_reactor_event(event)
        api = self._require_engine_api()

        if isinstance(event, SessionStarted):
            self._engine_session_epoch += 1
            self._engine_session_id = f"{event.session_id}:{self._engine_session_epoch}"
            self._engine_active = None
            self._last_conditioning = None
            context = await self._snapshot_attach_context()
            self._put_engine(api.attach(self._engine_session_id, context=context))
            self._sync_engine_active(force=True)
        elif isinstance(event, (ClientConnected, ClientDisconnected)):
            was_active = self._engine_active
            self._sync_engine_active()
            if self._engine_active and not was_active:
                self._last_conditioning = None
            if isinstance(event, ClientDisconnected) and not self._runnable.is_set():
                self._reset_inputs()
        elif isinstance(event, SessionEnded):
            session_id = self._engine_session_id
            if session_id is not None:
                self._put_engine(api.detach(session_id))
            self._engine_session_id = None
            self._engine_active = None
            self._last_conditioning = None
            self._reset_inputs()

    async def run(self) -> None:
        """Drive the engine, conditioning sampler, and output drain together."""
        api = self._require_engine_api()
        if self._engine_inbox is None or self._engine_outbox is None:
            raise RuntimeError("run() started before the model loop was ready")
        stop = self._engine_stop
        if stop is None:
            raise RuntimeError("run() started before the model loop was ready")

        engine_task = asyncio.create_task(
            api.run_engine(
                self._engine,
                self._engine_inbox,
                self._engine_outbox,
                stop=stop,
            ),
            name="realtime-engine",
        )
        conditioning_task = asyncio.create_task(
            self._conditioning_loop(), name="realtime-conditioning"
        )
        output_task = asyncio.create_task(self._output_loop(), name="realtime-output")
        workers = (engine_task, conditioning_task, output_task)

        try:
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            raise RuntimeError("a realtime pipeline worker stopped unexpectedly")
        finally:
            stop.set()
            conditioning_task.cancel()
            output_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conditioning_task
            with contextlib.suppress(asyncio.CancelledError):
                await output_task
            await engine_task

    async def _conditioning_loop(self) -> None:
        interval = self._conditioning_interval_s
        lock = self._gen_lock
        if interval is None or lock is None:
            raise RuntimeError("conditioning loop started before load() completed")

        while True:
            await self._runnable.wait()
            started = asyncio.get_running_loop().time()
            session_id = self._engine_session_id
            if session_id is not None:
                async with lock:
                    conditioning = _snapshot(self.state_to_conditioning())
                if self._last_conditioning is None or not _values_equal(
                    conditioning, self._last_conditioning
                ):
                    api = self._require_engine_api()
                    self._put_engine(api.action(session_id, conditioning=conditioning))
                    self._last_conditioning = conditioning
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _output_loop(self) -> None:
        while True:
            chunk = await self._engine_outbox.get()
            if chunk.session_id != self._engine_session_id:
                continue
            await self.on_output(chunk.output)

    async def _snapshot_attach_context(self) -> Mapping[str, Any]:
        lock = self._gen_lock
        if lock is None:
            raise RuntimeError("session started before the model loop was ready")
        async with lock:
            return _snapshot(self.attach_context())

    def _sync_engine_active(self, *, force: bool = False) -> None:
        self._set_engine_active(self._runnable.is_set(), force=force)

    def _set_engine_active(self, active: bool, *, force: bool = False) -> None:
        session_id = self._engine_session_id
        if session_id is None:
            return
        if force or active != self._engine_active:
            api = self._require_engine_api()
            self._put_engine(api.set_active(session_id, active=active))
            self._engine_active = active

    def _public_state(self) -> Mapping[str, Any]:
        state = self.state
        if state is None:
            return {}
        return {name: getattr(state, name) for name in type(state)._public_fields}

    def _reset_inputs(self) -> None:
        for buffer in self._input_buffers.values():
            buffer.reset()

    def _put_engine(self, message: Any) -> None:
        if self._engine_inbox is None:
            raise RuntimeError("engine inbox is not ready")
        self._engine_inbox.put_nowait(message)

    def _require_engine_api(self) -> _EngineAPI:
        if self._engine_api is None:
            raise RuntimeError("RealtimePipeline.load() has not completed")
        return self._engine_api


def _snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy the message envelope while preserving opaque tensor payloads."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"engine context and conditioning must be mappings, got {type(value).__name__}"
        )
    return dict(value)


def _values_equal(left: Any, right: Any) -> bool:
    """Compare nested conditioning values, including NumPy arrays."""
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and bool(np.array_equal(left, right, equal_nan=True))
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _load_engine_api() -> _EngineAPI:
    """Load the optional engine package only when the adapter is used."""
    try:
        module = importlib.import_module("realtime_engine")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RealtimePipeline requires reactor-realtime-engine>=0.3,<0.4 in the model environment."
        ) from exc

    required = {
        "action": "Action",
        "attach": "Attach",
        "detach": "Detach",
        "inbox": "Inbox",
        "outbox": "Outbox",
        "reset": "Reset",
        "set_active": "SetActive",
        "run_engine": "run_engine",
    }
    missing = [public for public in required.values() if not hasattr(module, public)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "the installed reactor-realtime-engine does not satisfy the >=0.3,<0.4 "
            f"adapter contract; missing: {names}"
        )
    return _EngineAPI(
        **{internal: getattr(module, public) for internal, public in required.items()}
    )
