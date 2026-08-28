"""Per-test isolation for the runtime's process-global state.

The interface layer auto-registers every declared ``Output`` / ``Input`` /
``ModelMessage`` / ``@event`` command into a process-global registry, and the
model schema is the union of those registries. That is correct for production —
one model per process — but in a test suite the classes one module declares at
import time would otherwise leak into another module's rendered schema. The
autouse fixture below clears the four registries before each test and restores
them after, so every test sees only the classes it declares within its own
scope.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator

import pytest

from reactor_runtime import log
from reactor_runtime.interface.events.decorators import EVENT_REGISTRY
from reactor_runtime.interface.events.messages import MESSAGE_REGISTRY, ModelMessage
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.tracks.input import INPUT_REGISTRY, Input
from reactor_runtime.interface.tracks.output import OUTPUT_REGISTRY, Output


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Save and restore root handlers and level across each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _clear_log_context() -> Iterator[None]:
    """Release the stamped session id and runtime state after each test.

    Both are process-global, so a test that opens a session or builds a runner
    would otherwise leave every later test's records claiming its context.
    """
    try:
        yield
    finally:
        log.clear_session_id()
        log.set_runtime_state(None)


@pytest.fixture(autouse=True)
def isolate_interface_registries() -> Iterator[None]:
    """Clear the interface registries for the test and restore them after."""
    registries: tuple[dict, ...] = (
        OUTPUT_REGISTRY,
        INPUT_REGISTRY,
        MESSAGE_REGISTRY,
        EVENT_REGISTRY,
    )
    saved = [dict(registry) for registry in registries]
    for registry in registries:
        registry.clear()
    try:
        yield
    finally:
        for registry, snapshot in zip(registries, saved, strict=True):
            registry.clear()
            registry.update(snapshot)


def _register(*classes: type) -> None:
    """Re-register track or message classes into the cleared registries."""
    for cls in classes:
        if issubclass(cls, Output) and cls is not Output and cls.__tracks__:
            OUTPUT_REGISTRY[cls.__name__] = cls
        elif issubclass(cls, Input) and cls is not Input and cls.__tracks__:
            INPUT_REGISTRY[cls.__name__] = cls
        elif issubclass(cls, ModelMessage) and cls is not ModelMessage:
            MESSAGE_REGISTRY[cls.name] = cls


def _register_model(model_cls: type) -> None:
    """Re-register a model's full client-facing surface into the registries.

    Track classes register when they are *defined*, so the replay walks the
    model's module for every module-level ``Output`` / ``Input`` subclass —
    the registrations that module's import made before the per-test clear.
    Track classes defined inside test functions stay invisible, preserving
    per-test isolation.
    """
    module = sys.modules.get(model_cls.__module__)
    if module is not None:
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, (Output, Input)):
                _register(obj)
    for name, spec in ModelContract.of(model_cls).commands.items():
        EVENT_REGISTRY[name] = spec.command
        if spec.response is not None:
            _register(spec.response)


@pytest.fixture
def register() -> Callable[..., None]:
    """Return a helper that re-registers track/message classes after the clear.

    Modules that declare their interface classes at import time call this from a
    function-scoped fixture so the classes land in the registry after
    :func:`isolate_interface_registries` has cleared it.
    """
    return _register


@pytest.fixture
def register_model() -> Callable[[type], None]:
    """Return a helper that re-registers a model's full surface after the clear.

    Restores the track classes the model's module declares, every command its
    handlers declare, and the message types those commands reply with — the
    registrations a class declaration makes at import, replayed after a
    per-test clear.
    """
    return _register_model
