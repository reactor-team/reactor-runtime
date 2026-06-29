"""Per-test isolation for the process-global interface registries.

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

from collections.abc import Callable, Iterator
from typing import get_type_hints

import pytest

from reactor_runtime.interface.events.decorators import EVENT_REGISTRY
from reactor_runtime.interface.events.messages import MESSAGE_REGISTRY, ModelMessage
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.tracks.input import INPUT_REGISTRY, Input
from reactor_runtime.interface.tracks.output import OUTPUT_REGISTRY, Output


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
        elif issubclass(cls, ModelMessage) and cls is not ModelMessage and cls.__message_fields__:
            MESSAGE_REGISTRY[cls.name] = cls


def _register_model(model_cls: type) -> None:
    """Re-register a model's full client-facing surface into the registries."""
    for hint in get_type_hints(model_cls).values():
        if isinstance(hint, type) and issubclass(hint, (Output, Input)):
            _register(hint)
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

    Restores the track holders the model annotates, every command its handlers
    declare, and the message types those commands reply with — the registrations
    a class declaration makes at import, replayed after a per-test clear.
    """
    return _register_model
