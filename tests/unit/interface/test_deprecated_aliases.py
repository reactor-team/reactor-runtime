"""Deprecated names still import and warn.

A renamed class resolves to its new class. A retired surface resolves to itself,
so a model written on it keeps working, and the warning names the replacement.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from reactor_runtime import MediaInput, ReactorApp
from reactor_runtime.interface.pipeline import Idle, ReactorPipeline

_ALIASES = [
    ("reactor_runtime", "ReactorModel", ReactorApp),
    ("reactor_runtime", "Input", MediaInput),
    ("reactor_runtime.interface", "ReactorModel", ReactorApp),
    ("reactor_runtime.interface", "Input", MediaInput),
    ("reactor_runtime.interface.model", "ReactorModel", ReactorApp),
    ("reactor_runtime.interface.model.reactor_model", "ReactorModel", ReactorApp),
    ("reactor_runtime.interface.tracks", "Input", MediaInput),
    ("reactor_runtime.interface.tracks.input", "Input", MediaInput),
]


@pytest.mark.parametrize(("module", "old", "target"), _ALIASES)
def test_old_name_warns_and_is_the_new_class(module: str, old: str, target: type) -> None:
    with pytest.warns(DeprecationWarning, match=f"{old} is now {target.__name__}"):
        resolved = getattr(importlib.import_module(module), old)
    assert resolved is target


@pytest.mark.parametrize(
    "module",
    [
        "reactor_runtime",
        "reactor_runtime.interface",
        "reactor_runtime.interface.model",
        "reactor_runtime.interface.model.reactor_model",
        "reactor_runtime.interface.tracks",
    ],
)
def test_old_names_are_not_in_all(module: str) -> None:
    exported = importlib.import_module(module).__all__
    assert "ReactorModel" not in exported
    assert "Input" not in exported


@pytest.mark.parametrize("module", ["reactor_runtime", "reactor_runtime.interface"])
def test_the_generator_pattern_is_retired_but_still_resolves(module: str) -> None:
    exported = importlib.import_module(module).__all__
    assert "ReactorPipeline" not in exported
    assert "Idle" not in exported
    with pytest.warns(DeprecationWarning, match="ReactorPipeline is deprecated.*ReactorApp"):
        pipeline = importlib.import_module(module).ReactorPipeline
    with pytest.warns(DeprecationWarning, match="Idle is deprecated.*ApplicationError"):
        idle = importlib.import_module(module).Idle
    assert pipeline is ReactorPipeline
    assert idle is Idle


def test_a_pipeline_imported_under_the_deprecated_name_still_runs_as_one() -> None:
    with pytest.warns(DeprecationWarning, match="ReactorPipeline is deprecated"):
        from reactor_runtime import ReactorPipeline as Deprecated

    class Legacy(Deprecated):
        def inference(self):
            yield None

    assert issubclass(Legacy, ReactorPipeline)
    assert issubclass(Legacy, ReactorApp)


def test_the_old_module_path_imports_the_same_class() -> None:
    # The from-import form is what a 3.3.2 model wrote; PEP 562 routes it
    # through the shim module's __getattr__, so it warns like the others.
    with pytest.warns(DeprecationWarning, match="ReactorModel is now ReactorApp"):
        from reactor_runtime.interface.model.reactor_model import ReactorModel
    assert ReactorModel is ReactorApp


def test_the_old_module_path_does_not_serve_the_new_name() -> None:
    # The shim exists for the old name only. The new class is reached through
    # its own module, so the deprecated path never resolves it silently.
    with pytest.raises(ImportError):
        from reactor_runtime.interface.model.reactor_model import ReactorApp  # noqa: F401


def test_unknown_name_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        importlib.import_module("reactor_runtime").NoSuchName  # noqa: B018


def test_a_subclass_of_the_old_name_is_a_reactor_app() -> None:
    with pytest.warns(DeprecationWarning, match="ReactorModel is now ReactorApp"):
        from reactor_runtime import ReactorModel

    class Legacy(ReactorModel):
        async def run(self) -> None: ...

    assert issubclass(Legacy, ReactorApp)


def test_the_first_resolution_of_a_name_is_logged_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reactor_runtime.interface.internal import aliases

    monkeypatch.setattr(aliases, "_reported", set())
    with caplog.at_level(logging.WARNING, logger=aliases.__name__):
        with pytest.warns(DeprecationWarning, match="ReactorModel is now ReactorApp"):
            first = importlib.import_module("reactor_runtime").ReactorModel
        with pytest.warns(DeprecationWarning, match="ReactorModel is now ReactorApp"):
            second = importlib.import_module("reactor_runtime.interface").ReactorModel
    assert first is second is ReactorApp
    records = [
        record for record in caplog.records if "deprecated name imported" in record.getMessage()
    ]
    assert len(records) == 1
