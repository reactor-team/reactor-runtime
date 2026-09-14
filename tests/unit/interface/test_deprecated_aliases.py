"""The old names of renamed classes still import, warn, and resolve to the new class."""

from __future__ import annotations

import importlib
import logging

import pytest

from reactor_runtime import MediaInput, ReactorApp

_ALIASES = [
    ("reactor_runtime", "ReactorModel", ReactorApp),
    ("reactor_runtime", "Input", MediaInput),
    ("reactor_runtime.interface", "ReactorModel", ReactorApp),
    ("reactor_runtime.interface", "Input", MediaInput),
    ("reactor_runtime.interface.model", "ReactorModel", ReactorApp),
    ("reactor_runtime.interface.tracks", "Input", MediaInput),
]


@pytest.mark.parametrize(("module", "old", "target"), _ALIASES)
def test_old_name_warns_and_is_the_new_class(module: str, old: str, target: type) -> None:
    with pytest.warns(DeprecationWarning, match=f"{old} is now {target.__name__}"):
        resolved = getattr(importlib.import_module(module), old)
    assert resolved is target


@pytest.mark.parametrize("module", ["reactor_runtime", "reactor_runtime.interface"])
def test_old_names_are_not_in_all(module: str) -> None:
    exported = importlib.import_module(module).__all__
    assert "ReactorModel" not in exported
    assert "Input" not in exported


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
