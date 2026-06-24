"""Tests for weights-root resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reactor_runtime import get_weights_path as get_weights_path_reexport
from reactor_runtime.paths import (
    DEFAULT_WEIGHTS_PATH,
    ENV_REACTOR_WEIGHTS_PATH,
    get_weights_path,
)


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_REACTOR_WEIGHTS_PATH, "/mnt/weights/model")
    assert get_weights_path() == Path("/mnt/weights/model")


def test_env_tilde_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_REACTOR_WEIGHTS_PATH, "~/custom_weights")
    assert get_weights_path() == Path(os.path.expanduser("~/custom_weights"))


def test_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_REACTOR_WEIGHTS_PATH, raising=False)
    assert get_weights_path() == Path(os.path.expanduser(DEFAULT_WEIGHTS_PATH))


def test_default_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_REACTOR_WEIGHTS_PATH, "")
    assert get_weights_path() == Path(os.path.expanduser(DEFAULT_WEIGHTS_PATH))


def test_exposed_at_package_root() -> None:
    assert get_weights_path_reexport is get_weights_path
