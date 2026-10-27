"""A runtime process stops once a new pod restart token is written."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from reactor_runtime import pod_control


def test_nothing_is_watched_without_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pod_control.POD_CONTROL_DIR_ENV, raising=False)
    assert pod_control.watch_restart() is None


def test_a_new_token_stops_the_process(tmp_path: Path) -> None:
    asked = threading.Event()
    thread = pod_control.watch_restart(tmp_path, poll=0.01, on_request=asked.set)
    assert thread is not None
    assert not asked.wait(0.1)
    (tmp_path / pod_control.TOKEN_FILE).write_text("t1\n")
    assert asked.wait(5)
    thread.join(5)
    assert not thread.is_alive()


def test_the_token_seen_at_start_does_not_stop_it_again(tmp_path: Path) -> None:
    (tmp_path / pod_control.TOKEN_FILE).write_text("t1")
    asked = threading.Event()
    thread = pod_control.watch_restart(tmp_path, poll=0.01, on_request=asked.set)
    assert not asked.wait(0.2)
    (tmp_path / pod_control.TOKEN_FILE).write_text("t2")
    assert asked.wait(5)
    assert thread is not None
    thread.join(5)


def test_an_emptied_token_file_stops_nothing(tmp_path: Path) -> None:
    (tmp_path / pod_control.TOKEN_FILE).write_text("t1")
    asked = threading.Event()
    pod_control.watch_restart(tmp_path, poll=0.01, on_request=asked.set)
    (tmp_path / pod_control.TOKEN_FILE).write_text("")
    assert not asked.wait(0.2)


def test_the_directory_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(pod_control.POD_CONTROL_DIR_ENV, str(tmp_path))
    asked = threading.Event()
    thread = pod_control.watch_restart(poll=0.01, on_request=asked.set)
    assert thread is not None
    (tmp_path / pod_control.TOKEN_FILE).write_text("go")
    assert asked.wait(5)
