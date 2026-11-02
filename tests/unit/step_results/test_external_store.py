"""Persist external sources before announcing a complete public result."""

import json
from pathlib import Path
from threading import Event

import pytest

from reactor_runtime import ExternalFile
from reactor_runtime.core import CompletedStep, StepResultsConfig
from reactor_runtime.step_results import SavedStep, StepStore
from reactor_runtime.step_results import store as store_module

SESSION = "7d9f5c1e-0000-4000-8000-000000000042"


def test_external_source_is_private_persisted_and_readable_after_writer_stops(
    tmp_path: Path,
) -> None:
    ready = Event()
    saved_steps: list[SavedStep] = []
    source = ExternalFile("output.mp4", "video/mp4", "example", "job-1", "https://media.example/v")

    def saved(step: SavedStep) -> None:
        assert (tmp_path / SESSION / "1" / "_external_results.json").is_file()
        saved_steps.append(step)
        ready.set()

    store = StepStore(StepResultsConfig(enabled=True), saved, root=tmp_path)
    try:
        assert store.admit(SESSION, 1, CompletedStep(bundle=None, external_files=(source,)))
        assert ready.wait(5)
    finally:
        store.close()
    assert saved_steps[0].files == ("_external_results.json",)
    assert saved_steps[0].usage == ()
    public_path = store.result_path(SESSION, 1)
    assert public_path is not None
    public = json.loads(public_path.read_text())
    assert public["files"] == []
    assert public["external_files"] == [source.public_entry()]
    assert "provider" not in public_path.read_text()
    assert "job-1" not in public_path.read_text()
    assert "media.example" not in public_path.read_text()
    private = store.file_path(SESSION, 1, "_external_results.json")
    assert private is not None
    assert json.loads(private[0].read_text()) == {"version": 1, "files": [source.source_entry()]}
    assert store.file_path(SESSION, 1, "output.mp4") is None


def test_failed_step_discards_external_sources(tmp_path: Path) -> None:
    ready = Event()
    store = StepStore(StepResultsConfig(enabled=True), lambda _: ready.set(), root=tmp_path)
    source = ExternalFile("output.mp4", "video/mp4", "example", "job", "https://media.example/v")
    try:
        assert store.admit(
            SESSION, 1, CompletedStep(bundle=None, error="failed", external_files=(source,))
        )
        assert ready.wait(5)
    finally:
        store.close()
    public_path = store.result_path(SESSION, 1)
    assert public_path is not None
    assert "external_files" not in json.loads(public_path.read_text())
    assert store.file_path(SESSION, 1, "_external_results.json") is None


def test_private_source_write_failure_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = Event()
    write = store_module._write_atomically

    def fail_descriptor(path: Path, data: bytes) -> None:
        if path.name == "_external_results.json":
            raise OSError("disk full")
        write(path, data)

    monkeypatch.setattr(store_module, "_write_atomically", fail_descriptor)
    store = StepStore(StepResultsConfig(enabled=True), lambda _: ready.set(), root=tmp_path)
    source = ExternalFile("output.mp4", "video/mp4", "example", "job", "https://media.example/v")
    try:
        assert store.admit(SESSION, 1, CompletedStep(bundle=None, external_files=(source,)))
        assert ready.wait(5)
    finally:
        store.close()
    public_path = store.result_path(SESSION, 1)
    assert public_path is not None
    public = json.loads(public_path.read_text())
    assert public["save_error"] == "OSError: disk full"
    assert "external_files" not in public
    assert store.file_path(SESSION, 1, "_external_results.json") is None
