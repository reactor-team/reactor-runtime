"""Local file handoff owns bytes before the model can discard its source."""

import json
import os
from pathlib import Path

import pytest

from reactor_runtime import LocalFile, StepCompleted
from reactor_runtime.core import CompletedStep, StepResultsConfig
from reactor_runtime.core.local_file import OwnedLocalFile, adopt_files, discard_files
from reactor_runtime.interface.internal.reactor_core import ReactorCore
from reactor_runtime.step_results import StepStore

SESSION = "7d9f5c1e-0000-4000-8000-000000000042"


def _app(reports: list[CompletedStep]) -> ReactorCore:
    app = ReactorCore()
    app.bind_output(
        broadcast=lambda _: None,
        addressed=lambda *_: None,
        media=lambda _: None,
        step=reports.append,
    )
    return app


async def test_handoff_survives_source_cleanup_and_stores_original_bytes(tmp_path: Path) -> None:
    original = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 1024
    path = tmp_path / "provider.mp4"
    path.write_bytes(original)
    reports: list[CompletedStep] = []
    await _app(reports).complete_step(
        StepCompleted(files={"output.mp4": LocalFile(path), "usage.json": b"{}"})
    )
    path.unlink()
    owned = reports[0].files["output.mp4"]
    assert isinstance(owned, OwnedLocalFile)
    staging = owned.path.parent
    assert owned.path.read_bytes() == original
    saved = []
    store = StepStore(StepResultsConfig(enabled=True), saved.append, root=tmp_path / "results")
    assert store.admit(SESSION, 1, reports[0])
    store.close()
    result_dir = tmp_path / "results" / SESSION / "1"
    assert (result_dir / "output.mp4").read_bytes() == original
    result = json.loads((result_dir / "result.json").read_text())
    assert result["error"] is None
    assert result["save_error"] is None
    assert result["usage"] == []
    assert result["files"][0] == {
        "name": "output.mp4",
        "content_type": "video/mp4",
        "size": len(original),
    }
    assert saved[0].files == ("output.mp4", "usage.json")
    assert not staging.exists()


async def test_modifying_original_after_handoff_does_not_change_saved_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"original")
    reports: list[CompletedStep] = []
    await _app(reports).complete_step(StepCompleted(files={"audio.wav": LocalFile(path)}))
    path.write_bytes(b"changed")
    owned = reports[0].files["audio.wav"]
    assert isinstance(owned, OwnedLocalFile)
    assert owned.path.read_bytes() == b"original"
    discard_files(reports[0].files)


async def test_failed_report_does_not_open_files(tmp_path: Path) -> None:
    reports: list[CompletedStep] = []
    await _app(reports).complete_step(
        StepCompleted(files={"output.mp4": LocalFile(tmp_path / "missing")}, error="failed")
    )
    assert reports[0].files == {}


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "fifo"])
async def test_handoff_refuses_nonregular_sources(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "file"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(tmp_path)
    elif kind == "fifo":
        os.mkfifo(path)
    reports: list[CompletedStep] = []
    with pytest.raises((OSError, ValueError)):
        await _app(reports).complete_step(StepCompleted(files={"output.mp4": LocalFile(path)}))
    assert reports == []


def test_failed_multi_file_handoff_removes_earlier_snapshots(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"video")
    copied: list[Path] = []
    close = OwnedLocalFile.close

    def record_close(self: OwnedLocalFile) -> None:
        copied.append(self.path.parent)
        close(self)

    monkeypatch.setattr(OwnedLocalFile, "close", record_close)
    with pytest.raises(FileNotFoundError):
        adopt_files({"first.mp4": LocalFile(path), "second.mp4": LocalFile(tmp_path / "missing")})
    assert len(copied) == 2
    assert all(not item.exists() for item in copied)


@pytest.mark.parametrize("reason", ["closed", "bad_session", "full"])
def test_refused_admission_discards_its_snapshot(tmp_path: Path, monkeypatch, reason: str) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"video")
    files = adopt_files({"output.mp4": LocalFile(path)})
    owned = files["output.mp4"]
    assert isinstance(owned, OwnedLocalFile)
    store = StepStore(
        StepResultsConfig(enabled=True, queue=1), lambda _: None, root=tmp_path / "steps"
    )
    if reason == "closed":
        store.close()
    if reason == "full":
        monkeypatch.setattr(store, "_ensure_started", lambda: None)
        assert store.admit(SESSION, 1, CompletedStep(bundle=None))
    assert not store.admit(
        "invalid" if reason == "bad_session" else SESSION,
        2,
        CompletedStep(bundle=None, files=files),
    )
    assert not owned.path.parent.exists()
    store.close()


def test_store_error_discards_snapshot_and_publishes_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"video")
    files = adopt_files({"output.mp4": LocalFile(path)})
    owned = files["output.mp4"]
    assert isinstance(owned, OwnedLocalFile)

    def fail_move(self: OwnedLocalFile, destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(OwnedLocalFile, "move_to", fail_move)
    store = StepStore(StepResultsConfig(enabled=True), lambda _: None, root=tmp_path / "steps")
    assert store.admit(SESSION, 1, CompletedStep(bundle=None, files=files))
    store.close()
    result = json.loads((tmp_path / "steps" / SESSION / "1" / "result.json").read_text())
    assert result["save_error"] == "OSError: disk full"
    assert result["files"] == []
    assert not owned.path.parent.exists()


def test_local_output_cannot_replace_generated_output_or_manifest(tmp_path: Path) -> None:
    from reactor_runtime import Output

    class EmptyOutput(Output):
        pass

    with pytest.raises(ValueError, match="reserved"):
        StepCompleted(output=EmptyOutput(), files={"output.mp4": LocalFile(tmp_path / "file")})
    with pytest.raises(ValueError, match="reserved"):
        StepCompleted(files={"result.json": LocalFile(tmp_path / "file")})
