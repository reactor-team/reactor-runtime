import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from reactor_runtime.core import (
    CompletedStep,
    MediaBundle,
    StepResultsConfig,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.step_results import SavedStep, StepStore
from reactor_runtime.step_results import store as store_module

_SESSION = "7d9f5c1e-0000-4000-8000-000000000042"


def _bundle(frames: int = 3) -> MediaBundle:
    info = TrackInfo(
        name="main_video", kind=TrackKind.VIDEO, rate=0.0, direction=TrackDirection.OUT
    )
    data = np.zeros((frames, 32, 32, 3), dtype=np.uint8)
    return MediaBundle(tracks={"main_video": TrackData(info=info, data=data)})


class _Saved:
    """Collects the steps the store reports saved, for a test to wait on."""

    def __init__(self) -> None:
        self.steps: list[SavedStep] = []
        self._changed = threading.Condition()

    def __call__(self, saved: SavedStep) -> None:
        with self._changed:
            self.steps.append(saved)
            self._changed.notify_all()

    def wait_for(self, count: int) -> list[SavedStep]:
        with self._changed:
            assert self._changed.wait_for(lambda: len(self.steps) >= count, timeout=10.0)
            return list(self.steps)


@pytest.fixture
def saved() -> _Saved:
    return _Saved()


@pytest.fixture
def make_store(tmp_path: Path, saved: _Saved) -> Iterator[Callable[..., StepStore]]:
    stores: list[StepStore] = []

    def make(**settings: Any) -> StepStore:
        store = StepStore(
            StepResultsConfig(enabled=True, **settings), saved, root=tmp_path / "steps"
        )
        stores.append(store)
        return store

    yield make
    for store in stores:
        store.close(drain_seconds=1.0)


def _result(store: StepStore, step: int) -> dict[str, Any]:
    assert store.root is not None
    return json.loads((store.root / _SESSION / str(step) / "result.json").read_text())


def _folder(store: StepStore, step: int) -> set[str]:
    assert store.root is not None
    return {path.name for path in (store.root / _SESSION / str(step)).iterdir()}


def test_a_step_is_saved_as_a_folder_with_its_output_files_and_result(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    step = CompletedStep(
        bundle=_bundle(), files={"prompt.txt": b"a red door"}, elapsed=0.25, fps=12.0
    )
    messages = [{"type": "progress", "data": {"percent": 50}}]

    assert store.admit(_SESSION, 1, step, messages) is True

    assert saved.wait_for(1) == [
        SavedStep(session_id=_SESSION, step=1, files=("output.mp4", "prompt.txt"))
    ]
    assert _folder(store, 1) == {"output.mp4", "prompt.txt", "result.json"}
    result = _result(store, 1)
    assert result["step"] == 1
    assert result["session_id"] == _SESSION
    assert [(entry["name"], entry["content_type"]) for entry in result["files"]] == [
        ("output.mp4", "video/mp4"),
        ("prompt.txt", "text/plain"),
    ]
    assert result["files"][1]["size"] == len(b"a red door")
    assert result["messages"] == messages
    assert result["error"] is None
    assert result["save_error"] is None
    assert result["timings"]["generate_s"] == 0.25
    assert result["timings"]["encode_s"] >= 0
    assert store.root is not None
    with av.open(str(store.root / _SESSION / "1" / "output.mp4")) as container:
        assert container.streams.video[0].average_rate == 12


def test_result_json_is_written_after_every_other_file(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store()
    seen_before_output: list[bool] = []
    real_write = store_module.write_mp4

    def write_and_look(path: Path, *args: Any) -> None:
        seen_before_output.append((path.parent / "result.json").exists())
        real_write(path, *args)

    monkeypatch.setattr(store_module, "write_mp4", write_and_look)
    store.admit(_SESSION, 1, CompletedStep(bundle=_bundle()))
    saved.wait_for(1)

    assert seen_before_output == [False]
    assert ".result.json.tmp" not in _folder(store, 1)


def test_a_step_that_failed_keeps_only_its_result(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None, error="ApplicationError: no seed image"))
    saved.wait_for(1)

    assert _folder(store, 1) == {"result.json"}
    result = _result(store, 1)
    assert result["files"] == []
    assert result["error"] == "ApplicationError: no seed image"


def test_a_step_that_produced_nothing_is_still_saved(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None))

    assert saved.wait_for(1)[0].files == ()
    assert _folder(store, 1) == {"result.json"}
    assert _result(store, 1)["error"] is None


def test_a_step_that_cannot_be_saved_keeps_its_result_with_the_reason(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_encoder(path: Path, *args: Any) -> None:
        path.write_bytes(b"half a file")
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(store_module, "write_mp4", broken_encoder)
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=_bundle(), files={"prompt.txt": b"x"}))

    assert saved.wait_for(1)[0].files == ()
    assert _folder(store, 1) == {"result.json"}
    result = _result(store, 1)
    assert result["files"] == []
    assert result["save_error"] == "RuntimeError: encoder exploded"


def test_a_step_that_finds_the_queue_full_is_not_saved(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoding = threading.Event()
    release = threading.Event()
    real_write = store_module.write_mp4

    def slow_write(path: Path, *args: Any) -> None:
        encoding.set()
        assert release.wait(timeout=10.0)
        real_write(path, *args)

    monkeypatch.setattr(store_module, "write_mp4", slow_write)
    store = make_store(queue=1)

    assert store.admit(_SESSION, 1, CompletedStep(bundle=_bundle())) is True
    assert encoding.wait(timeout=10.0)
    assert store.admit(_SESSION, 2, CompletedStep(bundle=_bundle())) is True
    assert store.admit(_SESSION, 3, CompletedStep(bundle=_bundle())) is False

    release.set()
    assert [step.step for step in saved.wait_for(2)] == [1, 2]
    assert store.root is not None
    assert not (store.root / _SESSION / "3").exists()


def test_a_session_id_that_is_not_a_uuid_is_not_saved(
    make_store: Callable[..., StepStore],
) -> None:
    store = make_store()

    assert store.admit("../../etc", 1, CompletedStep(bundle=None)) is False
    assert store.admit("Not-A-Uuid", 1, CompletedStep(bundle=None)) is False
    assert not (store.root or Path("/nonexistent")).exists()


def test_nothing_is_created_until_the_first_step(tmp_path: Path, saved: _Saved) -> None:
    store = StepStore(StepResultsConfig(enabled=True), saved)

    assert store.root is None
    store.close()


def test_a_closed_store_takes_no_steps(make_store: Callable[..., StepStore]) -> None:
    store = make_store()
    store.close()

    assert store.admit(_SESSION, 1, CompletedStep(bundle=None)) is False


def test_close_lets_pending_saves_finish(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    for step in (1, 2, 3):
        store.admit(_SESSION, step, CompletedStep(bundle=_bundle()))

    store.close(drain_seconds=10.0)

    assert [step.step for step in saved.steps] == [1, 2, 3]


def test_close_abandons_saves_still_waiting_after_the_drain(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoding = threading.Event()
    release = threading.Event()
    real_write = store_module.write_mp4

    def slow_write(path: Path, *args: Any) -> None:
        encoding.set()
        assert release.wait(timeout=10.0)
        real_write(path, *args)

    monkeypatch.setattr(store_module, "write_mp4", slow_write)
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=_bundle()))
    assert encoding.wait(timeout=10.0)
    store.admit(_SESSION, 2, CompletedStep(bundle=_bundle()))

    store.close(drain_seconds=0.1)
    release.set()

    assert [step.step for step in saved.wait_for(1)] == [1]
    worker = store._worker
    assert worker is not None
    worker.join(timeout=10.0)
    assert [step.step for step in saved.steps] == [1]


def test_a_folder_is_deleted_once_its_retention_has_passed(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None))
    saved.wait_for(1)
    assert store.root is not None
    written_at = (store.root / _SESSION / "1" / "result.json").stat().st_mtime

    store._reap_expired(written_at + 299)
    assert (store.root / _SESSION / "1").exists()

    store._reap_expired(written_at + 301)
    assert not (store.root / _SESSION).exists()


def test_a_folder_still_being_written_is_never_deleted(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None))
    saved.wait_for(1)
    assert store.root is not None
    unfinished = store.root / _SESSION / "2"
    unfinished.mkdir()
    written_at = (store.root / _SESSION / "1" / "result.json").stat().st_mtime

    store._reap_expired(written_at + 10_000)

    assert unfinished.exists()
    assert not (store.root / _SESSION / "1").exists()


def test_a_failed_step_keeps_only_its_result_whatever_its_report_carried(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    step = CompletedStep(
        bundle=_bundle(), files={"partial.txt": b"half"}, error="RuntimeError: oom"
    )
    store.admit(_SESSION, 1, step)

    assert saved.wait_for(1)[0].files == ()
    assert _folder(store, 1) == {"result.json"}
    assert _result(store, 1)["error"] == "RuntimeError: oom"


def test_a_reused_session_id_never_shows_a_step_while_it_is_rewritten(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None, files={"note.txt": b"first session"}))
    saved.wait_for(1)

    encoding = threading.Event()
    release = threading.Event()
    real_write = store_module.write_mp4

    def slow_write(path: Path, *args: Any) -> None:
        encoding.set()
        assert release.wait(timeout=10.0)
        real_write(path, *args)

    monkeypatch.setattr(store_module, "write_mp4", slow_write)
    # A later session under the same id reaches its own step 1.
    store.admit(_SESSION, 1, CompletedStep(bundle=_bundle(), files={"note.txt": b"second"}))
    assert encoding.wait(timeout=10.0)

    assert store.root is not None
    step_dir = store.root / _SESSION / "1"
    assert _folder(store, 1) == {"note.txt", "result.json"}
    assert (step_dir / "note.txt").read_bytes() == b"first session"
    assert [f["name"] for f in _result(store, 1)["files"]] == ["note.txt"]

    release.set()
    saved.wait_for(2)
    assert _folder(store, 1) == {"output.mp4", "note.txt", "result.json"}
    assert (step_dir / "note.txt").read_bytes() == b"second"
    assert [path.name for path in (store.root / _SESSION).iterdir()] == ["1"]


def test_a_step_admitted_while_the_store_closes_is_still_saved(
    make_store: Callable[..., StepStore], saved: _Saved, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store()
    admitting = threading.Event()
    release = threading.Event()
    real_start = store._ensure_started

    def paused_start() -> None:
        admitting.set()
        assert release.wait(timeout=10.0)
        real_start()

    monkeypatch.setattr(store, "_ensure_started", paused_start)
    outcome: list[bool] = []
    admitter = threading.Thread(
        target=lambda: outcome.append(store.admit(_SESSION, 1, CompletedStep(bundle=None)))
    )
    admitter.start()
    assert admitting.wait(timeout=10.0)
    closer = threading.Thread(target=store.close)
    closer.start()
    closer.join(timeout=0.2)
    # close() waits for the admission it raced instead of draining around it.
    assert closer.is_alive()

    release.set()
    admitter.join(timeout=10.0)
    closer.join(timeout=10.0)

    assert outcome == [True]
    assert [step.step for step in saved.steps] == [1]
    assert store.admit(_SESSION, 2, CompletedStep(bundle=None)) is False


def test_only_complete_steps_are_listed(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    for step in (2, 1, 10):
        store.admit(_SESSION, step, CompletedStep(bundle=None))
    saved.wait_for(3)
    assert store.root is not None
    (store.root / _SESSION / "11").mkdir()

    assert store.ready_steps(_SESSION) == [1, 2, 10]
    assert store.result_path(_SESSION, 11) is None
    assert store.result_path(_SESSION, 1) == store.root / _SESSION / "1" / "result.json"


def test_a_session_with_no_folder_is_unknown(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    assert store.ready_steps(_SESSION) is None

    store.admit(_SESSION, 1, CompletedStep(bundle=None))
    saved.wait_for(1)

    assert store.ready_steps("00000000-0000-4000-8000-000000000000") is None
    assert store.ready_steps("../" + _SESSION) is None


def test_only_a_file_the_result_lists_is_found(
    make_store: Callable[..., StepStore], saved: _Saved
) -> None:
    store = make_store()
    store.admit(_SESSION, 1, CompletedStep(bundle=None, files={"note.txt": b"hi"}))
    saved.wait_for(1)
    assert store.root is not None
    (store.root / _SESSION / "1" / "stray.bin").write_bytes(b"not listed")

    assert store.file_path(_SESSION, 1, "note.txt") == (
        store.root / _SESSION / "1" / "note.txt",
        "text/plain",
    )
    assert store.file_path(_SESSION, 1, "stray.bin") is None
    assert store.file_path(_SESSION, 1, "result.json") is None
    assert store.file_path(_SESSION, 1, "../1/note.txt") is None
    assert store.file_path(_SESSION, 2, "note.txt") is None
