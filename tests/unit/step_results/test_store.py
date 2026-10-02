import json
import logging
import threading
import time
from collections.abc import Callable
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
    TrackInfo,
    TrackKind,
)
from reactor_runtime.step_results import (
    StepResult,
    StepResultCancelledError,
    StepResultsDisabledError,
    StepResultStore,
)
from reactor_runtime.step_results import store as store_module

SID = "00000000-0000-0000-0000-000000000001"
OTHER = "00000000-0000-0000-0000-000000000002"


def _bundle(frames: int = 4, *, audio: bool = False) -> MediaBundle:
    tracks = {
        "main_video": TrackData(
            info=TrackInfo(name="main_video", kind=TrackKind.VIDEO),
            data=np.zeros((frames, 16, 16, 3), dtype=np.uint8),
        )
    }
    if audio:
        tracks["main_audio"] = TrackData(
            info=TrackInfo(name="main_audio", kind=TrackKind.AUDIO, rate=48_000.0),
            data=np.zeros((1, 8_000), dtype=np.int16),
        )
    return MediaBundle(tracks=tracks)


def _store(
    tmp_path: Path, *, enabled: bool = True, ready: list[StepResult] | None = None
) -> StepResultStore:
    config = StepResultsConfig(enabled=enabled, step_results_dir=str(tmp_path / "root"))
    return StepResultStore(config, on_ready=ready.append if ready is not None else None)


def _result(tmp_path: Path, step: int, sid: str = SID) -> dict[str, Any]:
    return json.loads((tmp_path / "root" / sid / "steps" / str(step) / "result.json").read_text())


# -- numbering --------------------------------------------------------------------


def test_steps_are_numbered_from_one_in_save_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)

    first = store.save(_bundle(), 24.0)
    second = store.save(_bundle(), 24.0)

    assert (first.step, second.step) == (1, 2)
    assert store.saved_count == 2
    assert first.session_id == SID


def test_the_count_resets_for_the_next_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0)
    store.stop()
    store.start(OTHER)

    result = store.save(_bundle(), 24.0)

    assert result.step == 1
    assert result.session_id == OTHER
    assert (tmp_path / "root" / SID / "steps" / "1" / "result.json").is_file()


def test_a_save_outside_a_session_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(StepResultsDisabledError, match="no session"):
        store.save(_bundle(), 24.0)


def test_a_disabled_store_refuses_a_save_and_starts_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path, enabled=False)
    store.start(SID)

    with pytest.raises(StepResultsDisabledError, match="not enabled"):
        store.save(_bundle(), 24.0)
    assert not (tmp_path / "root").exists()


def test_a_session_id_that_is_not_a_uuid_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="lowercase UUID"):
        store.start("../escape")


# -- the folder ---------------------------------------------------------------------


def test_the_folder_holds_the_media_the_extras_and_the_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    extra = tmp_path / "mask.bin"
    extra.write_bytes(b"mask")

    result = store.save(
        _bundle(audio=True),
        24.0,
        {"last_frame.png": b"\x89PNG", "mask.bin": extra},
        messages=[{"type": "chunk_done", "data": {"index": 0}}],
        timings={"generate_s": 1.25},
    )

    folder = tmp_path / "root" / SID / "steps" / "1"
    assert sorted(path.name for path in folder.iterdir()) == [
        "last_frame.png",
        "mask.bin",
        "output.mp4",
        "result.json",
    ]
    assert (folder / "mask.bin").read_bytes() == b"mask"
    assert result.files == ["output.mp4", "last_frame.png", "mask.bin", "result.json"]
    document = _result(tmp_path, 1)
    assert document["step"] == 1
    assert document["files"] == [
        {"name": "output.mp4", "content_type": "video/mp4"},
        {"name": "last_frame.png", "content_type": "image/png"},
        {"name": "mask.bin", "content_type": "application/octet-stream"},
    ]
    assert [track["name"] for track in document["tracks"]] == ["main_video", "main_audio"]
    assert document["messages"] == [{"type": "chunk_done", "data": {"index": 0}}]
    assert document["timings"]["generate_s"] == 1.25
    assert document["timings"]["encode_s"] > 0
    with av.open(str(folder / "output.mp4")) as container:
        assert [stream.type for stream in container.streams] == ["video", "audio"]


def test_a_save_without_messages_writes_no_messages_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)

    store.save(_bundle(), 24.0, {"note.txt": b"hi"})

    assert "messages" not in _result(tmp_path, 1)


def test_a_files_only_step_has_no_media(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)

    result = store.save(None, 24.0, {"report.json": b"{}"})

    assert result.files == ["report.json", "result.json"]
    document = _result(tmp_path, 1)
    assert document["tracks"] == []
    assert not (tmp_path / "root" / SID / "steps" / "1" / "output.mp4").exists()


def test_an_empty_step_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)

    with pytest.raises(ValueError, match="media or at least one file"):
        store.save(None, 24.0, {})


def test_result_json_is_written_last_and_the_folder_appears_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.start(SID)
    steps_dir = tmp_path / "root" / SID / "steps"
    order: list[str] = []
    seen_before_rename: list[list[str]] = []

    real_write_file = store_module._write_file
    real_write_result = store_module._write_result

    def spy_file(path: Path, payload: Any) -> None:
        order.append(path.name)
        real_write_file(path, payload)

    def spy_result(folder: Path, document: Any) -> None:
        order.append("result.json")
        real_write_result(folder, document)
        seen_before_rename.append(sorted(child.name for child in steps_dir.iterdir()))

    monkeypatch.setattr(store_module, "_write_file", spy_file)
    monkeypatch.setattr(store_module, "_write_result", spy_result)

    store.save(_bundle(), 24.0, {"a.txt": b"a", "b.txt": b"b"})

    assert order[-1] == "result.json"
    assert order[:-1] == ["a.txt", "b.txt"]
    # While result.json was being written, only the hidden partial existed.
    assert seen_before_rename == [[".1.partial"]]
    assert sorted(child.name for child in steps_dir.iterdir()) == ["1"]


def test_a_failed_save_leaves_no_partial_behind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)

    with pytest.raises(FileNotFoundError):
        store.save(_bundle(), 24.0, {"missing.bin": tmp_path / "nowhere"})

    steps_dir = tmp_path / "root" / SID / "steps"
    assert list(steps_dir.iterdir()) == []
    assert store.saved_count == 0


# -- file names ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["", "sub/dir.png", "..", ".hidden", "a\\b", "back/../out", "sp ace"]
)
def test_a_file_name_that_is_not_one_plain_segment_is_refused(tmp_path: Path, name: str) -> None:
    store = _store(tmp_path)
    store.start(SID)

    with pytest.raises(ValueError, match="one plain path segment"):
        store.save(_bundle(), 24.0, {name: b"x"})


@pytest.mark.parametrize("name", ["result.json", "output.mp4"])
def test_a_file_name_the_runtime_writes_is_refused(tmp_path: Path, name: str) -> None:
    store = _store(tmp_path)
    store.start(SID)

    with pytest.raises(ValueError, match="written by the runtime"):
        store.save(_bundle(), 24.0, {name: b"x"})


# -- the rolling window ---------------------------------------------------------------


def _windowed(tmp_path: Path, keep_last: int | None) -> StepResultStore:
    config = StepResultsConfig(
        enabled=True, keep_last=keep_last, step_results_dir=str(tmp_path / "root")
    )
    store = StepResultStore(config)
    store.start(SID)
    return store


def _present(tmp_path: Path) -> list[int]:
    steps_dir = tmp_path / "root" / SID / "steps"
    return sorted(
        int(child.name) for child in steps_dir.iterdir() if not child.name.startswith(".")
    )


def test_without_keep_last_every_step_stays_until_the_session_ends(tmp_path: Path) -> None:
    store = _windowed(tmp_path, None)
    for _ in range(4):
        store.save(None, 24.0, {"a.txt": b"a"})

    assert _present(tmp_path) == [1, 2, 3, 4]


def test_keep_last_drops_the_step_that_fell_out_of_the_window(tmp_path: Path) -> None:
    store = _windowed(tmp_path, 2)

    store.save(None, 24.0, {"a.txt": b"a"})
    store.save(None, 24.0, {"a.txt": b"a"})
    assert _present(tmp_path) == [1, 2]

    store.save(None, 24.0, {"a.txt": b"a"})
    assert _present(tmp_path) == [2, 3]

    store.save(None, 24.0, {"a.txt": b"a"})
    assert _present(tmp_path) == [3, 4]


def test_a_trimmed_step_is_gone_from_the_served_surface_too(tmp_path: Path) -> None:
    store = _windowed(tmp_path, 1)
    store.save(_bundle(), 24.0, {"a.txt": b"a"})
    store.save(_bundle(), 24.0, {"a.txt": b"a"})

    assert store.list_steps(SID) == [{"step": 2, "files": ["output.mp4", "a.txt", "result.json"]}]
    assert store.result_path(SID, 1) is None
    assert store.file_path(SID, 1, "output.mp4") is None
    assert store.result_path(SID, 2) is not None


def test_an_error_step_counts_toward_the_window(tmp_path: Path) -> None:
    store = _windowed(tmp_path, 1)
    store.save(None, 24.0, {"a.txt": b"a"})

    store.fail("internal_error", "boom")

    assert _present(tmp_path) == [2]


def test_the_window_resets_with_the_session(tmp_path: Path) -> None:
    store = _windowed(tmp_path, 1)
    store.save(None, 24.0, {"a.txt": b"a"})
    store.save(None, 24.0, {"a.txt": b"a"})
    store.stop()
    store.start(OTHER)

    store.save(None, 24.0, {"a.txt": b"a"})

    # The first session kept its last step; the second starts its own count.
    assert _present(tmp_path) == [2]
    assert [entry["step"] for entry in store.list_steps(OTHER) or []] == [1]
    store.close()


def test_the_window_trims_through_the_worker_as_well(tmp_path: Path) -> None:
    store = _windowed(tmp_path, 2)
    for _ in range(5):
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"a.txt": b"a"}))

    store.stop()

    assert _present(tmp_path) == [4, 5]
    store.close()


# -- fail ---------------------------------------------------------------------------


def test_fail_writes_an_error_manifest_as_a_step_of_its_own(tmp_path: Path) -> None:
    ready: list[StepResult] = []
    store = _store(tmp_path, ready=ready)
    store.start(SID)
    store.save(_bundle(), 24.0)

    result = store.fail("invalid_command", "seconds must be between 5 and 14")

    assert result == StepResult(SID, 2, ["result.json"])
    assert ready[-1] == result
    assert _result(tmp_path, 2) == {
        "step": 2,
        "files": [],
        "error": {"code": "invalid_command", "message": "seconds must be between 5 and 14"},
    }
    assert store.saved_count == 2


def test_fail_without_a_session_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.fail("internal_error", "boom") is None
    assert not (tmp_path / "root").exists()


def test_fail_on_a_disabled_store_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path, enabled=False)
    store.start(SID)

    assert store.fail("internal_error", "boom") is None


# -- announcement -------------------------------------------------------------------


def test_a_save_is_announced_once_the_folder_is_in_place(tmp_path: Path) -> None:
    seen: list[tuple[StepResult, bool]] = []
    steps_dir = tmp_path / "root" / SID / "steps"
    store = StepResultStore(
        StepResultsConfig(enabled=True, step_results_dir=str(tmp_path / "root")),
        on_ready=lambda result: seen.append(
            (result, (steps_dir / str(result.step) / "result.json").is_file())
        ),
    )
    store.start(SID)

    result = store.save(_bundle(), 24.0)

    assert seen == [(result, True)]


def test_a_failing_announcement_does_not_fail_the_save(tmp_path: Path) -> None:
    def explode(result: StepResult) -> None:
        raise RuntimeError("listener broke")

    store = StepResultStore(
        StepResultsConfig(enabled=True, step_results_dir=str(tmp_path / "root")),
        on_ready=explode,
    )
    store.start(SID)

    assert store.save(_bundle(), 24.0).step == 1


# -- cancellation -------------------------------------------------------------------


def test_a_cancelled_save_removes_its_partial(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    polls = iter([False, False, True])

    with pytest.raises(StepResultCancelledError):
        store.save(_bundle(frames=12), 24.0, cancelled=lambda: next(polls, True))

    assert list((tmp_path / "root" / SID / "steps").iterdir()) == []
    assert store.saved_count == 0


def test_a_closed_store_refuses_a_save(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.close()

    with pytest.raises(StepResultCancelledError):
        store.save(_bundle(), 24.0)


def test_saves_are_serialised(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    results: list[int] = []
    threads = [
        threading.Thread(target=lambda: results.append(store.save(_bundle(), 24.0).step))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [1, 2, 3, 4]
    assert store.list_steps(SID) is not None
    assert [entry["step"] for entry in store.list_steps(SID) or []] == [1, 2, 3, 4]


# -- the worker ---------------------------------------------------------------------


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def test_enqueue_returns_at_once_and_the_worker_saves_the_step(tmp_path: Path) -> None:
    ready: list[StepResult] = []
    store = _store(tmp_path, ready=ready)
    store.start(SID)

    started = time.perf_counter()
    store.enqueue(CompletedStep(bundle=_bundle(frames=48), fps=24.0, files={"a.txt": b"a"}))
    handed_over = time.perf_counter() - started

    assert handed_over < 0.1
    _wait_for(lambda: len(ready) == 1)
    assert ready == [StepResult(SID, 1, ["output.mp4", "a.txt", "result.json"])]
    assert _result(tmp_path, 1)["files"][1] == {"name": "a.txt", "content_type": "text/plain"}
    store.close()


def test_queued_steps_are_saved_in_the_order_they_were_announced(tmp_path: Path) -> None:
    ready: list[StepResult] = []
    store = _store(tmp_path, ready=ready)
    store.start(SID)

    for index in range(3):
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={f"{index}.txt": b"x"}))

    _wait_for(lambda: len(ready) == 3)
    assert [result.step for result in ready] == [1, 2, 3]
    assert [result.files[0] for result in ready] == ["0.txt", "1.txt", "2.txt"]
    store.close()


def test_stop_finishes_the_queued_steps_before_the_session_is_marked_finished(
    tmp_path: Path,
) -> None:
    ready: list[StepResult] = []
    store = _store(tmp_path, ready=ready)
    store.start(SID)
    for _ in range(3):
        store.enqueue(CompletedStep(bundle=_bundle(frames=24), fps=24.0))

    store.stop()

    assert len(ready) == 3
    assert (tmp_path / "root" / SID / ".complete").is_file()
    assert store.list_steps(SID) is not None
    assert [entry["step"] for entry in store.list_steps(SID) or []] == [1, 2, 3]
    store.close()


def test_a_step_announced_with_no_session_is_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)

    with caplog.at_level(logging.WARNING):
        store.enqueue(CompletedStep(bundle=_bundle(), fps=24.0))

    assert any("no session to save it into" in r.message for r in caplog.records)
    assert not (tmp_path / "root").exists()


def test_a_step_the_worker_cannot_save_is_logged_and_the_next_one_still_lands(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ready: list[StepResult] = []
    store = _store(tmp_path, ready=ready)
    store.start(SID)

    with caplog.at_level(logging.ERROR):
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"bad/name": b"x"}))
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"fine.txt": b"x"}))
        _wait_for(lambda: len(ready) == 1)

    assert any("failed to save the step result" in r.message for r in caplog.records)
    assert ready[0].step == 1
    assert ready[0].files == ["fine.txt", "result.json"]
    store.close()


def test_the_worker_waits_for_room_rather_than_piling_steps_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bound is what keeps a slow disk from turning into unbounded memory:
    # past it, an announcement waits for the worker instead of queueing.
    store = _store(tmp_path)
    store.start(SID)
    store._pending.maxsize = 1
    release = threading.Event()
    announced: list[float] = []

    def slow_save(*args: Any, **kwargs: Any) -> StepResult:
        release.wait(timeout=5.0)
        return StepResult(SID, 1)

    # The worker looks `save` up on the instance at call time, so a stand-in
    # that only waits is what makes the queue fill.
    monkeypatch.setattr(store, "save", slow_save)
    store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"a": b""}))  # taken by the worker

    def announce() -> None:
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"b": b""}))  # fills the one slot
        store.enqueue(CompletedStep(bundle=None, fps=24.0, files={"c": b""}))  # must wait
        announced.append(time.perf_counter())

    thread = threading.Thread(target=announce)
    thread.start()
    time.sleep(0.2)
    assert announced == []
    release.set()
    thread.join(timeout=5.0)
    assert len(announced) == 1
    store.close()


# -- serving ------------------------------------------------------------------------


def test_list_steps_names_the_complete_steps_in_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0, {"a.txt": b"a"})
    store.save(None, 24.0, {"b.txt": b"b"})
    # A partial folder and a stray file are not steps.
    (tmp_path / "root" / SID / "steps" / ".3.partial").mkdir()
    (tmp_path / "root" / SID / "steps" / "notes").write_text("")

    assert store.list_steps(SID) == [
        {"step": 1, "files": ["output.mp4", "a.txt", "result.json"]},
        {"step": 2, "files": ["b.txt", "result.json"]},
    ]


def test_list_steps_is_none_for_an_unknown_or_malformed_session(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.list_steps(SID) is None
    store.start(SID)
    assert store.list_steps(OTHER) is None
    assert store.list_steps("../root") is None
    assert store.list_steps(SID) == []


def test_result_path_points_at_a_complete_step_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0)
    (tmp_path / "root" / SID / "steps" / "2").mkdir()

    assert store.result_path(SID, 1) == tmp_path / "root" / SID / "steps" / "1" / "result.json"
    assert store.result_path(SID, 2) is None
    assert store.result_path(SID, 0) is None
    assert store.result_path(OTHER, 1) is None


def test_file_path_serves_only_what_the_manifest_lists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0, {"last_frame.png": b"png"})
    folder = tmp_path / "root" / SID / "steps" / "1"
    (folder / "secret.txt").write_text("not listed")

    assert store.file_path(SID, 1, "output.mp4") == (folder / "output.mp4", "video/mp4")
    assert store.file_path(SID, 1, "last_frame.png") == (folder / "last_frame.png", "image/png")
    assert store.file_path(SID, 1, "result.json") == (folder / "result.json", "application/json")
    assert store.file_path(SID, 1, "secret.txt") is None
    assert store.file_path(SID, 1, "../1/result.json") is None
    assert store.file_path(SID, 2, "output.mp4") is None


# -- retention ----------------------------------------------------------------------


def test_the_steps_of_a_finished_session_stay_until_the_window_passes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0)
    store.stop()
    session_dir = tmp_path / "root" / SID

    store._reap_expired(time.time())
    assert session_dir.is_dir()

    store._reap_expired(time.time() + store_module._RETENTION_SECONDS + 1)
    assert not session_dir.exists()
    store.close()


def test_the_live_session_is_never_reaped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0)
    session_dir = tmp_path / "root" / SID
    # A marker from an earlier run under the same id would read as finished.
    (session_dir / ".complete").write_text("")
    store.start(SID)

    store._reap_expired(time.time() + store_module._RETENTION_SECONDS * 10)

    assert session_dir.is_dir()
    store.close()


def test_a_second_session_under_the_same_id_is_not_reaped_by_the_first_marker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.start(SID)
    store.save(_bundle(), 24.0)
    store.stop()
    store.start(SID)

    assert not (tmp_path / "root" / SID / ".complete").exists()
    store.close()
