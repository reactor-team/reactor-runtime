import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.core import (
    MediaBundle,
    RecordingConfig,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.interface.internal.output_buffer import OutputBuffer
from reactor_runtime.recording import (
    ClipManifest,
    ClipResult,
    ClipSessionGoneError,
    Gone,
    NoMediaYetError,
    Pending,
    Recorder,
    RecorderDisabledError,
)

_SID = "00000000-0000-0000-0000-000000000001"


def _serving_recorder(root: Path) -> Recorder:
    """A recorder whose serving root is *root*, for file-driven manifest tests."""
    recorder = Recorder(RecordingConfig(enabled=True, chunk_seconds=4))
    recorder._root = root
    return recorder


def _write_segments(root: Path, *names: str) -> Path:
    session_dir = root / _SID
    session_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (session_dir / name).write_bytes(b"data")
    return session_dir


# -- manifest serving ------------------------------------------------------


def test_manifest_serves_a_closed_range(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")
    result = recorder.manifest(_SID, 0.0, 4.0)
    assert isinstance(result, ClipManifest)
    assert "#EXT-X-MAP" in result.body
    assert f"/clips/chunks/{_SID}/chunk_00000.m4s" in result.body
    assert result.body.endswith("#EXT-X-ENDLIST\n")


def test_manifest_is_pending_until_the_boundary_closes(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    # The boundary segment exists but has no successor and no completion marker.
    _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s")
    assert isinstance(recorder.manifest(_SID, 0.0, 4.0), Pending)


def test_manifest_serves_a_finished_recordings_final_segment(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s", ".complete")
    assert isinstance(recorder.manifest(_SID, 0.0, 4.0), ClipManifest)


def test_manifest_is_gone_for_an_unknown_recording(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    assert isinstance(recorder.manifest(_SID, 0.0, 4.0), Gone)


def test_manifest_is_gone_for_a_malformed_id(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    assert isinstance(recorder.manifest("not-a-uuid", 0.0, 4.0), Gone)


@pytest.mark.parametrize(("start", "end"), [(-1.0, 4.0), (4.0, 4.0), (5.0, 4.0)])
def test_manifest_rejects_a_bad_range(tmp_path: Path, start: float, end: float) -> None:
    recorder = _serving_recorder(tmp_path)
    _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")
    with pytest.raises(ValueError, match="must be a finite"):
        recorder.manifest(_SID, start, end)


# -- chunk serving ---------------------------------------------------------


def test_chunk_path_resolves_an_existing_segment(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    session_dir = _write_segments(tmp_path, "init.mp4")
    assert recorder.chunk_path(_SID, "init.mp4") == session_dir / "init.mp4"


def test_chunk_path_rejects_a_non_artifact_name(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    _write_segments(tmp_path, "init.mp4")
    assert recorder.chunk_path(_SID, "secrets.txt") is None


def test_chunk_path_is_none_for_a_missing_segment(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    _write_segments(tmp_path, "init.mp4")
    assert recorder.chunk_path(_SID, "chunk_00009.m4s") is None


def test_chunk_path_raises_for_a_malformed_id(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    with pytest.raises(ClipSessionGoneError):
        recorder.chunk_path("../etc", "init.mp4")


def test_chunk_path_raises_for_an_unknown_recording(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    with pytest.raises(ClipSessionGoneError):
        recorder.chunk_path(_SID, "init.mp4")


# -- clip requests ---------------------------------------------------------


def test_request_clip_on_a_disabled_recorder_fails() -> None:
    recorder = Recorder(RecordingConfig(enabled=False))
    with pytest.raises(RecorderDisabledError):
        recorder.request_clip(5.0)


def test_request_clip_before_any_media_fails(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        with pytest.raises(NoMediaYetError):
            recorder.request_clip(5.0)
    finally:
        recorder.stop()


def test_request_clip_resolves_to_a_pollable_result(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        assert recorder._markers is not None
        recorder._markers.mark_first_real_frame()
        clip = recorder.request_clip(30.0)
        assert clip.kind == "snap"
        assert clip.session_id == recorder._session_id
        assert clip.playlist_url.startswith("/clips?")
        assert clip.predicted_ready_at_ms > 0
    finally:
        recorder.stop()


def test_request_recording_covers_the_whole_session(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        assert recorder._markers is not None
        recorder._markers.mark_first_real_frame()
        clip = recorder.request_recording()
        assert clip.kind == "recording"
        assert clip.start_marker == 0.0
    finally:
        recorder.stop()


def test_request_clip_rejects_a_non_positive_duration(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        assert recorder._markers is not None
        recorder._markers.mark_first_real_frame()
        with pytest.raises(ValueError, match="positive"):
            recorder.request_clip(0.0)
    finally:
        recorder.stop()


def test_a_landed_clip_notifies_the_consumer(tmp_path: Path) -> None:
    fired: list[ClipResult] = []
    recorder = Recorder(
        RecordingConfig(enabled=True, recording_dir=str(tmp_path)),
        on_clip_ready=fired.append,
    )
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        assert recorder._markers is not None
        assert recorder._session_dir is not None
        recorder._markers.mark_first_real_frame()
        clip = recorder.request_clip(30.0)
        # The boundary segment lands once the encoder has rolled to the next one.
        for name in ("init.mp4", "chunk_00000.m4s", "chunk_00001.m4s"):
            (recorder._session_dir / name).write_bytes(b"data")
        recorder._fire_ready_clips()
        assert fired == [clip]
    finally:
        recorder.stop()


def test_a_landed_clip_fires_once_under_concurrent_callers(tmp_path: Path) -> None:
    # stop() and the watch thread can both call _fire_ready_clips at once; a
    # clip whose boundary has landed must still be announced exactly once.
    fired: list[ClipResult] = []
    recorder = Recorder(
        RecordingConfig(enabled=True, chunk_seconds=4),
        on_clip_ready=fired.append,
    )
    session_dir = _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")
    recorder._root = tmp_path
    recorder._session_dir = session_dir
    recorder._session_id = _SID
    clip = ClipResult(
        session_id=_SID,
        kind="snap",
        start_marker=0.0,
        end_marker=4.0,
        now_marker=4.0,
        predicted_ready_at_ms=0,
        playlist_url="/clips?x",
    )
    recorder._pending.append((0, clip))

    threads = [threading.Thread(target=recorder._fire_ready_clips) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert fired == [clip]


# -- chunk announcements ---------------------------------------------------


def test_closed_chunks_are_announced_in_order(tmp_path: Path) -> None:
    fired: list[tuple[str, int]] = []
    recorder = _serving_recorder(tmp_path)
    recorder._on_chunk_ready = lambda rec, idx: fired.append((rec, idx))
    session_dir = _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")
    recorder._session_dir = session_dir
    recorder._session_id = _SID

    # init is readable and chunk 0 is closed (chunk 1 is its successor); chunk 1
    # has no successor yet, so it is not announced.
    recorder._fire_ready_chunks()
    assert fired == [(_SID, -1), (_SID, 0)]

    # chunk 1 closes once chunk 2 appears; earlier segments are not re-announced.
    (session_dir / "chunk_00002.m4s").write_bytes(b"data")
    recorder._fire_ready_chunks()
    assert fired == [(_SID, -1), (_SID, 0), (_SID, 1)]


def test_the_final_chunk_is_announced_on_completion(tmp_path: Path) -> None:
    fired: list[tuple[str, int]] = []
    recorder = _serving_recorder(tmp_path)
    recorder._on_chunk_ready = lambda rec, idx: fired.append((rec, idx))
    session_dir = _write_segments(tmp_path, "init.mp4", "chunk_00000.m4s")
    recorder._session_dir = session_dir
    recorder._session_id = _SID

    # No successor yet, so only the init segment is announced.
    recorder._fire_ready_chunks()
    assert fired == [(_SID, -1)]

    # The completion marker closes the final segment.
    (session_dir / ".complete").write_bytes(b"")
    recorder._fire_ready_chunks()
    assert fired == [(_SID, -1), (_SID, 0)]


def test_disabled_recorder_never_starts(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    assert recorder._session_dir is None


# -- the encode path (needs ffmpeg) ----------------------------------------


def _video_bundle() -> MediaBundle:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    info = TrackInfo(
        name="main_video", kind=TrackKind.VIDEO, rate=30.0, direction=TrackDirection.OUT
    )
    return MediaBundle(tracks={"main_video": TrackData(info=info, data=frame)})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_encodes_segments_and_serves_a_manifest(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, chunk_seconds=1, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", OutputBuffer({}))
    try:
        bundle = _video_bundle()
        deadline = time.monotonic() + 6.0
        recording_id = recorder._session_id
        assert recording_id is not None
        manifest: object = Pending()
        while time.monotonic() < deadline:
            recorder._on_bundle(bundle, duplicate=False, is_fresh_black=False)
            time.sleep(1.0 / 30.0)
            if recorder._markers is not None and recorder._markers.recording_started:
                manifest = recorder.manifest(recording_id, 0.0, 1.0)
                if isinstance(manifest, ClipManifest):
                    break
        assert isinstance(manifest, ClipManifest)
        assert (tmp_path / recording_id / "init.mp4").is_file()
    finally:
        recorder.stop()
