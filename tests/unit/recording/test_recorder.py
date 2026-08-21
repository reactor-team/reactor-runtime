import contextlib
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.core import (
    MediaBundle,
    MediaChunk,
    RecordingConfig,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
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
from reactor_runtime.recording.recorder import (
    _FEED_DEPTH,
    _FEED_WAIT_SECONDS,
    _RETENTION_SECONDS,
    RECORDING_FPS,
)

_SID = "00000000-0000-0000-0000-000000000001"
# Ceiling on how long a deliberately stalled encoder stays stalled. Well above
# the deadline the tests measure, and absolute rather than per-frame, so a
# regression that waits forever fails in seconds instead of hanging the suite.
_WEDGE_TIMEOUT_SECONDS = 10.0


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
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        with pytest.raises(NoMediaYetError):
            recorder.request_clip(5.0)
    finally:
        recorder.stop()


def test_request_clip_resolves_to_a_pollable_result(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        assert recorder._markers is not None
        recorder._markers.advance(1.0)
        clip = recorder.request_clip(30.0)
        assert clip.kind == "snap"
        assert clip.session_id == recorder._session_id
        assert clip.playlist_url.startswith("/clips?")
        assert clip.predicted_ready_at_ms > 0
    finally:
        recorder.stop()


def test_request_recording_covers_the_whole_session(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        assert recorder._markers is not None
        recorder._markers.advance(1.0)
        clip = recorder.request_recording()
        assert clip.kind == "recording"
        assert clip.start_marker == 0.0
    finally:
        recorder.stop()


def test_request_clip_rejects_a_non_positive_duration(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        assert recorder._markers is not None
        recorder._markers.advance(1.0)
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
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        assert recorder._markers is not None
        assert recorder._session_dir is not None
        recorder._markers.advance(1.0)
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
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert recorder._session_dir is None


# -- the encode path --------------------------------------------------------


def _video_bundle() -> MediaBundle:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    info = TrackInfo(
        name="main_video", kind=TrackKind.VIDEO, rate=30.0, direction=TrackDirection.OUT
    )
    return MediaBundle(tracks={"main_video": TrackData(info=info, data=frame)})


def _av_bundle(width: int, height: int) -> MediaBundle:
    """A bundle at a real output size, with the audio slot that pairs with it."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    video = TrackInfo(
        name="main_video", kind=TrackKind.VIDEO, rate=30.0, direction=TrackDirection.OUT
    )
    audio = TrackInfo(
        name="main_audio", kind=TrackKind.AUDIO, rate=48_000.0, direction=TrackDirection.OUT
    )
    samples = np.zeros((1, 1600), dtype=np.int16)
    return MediaBundle(
        tracks={
            "main_video": TrackData(info=video, data=frame),
            "main_audio": TrackData(info=audio, data=samples),
        }
    )


def _batched_bundle(n_frames: int, fps: float = float(RECORDING_FPS)) -> MediaBundle:
    """A batched video track, the shape a model that emits several frames hands over."""
    data = np.zeros((n_frames, 64, 64, 3), dtype=np.uint8)
    info = TrackInfo(
        name="main_video", kind=TrackKind.VIDEO, rate=fps, direction=TrackDirection.OUT
    )
    return MediaBundle(tracks={"main_video": TrackData(info=info, data=data)})


def _park_feed_worker(recorder: Recorder) -> None:
    """Stop the feed worker so nothing drains what `on_chunk` queues."""
    recorder._feed_stop.set()
    feed_thread = recorder._feed_thread
    assert feed_thread is not None
    feed_thread.join(timeout=2.0)


def _saturate(recorder: Recorder, depth: int) -> None:
    """Fill the feed queue to *depth*, the capacity an emission of that size sees."""
    for _ in range(depth):
        recorder._feed_queue.put_nowait((np.zeros((4, 4, 3), dtype=np.uint8), None))


@contextlib.contextmanager
def _wedged_encoder(recorder: Recorder, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stall the encoder so the feed worker can never open room in the queue.

    ``_feed_stop`` is left clear, so a full queue is refused for the reason the
    refusal exists — the encoder is behind — rather than short-circuited by the
    recording winding down. The worker takes one frame and stalls inside the
    encoder, so saturating one frame past the emission's capacity leaves the
    queue full for as long as this context is open.
    """
    recorder._build_encoder(_batched_bundle(1))
    encoder = recorder._encoder
    assert encoder is not None
    stalled = threading.Event()
    give_up_at = time.monotonic() + _WEDGE_TIMEOUT_SECONDS

    def stall(frame: Any) -> None:
        stalled.wait(max(0.0, give_up_at - time.monotonic()))

    monkeypatch.setattr(encoder, "feed_video", stall)
    try:
        yield
    finally:
        # Released before the recorder is stopped, so teardown never waits out
        # the stall and a failed assertion cannot hang the suite.
        stalled.set()


def test_a_batched_emission_reaches_the_timeline_whole(tmp_path: Path) -> None:
    # A model that batches hands over more frames at once than the queue's own
    # depth. The bound is never applied below the emission being queued, so the
    # whole batch is taken; a queue that gated at its depth instead would keep a
    # fraction of every emission and record a fraction of the media produced.
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        _park_feed_worker(recorder)
        n_frames = _FEED_DEPTH * 8

        recorder.on_chunk(
            MediaChunk(
                bundle=_batched_bundle(n_frames),
                fps=float(RECORDING_FPS),
                n_frames=n_frames,
                wait=True,
            )
        )

        assert recorder._feed_queue.qsize() == n_frames
        assert recorder._dropped_frames == 0
        markers = recorder._markers
        assert markers is not None
        assert markers.now_marker() == pytest.approx(n_frames / RECORDING_FPS)
    finally:
        recorder.stop()


def test_a_batch_slower_than_the_grid_records_its_true_duration(tmp_path: Path) -> None:
    # The shape a real batching model emits: frames at a rate below the recording
    # grid, so the batch resamples up to more grid frames than it carries. The
    # timeline has to reach the media time the emission represents, since that is
    # what a clip's marker range and the encoded bytes are both read against.
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        _park_feed_worker(recorder)
        n_frames, fps = 33, 20.0

        recorder.on_chunk(
            MediaChunk(bundle=_batched_bundle(n_frames, fps), fps=fps, n_frames=n_frames, wait=True)
        )

        markers = recorder._markers
        assert markers is not None
        assert markers.now_marker() == pytest.approx(n_frames / fps, abs=1.0 / RECORDING_FPS)
        assert recorder._dropped_frames == 0
    finally:
        recorder.stop()


def test_a_saturated_feed_queue_drops_the_whole_overflow_and_keeps_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An encoder that falls behind costs frames, never the session. A chunk that
    # prefers skipping to waiting has its overflow dropped, and every abandoned
    # frame is counted: a count that stopped at the first one would report a
    # recording losing most of its media as losing a frame.
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        with _wedged_encoder(recorder, monkeypatch):
            n_frames = _FEED_DEPTH * 2
            _saturate(recorder, n_frames + 1)

            recorder.on_chunk(
                MediaChunk(
                    bundle=_batched_bundle(n_frames),
                    fps=float(RECORDING_FPS),
                    n_frames=n_frames,
                    wait=False,
                )
            )

            assert recorder._dropped_frames == n_frames
            assert not recorder._disabled
    finally:
        recorder.stop()


def test_a_waiting_emission_gives_up_on_an_encoder_that_never_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Backpressure is bounded, so a wedged encoder stalls the recording rather
    # than the session: the emission waits its budget for room, then abandons
    # what is left and hands the model thread back. The wait has to be the
    # deadline expiring and nothing else, so the band excludes a zero wait —
    # an emission that never waited would satisfy "gives up" for the wrong
    # reason and leave the budget itself unmeasured.
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        with _wedged_encoder(recorder, monkeypatch):
            n_frames = _FEED_DEPTH * 2
            _saturate(recorder, n_frames + 1)

            started = time.monotonic()
            recorder.on_chunk(
                MediaChunk(
                    bundle=_batched_bundle(n_frames),
                    fps=float(RECORDING_FPS),
                    n_frames=n_frames,
                    wait=True,
                )
            )
            elapsed = time.monotonic() - started

            assert _FEED_WAIT_SECONDS <= elapsed < 2 * _FEED_WAIT_SECONDS
            assert recorder._dropped_frames == n_frames
            assert not recorder._disabled
    finally:
        recorder.stop()


def test_a_recording_that_stops_mid_emission_reports_no_dropped_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Teardown is not encoder pressure. A stop that lands while an emission is
    # parked on a full queue releases it, and the frames it never handed over
    # are not the encoder falling behind — counting them would put phantom
    # losses on the very summary stop() logs as the recording's health.
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        with _wedged_encoder(recorder, monkeypatch):
            n_frames = _FEED_DEPTH * 2
            _saturate(recorder, n_frames + 1)
            emitting = threading.Thread(
                target=recorder.on_chunk,
                args=(
                    MediaChunk(
                        bundle=_batched_bundle(n_frames),
                        fps=float(RECORDING_FPS),
                        n_frames=n_frames,
                        wait=True,
                    ),
                ),
                daemon=True,
            )
            emitting.start()
            # Park the emission on the full queue before the stop lands, so the
            # stop is what releases it rather than the deadline.
            time.sleep(0.2)

            recorder.stop()
            emitting.join(timeout=2 * _FEED_WAIT_SECONDS)

            assert not emitting.is_alive()
            assert recorder._dropped_frames == 0
    finally:
        recorder.stop()


def test_encodes_segments_and_serves_a_manifest(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, chunk_seconds=1, recording_dir=str(tmp_path)))
    recorder.start("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    try:
        bundle = _video_bundle()
        deadline = time.monotonic() + 6.0
        recording_id = recorder._session_id
        assert recording_id is not None
        manifest: object = Pending()
        while time.monotonic() < deadline:
            recorder.on_chunk(MediaChunk(bundle=bundle, fps=30.0, n_frames=1))
            time.sleep(1.0 / 30.0)
            if recorder._markers is not None and recorder._markers.recording_started:
                manifest = recorder.manifest(recording_id, 0.0, 1.0)
                if isinstance(manifest, ClipManifest):
                    break
        assert isinstance(manifest, ClipManifest)
        assert (tmp_path / recording_id / "init.mp4").is_file()
    finally:
        recorder.stop()


@pytest.mark.parametrize("attempt", range(3))
def test_records_a_real_frame_size_with_audio(tmp_path: Path, attempt: int) -> None:
    # A recording at a size a model actually emits, with audio alongside it, has to
    # close segments and lose almost nothing. An encoder that stalls takes nothing
    # at all and drops every frame, so the bound below separates a stalled encoder
    # from a merely busy host. Repeated, because a stall is timing-dependent.
    root = tmp_path / str(attempt)
    recorder = Recorder(RecordingConfig(enabled=True, chunk_seconds=1, recording_dir=str(root)))
    recorder.start(_SID)
    try:
        bundle = _av_bundle(800, 600)
        recording_id = recorder._session_id
        assert recording_id is not None
        deadline = time.monotonic() + 20.0
        manifest: object = Pending()
        offered = 0
        while time.monotonic() < deadline:
            recorder.on_chunk(MediaChunk(bundle=bundle, fps=30.0, n_frames=1))
            offered += 1
            time.sleep(1.0 / 30.0)
            manifest = recorder.manifest(recording_id, 0.0, 1.0)
            if isinstance(manifest, ClipManifest):
                break
        assert isinstance(manifest, ClipManifest)
        assert recorder._dropped_frames < offered // 4
    finally:
        recorder.stop()


# -- retention -------------------------------------------------------------


def _finished_recording(root: Path, name: str, *, finished_at: float) -> Path:
    """A recording directory carrying a completion marker aged to *finished_at*."""
    session_dir = root / name
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "init.mp4").write_bytes(b"data")
    marker = session_dir / ".complete"
    marker.write_text("")
    os.utime(marker, (finished_at, finished_at))
    return session_dir


def test_reap_deletes_a_recording_past_its_retention_window(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    now = time.time()
    aged = _finished_recording(tmp_path, _SID, finished_at=now - _RETENTION_SECONDS - 60)
    recorder._reap_expired(now)
    assert not aged.exists()


def test_reap_keeps_a_recently_finished_recording(tmp_path: Path) -> None:
    recorder = _serving_recorder(tmp_path)
    now = time.time()
    fresh = _finished_recording(tmp_path, _SID, finished_at=now - 5)
    recorder._reap_expired(now)
    assert fresh.exists()


def test_reap_never_touches_an_in_progress_recording(tmp_path: Path) -> None:
    # A live recording carries no completion marker, so it is never a candidate
    # for reaping no matter how long the session has been running.
    recorder = _serving_recorder(tmp_path)
    live = tmp_path / _SID
    live.mkdir(parents=True, exist_ok=True)
    (live / "chunk_00000.m4s").write_bytes(b"data")
    old = time.time() - _RETENTION_SECONDS * 10
    os.utime(live, (old, old))
    recorder._reap_expired(time.time())
    assert live.exists()


def test_close_is_idempotent_when_the_reaper_never_started(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.close()
    recorder.close()


def test_start_clears_a_stale_completion_marker_from_a_reused_id(tmp_path: Path) -> None:
    # A recording started under an id used before must not inherit the earlier
    # run's completion marker, or the reaper would read the live recording as
    # finished and delete it mid-write.
    session_dir = tmp_path / _SID
    session_dir.mkdir(parents=True)
    (session_dir / ".complete").write_text("")
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        assert not (session_dir / ".complete").exists()
    finally:
        recorder.stop()
        recorder.close()


def test_reap_skips_the_active_recording_even_with_a_stale_marker(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_SID)
    try:
        active = recorder._session_dir
        assert active is not None
        marker = active / ".complete"
        marker.write_text("")
        old = time.time() - _RETENTION_SECONDS * 2
        os.utime(marker, (old, old))
        recorder._reap_expired(time.time())
        assert active.exists()
    finally:
        recorder.stop()
        recorder.close()
