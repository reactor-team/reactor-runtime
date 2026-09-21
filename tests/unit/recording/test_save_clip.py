from __future__ import annotations

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from reactor_runtime.core import (
    MediaBundle,
    MediaChunk,
    RecordingConfig,
    TrackData,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.recording import ClipManifest, Recorder, RecorderError
from reactor_runtime.recording.chunk_encoder import ChunkEncoder
from reactor_runtime.recording.recorder import _RETENTION_SECONDS, _saved_recording_id

_LIVE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _chunk(
    frames: int = 8,
    fps: float = 24.0,
    *,
    audio: np.ndarray[Any, Any] | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
) -> MediaChunk:
    video = np.full((frames, 16, 16, 3), color, dtype=np.uint8)
    tracks = {
        "video": TrackData(TrackInfo("video", TrackKind.VIDEO, fps), video),
    }
    if audio is not None:
        tracks["audio"] = TrackData(TrackInfo("audio", TrackKind.AUDIO, 48_000), audio)
    return MediaChunk(MediaBundle(tracks), fps=fps, n_frames=frames)


def _decode(root: Path, session_id: str) -> av.container.InputContainer:
    directory = root / session_id
    merged = root / f"{session_id}.mp4"
    parts = [directory / "init.mp4", *sorted(directory.glob("chunk_*.m4s"))]
    merged.write_bytes(b"".join(part.read_bytes() for part in parts))
    return av.open(str(merged))


def _decoded_video_count(root: Path, session_id: str) -> int:
    with _decode(root, session_id) as container:
        return sum(1 for _ in container.decode(video=0))


def _audio_duration_and_energy(root: Path, session_id: str) -> tuple[float, int]:
    with _decode(root, session_id) as container:
        stream = container.streams.audio[0]
        frames = list(container.decode(audio=0))
    samples = sum(frame.samples for frame in frames)
    energy = sum(int(np.abs(frame.to_ndarray()).sum()) for frame in frames)
    assert stream.rate is not None
    return samples / int(stream.rate), energy


def test_save_clip_encodes_101_frame_waveform_with_aligned_audio(tmp_path: Path) -> None:
    recorder = Recorder(
        RecordingConfig(enabled=False, recording_dir=str(tmp_path), chunk_seconds=2)
    )
    frames, fps = 101, 24.0
    target_samples = round(frames * 48_000 / fps)
    waveform = (np.arange(target_samples, dtype=np.int32) % 20_000 - 10_000).astype(np.int16)
    try:
        clip = recorder.save_clip(
            _chunk(frames, fps, audio=waveform.reshape(1, -1)),
            cancelled=lambda: False,
        )
        assert clip.end_marker == pytest.approx(frames / fps)
        with _decode(
            tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or "")
        ) as container:
            stream = container.streams.video[0]
            assert sum(1 for _ in container.decode(video=0)) == frames
            assert stream.average_rate is not None
            assert float(stream.average_rate) == pytest.approx(fps, abs=0.01)
            assert container.duration is not None
            assert container.duration / av.time_base == pytest.approx(frames / fps, abs=0.08)
        audio_duration, energy = _audio_duration_and_energy(
            tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or "")
        )
        assert energy > 0
        assert audio_duration == pytest.approx(frames / fps, abs=0.08)
    finally:
        recorder.close()


def test_save_clip_manifest_is_ready_with_a_partial_final_extinf(tmp_path: Path) -> None:
    recorder = Recorder(
        RecordingConfig(enabled=False, recording_dir=str(tmp_path), chunk_seconds=2)
    )
    try:
        clip = recorder.save_clip(_chunk(101, 24.0), cancelled=lambda: False)
        manifest = recorder.saved_manifest(clip.session_id, clip.clip_id or "")
        assert isinstance(manifest, ClipManifest)
        durations = [float(value) for value in re.findall(r"#EXTINF:([0-9.]+),", manifest.body)]
        assert manifest.body.endswith("#EXT-X-ENDLIST\n")
        assert len(durations) == 3
        assert durations[-1] == pytest.approx(101 / 24.0 - sum(durations[:-1]), abs=0.08)
    finally:
        recorder.close()


def test_save_clip_preserves_fractional_decode_rate(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    fps = 23.976
    try:
        clip = recorder.save_clip(_chunk(48, fps), cancelled=lambda: False)
        with _decode(
            tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or "")
        ) as container:
            stream = container.streams.video[0]
            assert sum(1 for _ in container.decode(video=0)) == 48
            assert stream.average_rate is not None
            assert float(stream.average_rate) == pytest.approx(fps, abs=0.01)
    finally:
        recorder.close()


@pytest.mark.parametrize(
    "chunk",
    [
        MediaChunk(MediaBundle(), fps=24.0),
        MediaChunk(
            MediaBundle(
                {
                    "video": TrackData(
                        TrackInfo("video", TrackKind.VIDEO, 24.0),
                        np.zeros((1, 8, 8, 3), dtype=np.float32),
                    )
                }
            ),
            fps=24.0,
        ),
        MediaChunk(
            MediaBundle(
                {
                    "video": TrackData(
                        TrackInfo("video", TrackKind.VIDEO, 24.0),
                        np.zeros((1, 8, 8, 3), dtype=np.uint8),
                    ),
                    "audio": TrackData(
                        TrackInfo("audio", TrackKind.AUDIO, 48_000),
                        np.zeros((2, 100), dtype=np.int16),
                    ),
                }
            ),
            fps=24.0,
        ),
        MediaChunk(
            MediaBundle(
                {
                    "video": TrackData(
                        TrackInfo("video", TrackKind.VIDEO, 24.0),
                        np.zeros((0, 8, 8, 3), dtype=np.uint8),
                    )
                }
            ),
            fps=24.0,
        ),
        MediaChunk(
            MediaBundle(
                {
                    "video": TrackData(
                        TrackInfo("video", TrackKind.VIDEO, 24.0),
                        np.zeros((1, 8, 8, 3), dtype=np.uint8),
                    ),
                    "audio": TrackData(
                        TrackInfo("audio", TrackKind.AUDIO, float("nan")),
                        np.zeros((1, 100), dtype=np.int16),
                    ),
                }
            ),
            fps=24.0,
        ),
        _chunk(1, float("nan")),
        _chunk(1, float("inf")),
        _chunk(1, 0.0),
    ],
)
def test_save_clip_rejects_invalid_tracks_rates_and_stereo_audio(
    tmp_path: Path, chunk: MediaChunk
) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    try:
        with pytest.raises(ValueError, match="clip"):
            recorder.save_clip(chunk, cancelled=lambda: False)
    finally:
        recorder.close()


def test_save_clip_cancels_after_a_real_encoder_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    original_feed = ChunkEncoder.feed_video
    fed = False

    def feed_then_cancel(self: ChunkEncoder, frame: np.ndarray[Any, Any]) -> None:
        nonlocal fed
        original_feed(self, frame)
        fed = True

    monkeypatch.setattr(ChunkEncoder, "feed_video", feed_then_cancel)
    try:
        with pytest.raises(RecorderError, match="cancelled"):
            recorder.save_clip(_chunk(4), cancelled=lambda: fed)
        assert fed
        assert not list(tmp_path.glob(".save-*"))
        assert not [path for path in tmp_path.iterdir() if path.is_dir()]
    finally:
        recorder.close()


def test_save_clip_strict_finalization_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_callbacks: list[object] = []
    chunk_callbacks: list[object] = []
    recorder = Recorder(
        RecordingConfig(enabled=False, recording_dir=str(tmp_path)),
        on_clip_ready=clip_callbacks.append,
        on_chunk_ready=lambda recording_id, index: chunk_callbacks.append((recording_id, index)),
        on_saved_clip_ready=lambda clip, recording_id: clip_callbacks.append(clip),
    )
    original_stop = ChunkEncoder.stop

    def fail_strict_stop(self: ChunkEncoder, *, strict: bool = False) -> None:
        original_stop(self, strict=strict)
        if strict:
            raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(ChunkEncoder, "stop", fail_strict_stop)
    try:
        with pytest.raises(RuntimeError, match="injected finalization failure"):
            recorder.save_clip(_chunk(3), cancelled=lambda: False)
        assert clip_callbacks == []
        assert chunk_callbacks == []
        assert not list(tmp_path.iterdir())
    finally:
        recorder.close()


def test_saved_ids_are_session_scoped_and_never_overwrite_completed_media(tmp_path: Path) -> None:
    notifications: list[object] = []
    recorder = Recorder(
        RecordingConfig(enabled=False, recording_dir=str(tmp_path)),
        on_chunk_ready=lambda *args: pytest.fail("saved clips must not use live chunk uploads"),
        on_clip_ready=lambda *args: pytest.fail("saved clips must not use live clip events"),
        on_saved_clip_ready=lambda clip, recording_id: notifications.append((clip, recording_id)),
    )
    clip_id = str(uuid.uuid4())
    owners = [str(uuid.uuid4()), str(uuid.uuid4())]
    try:
        for owner in owners:
            clip = recorder.save_clip(
                _chunk(3), session_id=owner, clip_id=clip_id, cancelled=lambda: False
            )
            assert clip.session_id == owner
            assert clip.to_dict()["clip_id"] == clip_id
            assert isinstance(recorder.saved_manifest(owner, clip_id), ClipManifest)
        before = {p: p.read_bytes() for p in tmp_path.rglob("*.m4s")}
        with pytest.raises(FileExistsError):
            recorder.save_clip(
                _chunk(10), session_id=owners[0], clip_id=clip_id, cancelled=lambda: False
            )
        assert {p: p.read_bytes() for p in tmp_path.rglob("*.m4s")} == before
        assert len(notifications) == 2
    finally:
        recorder.close()


def test_save_clips_use_unique_ids_without_moving_live_markers(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=True, recording_dir=str(tmp_path)))
    recorder.start(_LIVE_ID)
    try:
        assert recorder._markers is not None
        recorder._markers.advance(2.5)
        before = recorder._markers.now_marker()
        first = recorder.save_clip(_chunk(2), cancelled=lambda: False)
        second = recorder.save_clip(_chunk(2), cancelled=lambda: False)
        assert first.session_id == second.session_id == _LIVE_ID
        assert first.clip_id != second.clip_id
        assert recorder._session_id == _LIVE_ID
        assert recorder._markers.now_marker() == before
        assert (tmp_path / _LIVE_ID).is_dir()
        assert not (tmp_path / _LIVE_ID / ".complete").exists()
    finally:
        recorder.stop()
        recorder.close()


def test_simultaneous_saves_have_independent_media_and_ids(tmp_path: Path) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))

    def save(index: int) -> tuple[str, tuple[int, int, int]]:
        color = (index * 60, 255 - index * 40, index * 30)
        clip = recorder.save_clip(_chunk(5, color=color), cancelled=lambda: False)
        return _saved_recording_id(clip.session_id, clip.clip_id or ""), color

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(save, range(4)))
        assert len({session_id for session_id, _ in results}) == 4
        for session_id, expected in results:
            assert _decoded_video_count(tmp_path, session_id) == 5
            with _decode(tmp_path, session_id) as container:
                frame = next(container.decode(video=0))
                actual = frame.to_ndarray(format="rgb24").mean(axis=(0, 1))
            assert actual == pytest.approx(expected, abs=15)
    finally:
        recorder.close()


def test_save_clip_reaps_after_completion_when_now_crosses_retention(
    tmp_path: Path,
) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    try:
        clip = recorder.save_clip(_chunk(1), cancelled=lambda: False)
        completion = (
            tmp_path / _saved_recording_id(clip.session_id, clip.clip_id or "") / ".complete"
        )
        finished_at = time.time()
        os.utime(completion, (finished_at, finished_at))
        recorder._reap_expired(finished_at + _RETENTION_SECONDS + 0.1)
        assert not (tmp_path / _saved_recording_id(clip.session_id, clip.clip_id or "")).exists()
    finally:
        recorder.close()


def test_short_video_only_save_has_silent_audio_and_aligned_duration(
    tmp_path: Path,
) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    try:
        clip = recorder.save_clip(_chunk(1, 24.0), cancelled=lambda: False)
        audio_duration, energy = _audio_duration_and_energy(
            tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or "")
        )
        assert (
            _decoded_video_count(tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or ""))
            == 1
        )
        assert energy == 0
        assert audio_duration == pytest.approx(1 / 24.0, abs=0.08)
    finally:
        recorder.close()


@pytest.mark.parametrize("sample_count", [1_000, 20_000])
def test_save_clip_pads_or_trims_audio_to_video_duration(tmp_path: Path, sample_count: int) -> None:
    recorder = Recorder(RecordingConfig(enabled=False, recording_dir=str(tmp_path)))
    frames, fps = 5, 24.0
    samples = np.full((1, sample_count), 8_000, dtype=np.int16)
    try:
        clip = recorder.save_clip(_chunk(frames, fps, audio=samples), cancelled=lambda: False)
        audio_duration, energy = _audio_duration_and_energy(
            tmp_path, _saved_recording_id(clip.session_id, clip.clip_id or "")
        )
        assert energy > 0
        assert audio_duration == pytest.approx(frames / fps, abs=0.08)
    finally:
        recorder.close()
