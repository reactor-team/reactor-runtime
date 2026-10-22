"""The frames a step request carries in, and a step result carries out."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_runtime.core.values import MediaBundle, TrackData, TrackInfo, TrackKind
from reactor_runtime.protocol.frames import (
    FrameEncodingError,
    check_frame,
    decode_frame,
    encode_frame,
    track_frames,
)
from reactor_wire.v1 import platform_pb2, step_pb2

RAW = step_pb2.FRAME_ENCODING_RAW


def _raw(
    array: np.ndarray, *, metadata: bytes | None = None, capture_time_us: int | None = None
) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=RAW,
        dtype=str(array.dtype),
        shape=list(array.shape),
        data=array.astype(array.dtype.newbyteorder("<")).tobytes(),
        metadata=metadata,
        capture_time_us=capture_time_us,
    )


def test_a_raw_video_frame_decodes_to_the_rgb_array_it_carried() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = decode_frame(_raw(image), TrackKind.VIDEO)
    assert frame.data.dtype == np.uint8
    np.testing.assert_array_equal(frame.data, image)
    assert frame.metadata is None
    assert frame.capture_time_us is None


def test_a_raw_mono_audio_frame_decodes_to_the_samples_it_carried() -> None:
    pcm = np.array([[1, -2, 300]], dtype=np.int16)
    frame = decode_frame(_raw(pcm), TrackKind.AUDIO)
    assert frame.data.dtype == np.int16
    np.testing.assert_array_equal(frame.data, pcm)


def test_a_raw_audio_frame_with_several_channels_is_mixed_down_to_mono() -> None:
    pcm = np.array([[1, -2, 300], [5, 6, -32768]], dtype=np.int16)
    frame = decode_frame(_raw(pcm), TrackKind.AUDIO)
    assert frame.data.dtype == np.int16
    np.testing.assert_array_equal(frame.data, np.array([[3, 2, -16234]], dtype=np.int16))


def test_a_decoded_frame_is_writable_and_owns_its_samples() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    frame = decode_frame(_raw(image), TrackKind.VIDEO)
    frame.data[0, 0, 0] = 9
    assert frame.data[0, 0, 0] == 9


def test_metadata_and_capture_time_ride_along() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    frame = decode_frame(_raw(image, metadata=b'{"seed":1}', capture_time_us=42), TrackKind.VIDEO)
    assert frame.metadata == b'{"seed":1}'
    assert frame.capture_time_us == 42


@pytest.mark.parametrize("encoding", [step_pb2.FRAME_ENCODING_UNSPECIFIED, 99])
def test_an_unknown_encoding_is_refused(encoding: int) -> None:
    frame = _raw(np.zeros((1, 1, 3), dtype=np.uint8))
    # Proto3 enums are open: a peer may send a value this build has no name for.
    setattr(frame, "encoding", encoding)  # noqa: B010
    with pytest.raises(FrameEncodingError, match="not supported"):
        decode_frame(frame, TrackKind.VIDEO)


@pytest.mark.parametrize(
    ("array", "kind", "match"),
    [
        (np.zeros((1, 1, 3), dtype=np.float32), TrackKind.VIDEO, "uint8"),
        (np.zeros((1, 1, 4), dtype=np.uint8), TrackKind.VIDEO, r"\(height, width, 3\)"),
        (np.zeros((2, 3), dtype=np.uint8), TrackKind.VIDEO, "3 non-zero dimensions"),
        (np.zeros((1, 3), dtype=np.uint8), TrackKind.AUDIO, "int16"),
        (np.zeros((4,), dtype=np.int16), TrackKind.AUDIO, "2 non-zero dimensions"),
        (np.zeros((1, 0), dtype=np.int16), TrackKind.AUDIO, "2 non-zero dimensions"),
    ],
)
def test_a_frame_that_does_not_fit_the_track_kind_is_refused(
    array: np.ndarray, kind: TrackKind, match: str
) -> None:
    with pytest.raises(FrameEncodingError, match=match):
        decode_frame(_raw(array), kind)


def test_bytes_that_do_not_fill_the_shape_are_refused() -> None:
    frame = _raw(np.zeros((2, 2, 3), dtype=np.uint8))
    frame.data = frame.data[:-1]
    with pytest.raises(FrameEncodingError, match="is 12 bytes, got 11"):
        decode_frame(frame, TrackKind.VIDEO)


@pytest.mark.parametrize(
    ("array", "kind"),
    [
        (np.zeros((2, 3, 3), dtype=np.uint8), TrackKind.VIDEO),
        (np.zeros((1, 5), dtype=np.int16), TrackKind.AUDIO),
        (np.zeros((2, 5), dtype=np.int16), TrackKind.AUDIO),
    ],
)
def test_a_checked_frame_has_the_shape_it_decodes_to(array: np.ndarray, kind: TrackKind) -> None:
    frame = _raw(array)
    assert check_frame(frame, kind) == decode_frame(frame, kind).data.shape


def test_a_check_refuses_what_decoding_refuses() -> None:
    frame = _raw(np.zeros((2, 3, 3), dtype=np.uint8))
    frame.data = frame.data[:-1]
    with pytest.raises(FrameEncodingError, match="bytes"):
        check_frame(frame, TrackKind.VIDEO)


# -- encoding a step's output -----------------------------------------------------


def test_an_encoded_video_frame_decodes_back_to_itself() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    frame = encode_frame(image, TrackKind.VIDEO, metadata=b"m")
    assert (frame.encoding, frame.dtype, list(frame.shape)) == (RAW, "uint8", [4, 5, 3])
    assert frame.metadata == b"m"
    np.testing.assert_array_equal(decode_frame(frame, TrackKind.VIDEO).data, image)


def test_encoded_audio_keeps_its_channels_and_scales_floats() -> None:
    stereo = np.array([[0.5, -1.0], [1.0, 0.0]], dtype=np.float32)
    frame = encode_frame(stereo, TrackKind.AUDIO)
    assert list(frame.shape) == [2, 2]
    np.testing.assert_array_equal(
        np.frombuffer(frame.data, dtype="<i2").reshape(2, 2), [[16383, -32767], [32767, 0]]
    )


def test_one_dimensional_audio_is_one_channel() -> None:
    frame = encode_frame(np.array([1, 2, 3], dtype=np.int16), TrackKind.AUDIO)
    assert list(frame.shape) == [1, 3]
    assert not frame.HasField("metadata")


def test_a_video_payload_that_is_not_an_image_cannot_be_encoded() -> None:
    with pytest.raises(FrameEncodingError, match="cannot be shaped"):
        encode_frame(np.zeros((4, 4), dtype=np.uint8), TrackKind.VIDEO)


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (np.zeros((0, 4, 3), dtype=np.uint8), TrackKind.VIDEO),
        (np.zeros((4, 0, 3), dtype=np.uint8), TrackKind.VIDEO),
        (np.zeros((1, 0), dtype=np.int16), TrackKind.AUDIO),
        (np.zeros((0,), dtype=np.float32), TrackKind.AUDIO),
    ],
)
def test_an_empty_frame_cannot_be_encoded(data: np.ndarray, kind: TrackKind) -> None:
    # One frame with no samples is not a frame. An audio track with no samples
    # is still encoded, as a track with no frames: see track_frames below.
    with pytest.raises(FrameEncodingError, match="cannot be shaped"):
        encode_frame(data, kind)


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (np.full((1, 1, 3), "red"), TrackKind.VIDEO),
        (np.array(["loud", "quiet"], dtype=object), TrackKind.AUDIO),
    ],
)
def test_a_payload_that_does_not_convert_is_a_frame_encoding_error(
    data: np.ndarray, kind: TrackKind
) -> None:
    # Not a bare numpy error: the runtime answers a step whose media fails this
    # way as FAILED, and only knows to through FrameEncodingError.
    with pytest.raises(FrameEncodingError, match="cannot be carried"):
        encode_frame(data, kind)


def _track(
    name: str, kind: TrackKind, data: np.ndarray, metadata: bytes | list[bytes] | None = None
) -> TrackData:
    return TrackData(info=TrackInfo(name=name, kind=kind), data=data, metadata=metadata)


def test_a_batched_video_track_becomes_its_frames_each_with_its_metadata() -> None:
    batch = np.stack([np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)])
    bundle = MediaBundle(
        tracks={"main_video": _track("main_video", TrackKind.VIDEO, batch, [b"0", b"1", b"2"])}
    )
    [entry] = track_frames(bundle)
    assert (entry.track, entry.kind) == ("main_video", platform_pb2.TRACK_KIND_VIDEO)
    assert [f.metadata for f in entry.frames] == [b"0", b"1", b"2"]
    assert [decode_frame(f, TrackKind.VIDEO).data[0, 0, 0] for f in entry.frames] == [0, 1, 2]


def test_one_metadata_value_goes_on_every_frame_of_a_batch() -> None:
    batch = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    bundle = MediaBundle(tracks={"v": _track("v", TrackKind.VIDEO, batch, b"same")})
    [entry] = track_frames(bundle)
    assert [f.metadata for f in entry.frames] == [b"same", b"same"]


def test_each_output_track_is_its_own_entry() -> None:
    bundle = MediaBundle(
        tracks={
            "v": _track("v", TrackKind.VIDEO, np.zeros((2, 2, 3), dtype=np.uint8)),
            "a": _track("a", TrackKind.AUDIO, np.zeros((1, 480), dtype=np.int16)),
        }
    )
    entries = track_frames(bundle)
    assert [(e.track, e.kind, len(e.frames)) for e in entries] == [
        ("v", platform_pb2.TRACK_KIND_VIDEO, 1),
        ("a", platform_pb2.TRACK_KIND_AUDIO, 1),
    ]


def test_each_entry_carries_the_rate_its_frames_play_at() -> None:
    bundle = MediaBundle(
        tracks={
            "v": TrackData(
                info=TrackInfo(name="v", kind=TrackKind.VIDEO, rate=30.0),
                data=np.zeros((2, 2, 3), dtype=np.uint8),
            ),
            "a": TrackData(
                info=TrackInfo(name="a", kind=TrackKind.AUDIO, rate=48000.0),
                data=np.zeros((1, 480), dtype=np.int16),
            ),
        }
    )
    assert [e.rate for e in track_frames(bundle, video_fps=24.0)] == [24.0, 48000.0]
    assert [e.rate for e in track_frames(bundle)] == [30.0, 48000.0]


def test_a_step_without_media_has_no_tracks() -> None:
    assert track_frames(None) == []


def test_metadata_that_does_not_cover_the_batch_cannot_be_encoded() -> None:
    batch = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    bundle = MediaBundle(tracks={"v": _track("v", TrackKind.VIDEO, batch, [b"only one"])})
    with pytest.raises(FrameEncodingError, match="1 metadata entries for 2 frames"):
        track_frames(bundle)


@pytest.mark.parametrize(
    "silence",
    [
        np.zeros((1, 0), dtype=np.int16),
        np.zeros((0,), dtype=np.float32),
        np.zeros((2, 0), dtype=np.int16),
    ],
)
def test_an_audio_track_with_no_samples_has_no_frames(silence: np.ndarray) -> None:
    # (1, 0) is how a model says it has no audio this step; examples/echo does.
    # The step keeps its video, and its audio track goes out with no frames.
    bundle = MediaBundle(
        tracks={
            "main_video": _track("main_video", TrackKind.VIDEO, np.zeros((2, 2, 3), np.uint8)),
            "main_audio": TrackData(
                info=TrackInfo(name="main_audio", kind=TrackKind.AUDIO, rate=48000.0),
                data=silence,
            ),
        }
    )
    video, audio = track_frames(bundle)
    assert (video.track, len(video.frames)) == ("main_video", 1)
    assert (audio.track, audio.kind, len(audio.frames), audio.rate) == (
        "main_audio",
        platform_pb2.TRACK_KIND_AUDIO,
        0,
        48000.0,
    )


def test_a_video_track_with_an_empty_frame_cannot_be_encoded() -> None:
    bundle = MediaBundle(
        tracks={"v": _track("v", TrackKind.VIDEO, np.zeros((0, 4, 3), dtype=np.uint8))}
    )
    with pytest.raises(FrameEncodingError, match="cannot be shaped"):
        track_frames(bundle)
