import asyncio
import threading
import weakref
from collections.abc import Callable

import numpy as np
import pytest
import reactor_webrtc as rw

from reactor_runtime.codes import INPUT_DECODE_FAILED
from reactor_runtime.core import DecodedStep
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_runtime.transport.step_frames import (
    Pictures,
    StepFrameGate,
    StepFrameRefusal,
    TrackDecoder,
    TrackEncoder,
    decode_pushed_frames,
    decode_step_frames,
    encoded_step,
    unpublished_track,
)
from reactor_wire.v1 import (
    common_pb2,
    control_pb2,
    data_pb2,
    model_pb2,
    platform_pb2,
    step_pb2,
    track_pb2,
)

WIDTH, HEIGHT = 64, 48
VIDEO_INPUTS = frozenset({"webcam", "other"})

_CODECS = {
    step_pb2.FRAME_ENCODING_H264: rw.VideoCodec.H264,
    step_pb2.FRAME_ENCODING_VP8: rw.VideoCodec.Vp8,
    step_pb2.FRAME_ENCODING_VP9: rw.VideoCodec.Vp9,
    step_pb2.FRAME_ENCODING_AV1: rw.VideoCodec.Av1,
}
_CAPABILITIES = rw.VideoCodecs().capabilities()


def _supported(encoding: step_pb2.FrameEncoding) -> bool:
    codec = _CODECS[encoding]
    return _CAPABILITIES.can_encode(codec) and _CAPABILITIES.can_decode(codec)


ENCODINGS = [
    pytest.param(
        encoding,
        id=step_pb2.FrameEncoding.Name(encoding).removeprefix("FRAME_ENCODING_"),
        marks=pytest.mark.skipif(
            not _supported(encoding), reason="no backend for this codec on this platform"
        ),
    )
    for encoding in _CODECS
]
"""Each encoded encoding, skipped where this platform can not both encode and decode it."""

requires_h264 = pytest.mark.skipif(
    not _supported(step_pb2.FRAME_ENCODING_H264), reason="no H.264 backend on this platform"
)
pytestmark = requires_h264


def _pictures(count: int) -> list[np.ndarray]:
    """A smooth scene sliding one pixel a frame, so the encoder predicts it from the last."""
    rng = np.random.default_rng(7)
    base = np.kron(
        rng.integers(0, 256, (HEIGHT // 8 + 2, WIDTH // 8 + 2, 3), np.uint8),
        np.ones((8, 8, 1), np.uint8),
    )
    return [np.ascontiguousarray(base[i : i + HEIGHT, i : i + WIDTH]) for i in range(count)]


def _encode(
    pictures: list[np.ndarray],
    encoding: step_pb2.FrameEncoding = step_pb2.FRAME_ENCODING_H264,
) -> list[bytes]:
    """Encode *pictures* as one stream: a keyframe, then frames that each reference the last."""
    encoder = rw.VideoCodecs().encoder(_CODECS[encoding], WIDTH, HEIGHT)
    frames = []
    for n, picture in enumerate(pictures):
        frame = encoder.encode(picture, n * 33_333)
        assert frame is not None
        assert frame.key_frame == (n == 0), "the stream holds one keyframe, its first frame"
        frames.append(frame.data)
    return frames


def _reference(
    packets: list[bytes], encoding: step_pb2.FrameEncoding = step_pb2.FRAME_ENCODING_H264
) -> list[np.ndarray]:
    """What a plain decoder makes of *packets*, a stream from its keyframe."""
    decoder = rw.VideoCodecs().decoder(_CODECS[encoding])
    pictures = [decoder.decode(p) for p in packets]
    assert all(p is not None for p in pictures)
    return [p for p in pictures if p is not None]


def _frame(
    data: bytes,
    encoding: step_pb2.FrameEncoding = step_pb2.FRAME_ENCODING_H264,
    *,
    shape: list[int] | None = None,
) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=encoding,
        dtype="uint8",
        shape=[HEIGHT, WIDTH, 3] if shape is None else shape,
        data=data,
    )


def _h264(data: bytes, *, shape: list[int] | None = None) -> step_pb2.Frame:
    return _frame(data, shape=shape)


def _step(*frames: step_pb2.Frame, track: str = "webcam") -> data_pb2.DataClientMessage:
    return data_pb2.DataClientMessage(
        request_id="data_7",
        kind=common_pb2.MessageKind.MESSAGE_KIND_REQUEST,
        request_step=step_pb2.RequestStep(
            media_input=[
                step_pb2.TrackFrames(
                    track=track, kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO, frames=frames
                )
            ]
        ),
    )


def _entries(*entries: tuple[str, list[step_pb2.Frame]]) -> data_pb2.DataClientMessage:
    """A step request with one entry per ``(track, frames)``, in order."""
    message = _step()
    del message.request_step.media_input[:]
    for track, frames in entries:
        message.request_step.media_input.append(
            step_pb2.TrackFrames(
                track=track, kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO, frames=frames
            )
        )
    return message


def _as_bytes(payload: bytes | str) -> bytes:
    assert isinstance(payload, bytes)
    return payload


def _wire(message: data_pb2.DataClientMessage | control_pb2.ControlClientMessage) -> bytes:
    _, frame = V1Codec().encode(message)
    assert isinstance(frame, bytes)
    return frame


def _publish(name: str) -> bytes:
    return _wire(control_pb2.ControlClientMessage(publish_track=track_pb2.PublishTrack(name=name)))


def _unpublish(name: str) -> bytes:
    return _wire(
        control_pb2.ControlClientMessage(unpublish_track=track_pb2.UnpublishTrack(name=name))
    )


def _decode(
    message: data_pb2.DataClientMessage, decoders: dict[str, TrackDecoder] | None = None
) -> Pictures:
    outcome = decode_step_frames(message, {} if decoders is None else decoders, VIDEO_INPUTS)
    assert not isinstance(outcome, StepFrameRefusal), outcome
    return outcome


def _refused(
    message: data_pb2.DataClientMessage, decoders: dict[str, TrackDecoder] | None = None
) -> StepFrameRefusal:
    outcome = decode_step_frames(message, {} if decoders is None else decoders, VIDEO_INPUTS)
    assert isinstance(outcome, StepFrameRefusal)
    return outcome


def _matches(pictures: Pictures, entry: int, expected: list[np.ndarray]) -> bool:
    return all(np.array_equal(pictures[entry, i], e) for i, e in enumerate(expected))


# ── decode_step_frames ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_each_encoding_decodes_to_its_pictures_and_the_request_is_left_as_it_is(
    encoding: step_pb2.FrameEncoding,
) -> None:
    pictures = _pictures(4)
    packets = _encode(pictures, encoding)
    message = _step(*(_frame(p, encoding) for p in packets))
    sent = message.SerializeToString()

    decoded = _decode(message)

    assert sorted(decoded) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    for position, expected in enumerate(_reference(packets, encoding)):
        picture = decoded[0, position]
        assert (picture.dtype, picture.shape) == (np.uint8, (HEIGHT, WIDTH, 3))
        assert np.array_equal(picture, expected)
        # Lossy, but the picture the client encoded.
        assert np.abs(picture.astype(int) - pictures[position].astype(int)).mean() < 24
    assert message.SerializeToString() == sent


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_a_stream_spans_requests_with_no_keyframe_in_the_second(
    encoding: step_pb2.FrameEncoding,
) -> None:
    packets = _encode(_pictures(8), encoding)
    decoders: dict[str, TrackDecoder] = {}

    _decode(_step(*(_frame(p, encoding) for p in packets[:4])), decoders)
    (decoder,) = decoders.values()
    # The second request carries only inter frames, which reference the last
    # frame of the first.
    pictures = _decode(_step(*(_frame(p, encoding) for p in packets[4:])), decoders)

    assert decoders == {"webcam": decoder}
    assert _matches(pictures, 0, _reference(packets, encoding)[4:])


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_a_new_connection_whose_first_frame_is_not_a_keyframe_is_refused(
    encoding: step_pb2.FrameEncoding,
) -> None:
    packets = _encode(_pictures(4), encoding)
    # A connection's gate starts with no decoder, as this empty mapping does.
    decoders: dict[str, TrackDecoder] = {}

    refusal = _refused(_step(*(_frame(p, encoding) for p in packets[1:])), decoders)

    assert refusal.entry == "media_input[0].frames[0]"
    assert "must start at a keyframe" in refusal.reason
    assert refusal.code == INPUT_DECODE_FAILED
    # The track's next request decodes once it starts at a keyframe.
    assert _matches(
        _decode(_step(*(_frame(p, encoding) for p in packets)), decoders),
        0,
        _reference(packets, encoding),
    )


def test_a_request_refused_after_its_frames_decode_still_moves_the_stream_on() -> None:
    packets = _encode(_pictures(8))
    decoders: dict[str, TrackDecoder] = {}
    _decode(_step(*(_h264(p) for p in packets[:4])), decoders)

    # Its frames decode, but the second is not the shape it declares.
    bad = _step(_h264(packets[4]), _h264(packets[5], shape=[HEIGHT * 2, WIDTH, 3]))
    refusal = _refused(bad, decoders)
    assert refusal.entry == "media_input[0].frames[1]"
    # The stream is whole, so the client is not told to start over.
    assert refusal.code is None
    pictures = _decode(_step(*(_h264(p) for p in packets[6:])), decoders)

    assert _matches(pictures, 0, _reference(packets)[6:])


def test_a_track_whose_frames_do_not_decode_starts_over_and_the_others_go_on() -> None:
    webcam, other = _encode(_pictures(8)), _encode(_pictures(8))
    decoders: dict[str, TrackDecoder] = {}
    _decode(_entries(("webcam", [_h264(p) for p in webcam[:4]])), decoders)

    # "other" opens on an inter frame, which its new decoder cannot start at.
    refusal = _refused(
        _entries(
            ("webcam", [_h264(p) for p in webcam[4:6]]),
            ("other", [_h264(p) for p in other[1:3]]),
        ),
        decoders,
    )
    assert refusal.entry == "media_input[1].frames[0]"
    assert "must start at a keyframe" in refusal.reason
    assert refusal.code == INPUT_DECODE_FAILED
    assert set(decoders) == {"webcam"}

    pictures = _decode(
        _entries(
            ("webcam", [_h264(p) for p in webcam[6:]]),
            ("other", [_h264(p) for p in other[:2]]),
        ),
        decoders,
    )
    assert _matches(pictures, 0, _reference(webcam)[6:])
    assert _matches(pictures, 1, _reference(other)[:2])


def test_a_decoder_decodes_from_the_next_keyframe_after_a_corrupt_frame() -> None:
    packets = _encode(_pictures(4))
    decoders: dict[str, TrackDecoder] = {}
    # A keyframe cut short: still one access unit, but its slice data is gone.
    refusal = _refused(_step(_h264(packets[0][: len(packets[0]) // 3])), decoders)
    assert refusal.entry == "media_input[0].frames[0]"

    pictures = _decode(_step(*(_h264(p) for p in packets)), decoders)

    assert _matches(pictures, 0, _reference(packets))


def test_raw_frames_beside_encoded_ones_are_not_decoded() -> None:
    (packet,) = _encode(_pictures(1))
    raw = step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_RAW, dtype="uint8", shape=[2, 2, 3], data=bytes(12)
    )
    message = _step(_h264(packet))
    message.request_step.media_input.append(
        step_pb2.TrackFrames(
            track="other", kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO, frames=[raw]
        )
    )

    assert list(_decode(message)) == [(0, 0)]


def test_each_track_of_a_request_decodes_on_its_own_decoder() -> None:
    webcam, other = _encode(_pictures(4)), _encode(_pictures(8)[4:])
    decoders: dict[str, TrackDecoder] = {}

    pictures = _decode(
        _entries(("webcam", [_h264(p) for p in webcam]), ("other", [_h264(p) for p in other])),
        decoders,
    )

    assert set(decoders) == {"webcam", "other"}
    assert decoders["webcam"] is not decoders["other"]
    assert _matches(pictures, 0, _reference(webcam))
    assert _matches(pictures, 1, _reference(other))


def test_entries_naming_the_same_track_continue_one_stream() -> None:
    packets = _encode(_pictures(8))
    decoders: dict[str, TrackDecoder] = {}

    pictures = _decode(
        _entries(
            ("webcam", [_h264(p) for p in packets[:4]]),
            ("webcam", [_h264(p) for p in packets[4:]]),
        ),
        decoders,
    )

    assert len(decoders) == 1
    expected = _reference(packets)
    assert _matches(pictures, 0, expected[:4])
    assert _matches(pictures, 1, expected[4:])


@pytest.mark.skipif(
    not _supported(step_pb2.FRAME_ENCODING_VP8), reason="no VP8 backend on this platform"
)
def test_a_track_whose_frames_change_encoding_gets_a_new_decoder_from_a_keyframe() -> None:
    decoders: dict[str, TrackDecoder] = {}
    _decode(_step(*(_h264(p) for p in _encode(_pictures(2)))), decoders)
    first = decoders["webcam"]

    vp8 = _encode(_pictures(2), step_pb2.FRAME_ENCODING_VP8)
    pictures = _decode(_step(*(_frame(p, step_pb2.FRAME_ENCODING_VP8) for p in vp8)), decoders)

    assert decoders["webcam"] is not first
    assert decoders["webcam"].encoding == step_pb2.FRAME_ENCODING_VP8
    assert _matches(pictures, 0, _reference(vp8, step_pb2.FRAME_ENCODING_VP8))


@pytest.mark.skipif(
    not _supported(step_pb2.FRAME_ENCODING_VP8), reason="no VP8 backend on this platform"
)
def test_two_encodings_on_one_track_in_a_request_are_refused() -> None:
    (h264,) = _encode(_pictures(1))
    (vp8,) = _encode(_pictures(1), step_pb2.FRAME_ENCODING_VP8)

    refusal = _refused(
        _entries(
            ("webcam", [_h264(h264)]),
            ("webcam", [_frame(vp8, step_pb2.FRAME_ENCODING_VP8)]),
        )
    )

    assert refusal.entry == "media_input[1].frames[0]"
    assert "one frame encoding" in refusal.reason


def test_the_first_entry_at_fault_is_the_one_refused() -> None:
    packets = _encode(_pictures(4))
    message = _entries(
        ("webcam", [_h264(p) for p in packets]), ("other", [_h264(p) for p in packets[1:]])
    )
    message.request_step.media_input[0].frames[0].shape[:] = [HEIGHT * 2, WIDTH, 3]

    refusal = _refused(message)

    assert refusal.entry == "media_input[0].frames[0]"


def test_a_track_after_one_at_fault_still_decodes_on_its_decoder() -> None:
    packets = _encode(_pictures(4))
    # The first track starts past its keyframe; the second is whole.
    message = _entries(
        ("webcam", [_h264(p) for p in packets[1:]]), ("other", [_h264(p) for p in packets])
    )
    decoders: dict[str, TrackDecoder] = {}

    outcome = decode_step_frames(message, decoders, VIDEO_INPUTS)

    assert isinstance(outcome, StepFrameRefusal)
    assert outcome.entry == "media_input[0].frames[0]"
    assert set(decoders) == {"other"}


def test_encoded_frames_for_a_track_that_is_not_an_input_video_track_open_no_decoder() -> None:
    packets = _encode(_pictures(4))
    decoders: dict[str, TrackDecoder] = {}

    outcome = decode_step_frames(
        _step(*(_h264(p) for p in packets), track="mic"), decoders, VIDEO_INPUTS
    )

    # No code: the track has no stream for a keyframe to restart.
    assert outcome == StepFrameRefusal(
        0,
        "media_input[0] names 'mic', which is not an input video track of the model",
    )
    assert decoders == {}


def test_an_unknown_encoding_is_refused_naming_its_frame() -> None:
    (packet,) = _encode(_pictures(1))
    unknown = step_pb2.Frame(dtype="uint8", shape=[HEIGHT, WIDTH, 3], data=b"x")
    # Proto3 enums are open: a peer may send a value this build has no name for.
    setattr(unknown, "encoding", 99)  # noqa: B010
    refusal = _refused(_step(_h264(packet), unknown))

    assert refusal.entry == "media_input[0].frames[1]"
    assert "99 is not supported" in refusal.reason
    # Restarting at a keyframe would not help: the encoding is not decoded here.
    assert refusal.code is None


def test_a_frame_holding_a_p_frame_then_an_idr_is_refused() -> None:
    # A decoder would take the two as one frame and keep only one picture.
    packets = _encode(_pictures(4))
    keyframe = _encode(_pictures(1))[0]
    refusal = _refused(_step(_h264(packets[1] + keyframe)))

    assert refusal.entry == "media_input[0].frames[0]"
    assert "more than one access unit" in refusal.reason


def test_a_frame_holding_two_pictures_is_refused() -> None:
    packets = _encode(_pictures(4))
    refusal = _refused(_step(_h264(packets[1] + packets[2])))

    assert refusal.entry == "media_input[0].frames[0]"
    assert "more than one picture" in refusal.reason


def test_a_frame_with_no_picture_is_refused() -> None:
    from reactor_runtime.transport.step_frames import _nal_units

    (first, *_) = _encode(_pictures(4))
    parameter_sets = b"".join(
        b"\x00\x00\x00\x01" + unit for unit in _nal_units(first) if unit[0] & 0x1F in (7, 8)
    )
    refusal = _refused(_step(_h264(parameter_sets)))

    assert "holds no picture" in refusal.reason


def test_a_frame_that_decodes_to_another_size_is_refused() -> None:
    (packet,) = _encode(_pictures(1))
    refusal = _refused(_step(_h264(packet, shape=[720, 1280, 3])))

    assert refusal.entry == "media_input[0].frames[0]"
    assert f"[{HEIGHT}, {WIDTH}, 3]" in refusal.reason


def test_an_empty_encoded_frame_is_refused() -> None:
    refusal = _refused(_step(_h264(b"")))

    assert refusal.entry == "media_input[0].frames[0]"
    assert "no data" in refusal.reason


# ── encoded_step ───────────────────────────────────────────────────────────────


def test_only_a_v1_data_step_with_an_encoded_frame_is_picked_out() -> None:
    (packet,) = _encode(_pictures(1))
    encoded = _wire(_step(_h264(packet)))
    raw = _wire(
        _step(
            step_pb2.Frame(
                encoding=step_pb2.FRAME_ENCODING_RAW, dtype="uint8", shape=[1, 1, 3], data=bytes(3)
            )
        )
    )
    command = _wire(
        data_pb2.DataClientMessage(request_id="data_1", command=model_pb2.Command(type="ping"))
    )

    assert encoded_step(encoded, ProtocolVersion.V1, Channel.DATA) is not None
    assert encoded_step(raw, ProtocolVersion.V1, Channel.DATA) is None
    assert encoded_step(command, ProtocolVersion.V1, Channel.DATA) is None
    assert encoded_step(encoded, ProtocolVersion.V1, Channel.CONTROL) is None
    assert encoded_step(encoded, ProtocolVersion.V0, Channel.DATA) is None
    assert encoded_step(b"\xff not protobuf", ProtocolVersion.V1, Channel.DATA) is None


# ── StepFrameGate ──────────────────────────────────────────────────────────────


class _Recorder:
    def __init__(self) -> None:
        self.delivered: list[bytes] = []
        self.steps: list[DecodedStep] = []
        self.order: list[str] = []
        self.replies: list[bytes] = []

    def gate(self) -> StepFrameGate:
        def deliver(payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
            self.delivered.append(_as_bytes(payload))
            self.order.append("message")

        def deliver_step(step: DecodedStep, version: ProtocolVersion) -> None:
            assert version is ProtocolVersion.V1
            self.steps.append(step)
            self.order.append("step")

        return StepFrameGate(
            deliver,
            deliver_step,
            lambda payload: self.replies.append(_as_bytes(payload)),
            VIDEO_INPUTS,
        )


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0.01)


async def _until(predicate: Callable[[], bool]) -> None:
    """Wait for *predicate*, which a decode on a worker thread makes true."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within 10 s")


async def test_a_plain_message_is_delivered_at_once() -> None:
    rec = _Recorder()
    rec.gate().submit(b"anything", ProtocolVersion.V1, Channel.DATA)

    assert rec.delivered == [b"anything"]


async def test_an_encoded_step_goes_up_with_its_pictures_and_nothing_overtakes_it() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    request = _wire(_step(*(_h264(p) for p in packets)))
    command = _wire(
        data_pb2.DataClientMessage(request_id="data_8", command=model_pb2.Command(type="ping"))
    )

    gate.submit(request, ProtocolVersion.V1, Channel.DATA)
    gate.submit(command, ProtocolVersion.V1, Channel.DATA)
    assert rec.order == []  # the command waits behind the decode
    await _until(lambda: len(rec.order) == 2)

    assert rec.order == ["step", "message"]
    (step,) = rec.steps
    # The request goes up as it was sent, the pictures beside it.
    assert step.payload == request
    assert sorted(step.pictures) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert rec.delivered == [command]
    assert rec.replies == []


async def test_a_step_that_does_not_decode_is_refused_and_not_delivered() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    command = _wire(
        data_pb2.DataClientMessage(request_id="data_9", command=model_pb2.Command(type="ping"))
    )

    gate.submit(_wire(_step(*(_h264(p) for p in packets[1:]))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(command, ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.delivered) == 1)

    assert rec.delivered == [command]
    assert rec.steps == []
    (reply,) = rec.replies
    answer = data_pb2.DataServerMessage.FromString(reply)
    assert answer.request_id == "data_7"
    assert answer.kind == common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
    assert answer.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert answer.step_result.entry.startswith("media_input[0]")
    assert answer.step_result.code == INPUT_DECODE_FAILED


async def test_a_step_refused_with_its_stream_whole_carries_no_code() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(2))
    # Both frames decode, but the second is not the shape it declares.
    bad = _step(_h264(packets[0]), _h264(packets[1], shape=[HEIGHT * 2, WIDTH, 3]))

    gate.submit(_wire(bad), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.replies) == 1)

    (reply,) = rec.replies
    answer = data_pb2.DataServerMessage.FromString(reply)
    assert answer.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert answer.step_result.entry == "media_input[0].frames[1]"
    assert not answer.step_result.HasField("code")


async def test_close_drops_what_is_still_queued() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(b"after", ProtocolVersion.V1, Channel.DATA)

    gate.close()
    await _settle()

    assert rec.order == []
    assert rec.replies == []


# ── decoder lifecycle ──────────────────────────────────────────────────────────


def test_only_a_v1_control_unpublish_names_an_unpublished_track() -> None:
    assert unpublished_track(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL) == "webcam"
    assert unpublished_track(_publish("webcam"), ProtocolVersion.V1, Channel.CONTROL) is None
    assert unpublished_track(_unpublish("webcam"), ProtocolVersion.V1, Channel.DATA) is None
    assert unpublished_track(_unpublish("webcam"), ProtocolVersion.V0, Channel.CONTROL) is None
    ping = _wire(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    assert unpublished_track(ping, ProtocolVersion.V1, Channel.CONTROL) is None
    assert unpublished_track(b"\xff not protobuf", ProtocolVersion.V1, Channel.CONTROL) is None


async def test_publishing_a_track_opens_no_decoder() -> None:
    rec = _Recorder()
    gate = rec.gate()

    for name in ("webcam", "mic", "screen"):
        gate.submit(_publish(name), ProtocolVersion.V1, Channel.CONTROL)

    assert gate.decoders == 0
    # The runner still sees each, to arbitrate the track.
    assert rec.delivered == [_publish("webcam"), _publish("mic"), _publish("screen")]


async def test_unpublishing_a_track_releases_its_decoder() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))

    gate.submit(_publish("webcam"), ProtocolVersion.V1, Channel.CONTROL)
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 1)
    assert gate.decoders == 1
    gate.submit(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL)

    assert gate.decoders == 0
    assert rec.delivered == [_publish("webcam"), _unpublish("webcam")]


async def test_h264_frames_for_a_track_that_is_not_an_input_video_track_are_refused() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))

    gate.submit(
        _wire(_step(*(_h264(p) for p in packets), track="mic")),
        ProtocolVersion.V1,
        Channel.DATA,
    )
    await _until(lambda: len(rec.replies) == 1)

    assert rec.steps == []
    assert gate.decoders == 0


async def test_a_track_never_published_gets_its_decoder_with_its_first_frames() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(8))

    gate.submit(_wire(_step(*(_h264(p) for p in packets[:4]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 1)
    assert gate.decoders == 1
    gate.submit(_wire(_step(*(_h264(p) for p in packets[4:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 2)

    assert gate.decoders == 1


@pytest.mark.parametrize("release", ["unpublish", "close"])
async def test_a_released_decoder_is_freed_off_the_event_loop(release: str) -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 1)
    freed: list[threading.Thread] = []
    weakref.finalize(
        next(iter(gate._decoders.values())), lambda: freed.append(threading.current_thread())
    )

    if release == "unpublish":
        gate.submit(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL)
    else:
        gate.close()
    await _until(lambda: len(freed) == 1)

    assert freed[0] is not threading.current_thread()


async def test_an_unpublish_queued_behind_a_decode_releases_the_decoder_after_it() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))

    gate.submit(_publish("webcam"), ProtocolVersion.V1, Channel.CONTROL)
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL)
    assert rec.order == ["message"]
    await _until(lambda: len(rec.order) == 3)

    assert rec.order == ["message", "step", "message"]
    assert gate.decoders == 0


async def test_close_releases_every_decoder_even_one_a_running_decode_opens() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)

    gate.close()
    await _settle()

    assert gate.decoders == 0


async def test_a_new_connection_starts_its_tracks_over_at_a_keyframe() -> None:
    packets = _encode(_pictures(8))
    first, second = _Recorder(), _Recorder()
    gate = first.gate()
    gate.submit(_wire(_step(*(_h264(p) for p in packets[:4]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(first.steps) == 1)
    gate.close()

    # The same client reconnects and goes on with the stream it was sending.
    again = second.gate()
    again.submit(_wire(_step(*(_h264(p) for p in packets[4:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(second.replies) == 1)

    assert second.steps == []
    answer = data_pb2.DataServerMessage.FromString(second.replies[0])
    assert answer.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert "must start at a keyframe" in answer.step_result.error


async def test_a_track_that_loses_its_stream_releases_its_decoder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(8))
    gate.submit(_wire(_step(*(_h264(p) for p in packets[:4]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 1)
    assert gate.decoders == 1

    empty = _h264(packets[4])
    empty.data = b""
    with caplog.at_level("WARNING", logger="reactor_runtime.transport.step_frames"):
        gate.submit(_wire(_step(empty)), ProtocolVersion.V1, Channel.DATA)
        await _until(lambda: len(rec.replies) == 1)

    assert gate.decoders == 0
    # The operator's log names the codec whose stream broke.
    assert "encoding=FRAME_ENCODING_H264" in caplog.text


async def test_an_unknown_encoding_keeps_the_tracks_decoder_and_stream() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(8))
    gate.submit(_wire(_step(*(_h264(p) for p in packets[:4]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 1)

    unknown = _h264(packets[4])
    # An encoding this build has no name for: the frame never reaches the decoder.
    setattr(unknown, "encoding", 99)  # noqa: B010
    gate.submit(_wire(_step(unknown)), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.replies) == 1)
    assert gate.decoders == 1

    gate.submit(_wire(_step(*(_h264(p) for p in packets[4:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 2)


# ── pushed frames ──────────────────────────────────────────────────────────────


def _keyed(packets: list[bytes], keyframes: set[int]) -> list[step_pb2.Frame]:
    """H.264 frames of *packets*, the ones at *keyframes* marked as keyframes."""
    return [
        step_pb2.Frame(
            encoding=step_pb2.FRAME_ENCODING_H264,
            dtype="uint8",
            shape=[HEIGHT, WIDTH, 3],
            data=packet,
            keyframe=n in keyframes,
        )
        for n, packet in enumerate(packets)
    ]


def _encode_keyed(pictures: list[np.ndarray], keyframes: set[int]) -> list[step_pb2.Frame]:
    """Encode *pictures* as one H.264 stream with a keyframe at each of *keyframes*."""
    encoder = rw.VideoCodecs().encoder(rw.VideoCodec.H264, WIDTH, HEIGHT)
    frames = []
    for n, picture in enumerate(pictures):
        if n in keyframes:
            encoder.request_key_frame()
        frame = encoder.encode(picture, n * 33_333)
        assert frame is not None
        assert frame.key_frame == (n in keyframes)
        frames.append(frame.data)
    return _keyed(frames, keyframes)


def _push(*entries: tuple[str, list[step_pb2.Frame]]) -> data_pb2.DataClientMessage:
    """A push with one entry per ``(track, frames)``, in order."""
    return data_pb2.DataClientMessage(
        kind=common_pb2.MessageKind.MESSAGE_KIND_NOTIFICATION,
        push_frames=step_pb2.PushFrames(
            tracks=[
                step_pb2.TrackFrames(
                    track=track, kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO, frames=frames
                )
                for track, frames in entries
            ]
        ),
    )


def test_a_push_with_an_encoded_frame_is_picked_out() -> None:
    (packet,) = _encode(_pictures(1))
    raw = step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_RAW, dtype="uint8", shape=[1, 1, 3], data=bytes(3)
    )

    assert encoded_step(_wire(_push(("webcam", [_h264(packet)]))), ProtocolVersion.V1, Channel.DATA)
    assert encoded_step(_wire(_push(("webcam", [raw]))), ProtocolVersion.V1, Channel.DATA) is None


def test_every_pushed_frame_decodes_even_ones_the_model_will_drop_so_the_stream_goes_on() -> None:
    # The model's buffer may keep only the newest of these; the decoder still
    # sees every one, so the next push's inter frames decode against them.
    packets = _encode(_pictures(8))
    decoders: dict[str, TrackDecoder] = {}
    lost: set[str] = set()

    first = decode_pushed_frames(
        _push(("webcam", _keyed(packets[:6], {0}))), decoders, VIDEO_INPUTS, lost
    )
    second = decode_pushed_frames(
        _push(("webcam", _keyed(packets[6:], set()))), decoders, VIDEO_INPUTS, lost
    )

    assert first.failed == second.failed == {}
    assert sorted(first.pictures) == [(0, i) for i in range(6)]
    reference = _reference(packets)
    assert _matches(second.pictures, 0, reference[6:])
    assert lost == set()


def test_pushes_and_step_requests_carry_one_stream_per_track() -> None:
    packets = _encode(_pictures(9))
    decoders: dict[str, TrackDecoder] = {}
    lost: set[str] = set()
    reference = _reference(packets)

    pushed = decode_pushed_frames(
        _push(("webcam", _keyed(packets[:3], {0}))), decoders, VIDEO_INPUTS, lost
    )
    stepped = _decode(_step(*(_h264(p) for p in packets[3:6])), decoders)
    again = decode_pushed_frames(
        _push(("webcam", _keyed(packets[6:], set()))), decoders, VIDEO_INPUTS, lost
    )

    assert _matches(pushed.pictures, 0, reference[:3])
    assert _matches(stepped, 0, reference[3:6])
    assert _matches(again.pictures, 0, reference[6:])


def test_a_frame_that_does_not_decode_is_reported_once_and_the_track_resumes_at_a_keyframe() -> (
    None
):
    frames = _encode_keyed(_pictures(10), {0, 6})
    decoders: dict[str, TrackDecoder] = {}
    lost: set[str] = set()
    decode_pushed_frames(_push(("webcam", frames[:2])), decoders, VIDEO_INPUTS, lost)

    corrupt = step_pb2.Frame()
    corrupt.CopyFrom(frames[2])
    corrupt.data = b"\x00\x00\x00\x01\x41" + bytes(8)
    broken = decode_pushed_frames(
        _push(("webcam", [corrupt, *frames[3:5]])), decoders, VIDEO_INPUTS, lost
    )
    # Inter frames after the break are dropped without another notice.
    waiting = decode_pushed_frames(_push(("webcam", [frames[5]])), decoders, VIDEO_INPUTS, lost)
    resumed = decode_pushed_frames(_push(("webcam", frames[6:])), decoders, VIDEO_INPUTS, lost)

    assert list(broken.failed) == ["webcam"]
    assert broken.failed["webcam"].startswith("tracks[0].frames[0]")
    assert broken.pictures == {}
    assert (waiting.failed, waiting.pictures) == ({}, {})
    assert resumed.failed == {}
    assert sorted(resumed.pictures) == [(0, i) for i in range(4)]
    assert lost == set()
    assert "webcam" in decoders


def test_a_new_connections_first_pushed_frame_must_be_a_keyframe() -> None:
    packets = _encode(_pictures(4))
    decoders: dict[str, TrackDecoder] = {}
    lost: set[str] = set()

    pushed = decode_pushed_frames(
        _push(("webcam", _keyed(packets[1:], set()))), decoders, VIDEO_INPUTS, lost
    )

    assert "must start at a keyframe" in pushed.failed["webcam"]
    assert pushed.pictures == {}
    assert (decoders, lost) == ({}, {"webcam"})


def test_pushed_frames_for_a_track_that_is_not_an_input_video_track_are_not_decoded() -> None:
    packets = _encode(_pictures(2))
    decoders: dict[str, TrackDecoder] = {}

    pushed = decode_pushed_frames(
        _push(("mic", _keyed(packets, {0})), ("webcam", _keyed(packets, {0}))),
        decoders,
        VIDEO_INPUTS,
        set(),
    )

    assert pushed.failed == {}
    assert sorted(pushed.pictures) == [(1, 0), (1, 1)]
    assert list(decoders) == ["webcam"]


async def test_a_push_goes_up_with_its_pictures_in_order_with_commands_and_steps() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(6))
    push = _wire(_push(("webcam", _keyed(packets[:3], {0}))))
    command = _wire(
        data_pb2.DataClientMessage(request_id="data_8", command=model_pb2.Command(type="ping"))
    )
    request = _wire(_step(*(_h264(p) for p in packets[3:])))

    gate.submit(push, ProtocolVersion.V1, Channel.DATA)
    gate.submit(command, ProtocolVersion.V1, Channel.DATA)
    gate.submit(request, ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.order) == 3)

    assert rec.order == ["step", "message", "step"]
    pushed, stepped = rec.steps
    assert pushed.payload == push
    assert sorted(pushed.pictures) == [(0, 0), (0, 1), (0, 2)]
    assert sorted(stepped.pictures) == [(0, 0), (0, 1), (0, 2)]
    assert rec.replies == []


async def test_a_push_that_does_not_decode_is_delivered_and_the_client_is_told_once() -> None:
    rec = _Recorder()
    gate = rec.gate()
    frames = _encode_keyed(_pictures(6), {0, 4})

    # A first push that does not start at a keyframe, then more inter frames.
    gate.submit(_wire(_push(("webcam", frames[1:3]))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_wire(_push(("webcam", frames[3:4]))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_wire(_push(("webcam", frames[4:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 3)

    (notice,) = rec.replies
    answer = data_pb2.DataServerMessage.FromString(notice)
    assert answer.kind == common_pb2.MessageKind.MESSAGE_KIND_NOTIFICATION
    assert answer.WhichOneof("payload") == "error"
    assert (answer.error.code, answer.error.track) == ("input_decode_failed", "webcam")
    # Never a refusal: every push goes up, the dropped frames without pictures.
    assert [sorted(step.pictures) for step in rec.steps] == [[], [], [(0, 0), (0, 1)]]
    assert gate.decoders == 1


async def test_a_step_that_breaks_a_stream_is_not_reported_again_by_the_push_behind_it() -> None:
    rec = _Recorder()
    gate = rec.gate()
    frames = _encode_keyed(_pictures(6), {0, 4})

    # The step's first frame is not a keyframe; the client had already
    # pipelined a push of more inter frames before it heard.
    gate.submit(_wire(_step(*frames[1:3])), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_wire(_push(("webcam", frames[3:4]))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_wire(_push(("webcam", frames[4:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 2)

    # One report of the break, the step's refusal; the push waits for the keyframe.
    (refusal,) = rec.replies
    assert data_pb2.DataServerMessage.FromString(refusal).step_result.code == (
        "input_decode_failed"
    )
    assert [sorted(step.pictures) for step in rec.steps] == [[], [(0, 0), (0, 1)]]


async def test_a_step_keyframe_restarts_a_stream_a_push_broke() -> None:
    rec = _Recorder()
    gate = rec.gate()
    frames = _encode_keyed(_pictures(6), {0})

    gate.submit(_wire(_push(("webcam", frames[1:2]))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(_wire(_step(*frames[:3])), ProtocolVersion.V1, Channel.DATA)
    # The step restarted the stream, so the push's inter frames decode on it.
    gate.submit(_wire(_push(("webcam", frames[3:]))), ProtocolVersion.V1, Channel.DATA)
    await _until(lambda: len(rec.steps) == 3)

    assert len(rec.replies) == 1  # the push's notice, nothing after it
    assert [sorted(step.pictures) for step in rec.steps] == [
        [],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (0, 2)],
    ]


async def test_unpublishing_a_track_forgets_that_its_pushed_stream_broke() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    gate.submit(
        _wire(_push(("webcam", _keyed(packets[1:2], set())))), ProtocolVersion.V1, Channel.DATA
    )
    await _until(lambda: len(rec.steps) == 1)

    gate.submit(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL)
    # Published again, its first frame breaks a new stream, which is told again.
    gate.submit(
        _wire(_push(("webcam", _keyed(packets[2:3], set())))), ProtocolVersion.V1, Channel.DATA
    )
    await _until(lambda: len(rec.steps) == 2)

    assert len(rec.replies) == 2


# -- the output side: one encoder per track and connection ---------------------


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_a_track_encoder_is_one_stream_across_calls(encoding: step_pb2.FrameEncoding) -> None:
    pictures = _pictures(4)
    encoder = TrackEncoder(encoding)
    first = encoder.encode(pictures[:1], 30.0)
    rest = encoder.encode(pictures[1:], 30.0)
    assert first is not None
    assert rest is not None

    assert [f.key_frame for f in [*first, *rest]] == [True, False, False, False]
    decoded = _reference([f.data for f in [*first, *rest]], encoding)
    assert [p.shape for p in decoded] == [(HEIGHT, WIDTH, 3)] * 4


def test_a_track_encoder_starts_at_a_keyframe_when_asked() -> None:
    pictures = _pictures(3)
    encoder = TrackEncoder(step_pb2.FRAME_ENCODING_H264)
    assert encoder.encode(pictures[:1], None) is not None

    encoder.request_keyframe()
    again = encoder.encode(pictures[1:], None)

    assert again is not None
    assert [f.key_frame for f in again] == [True, False]


def test_frames_a_track_encoder_cannot_take_start_the_stream_over() -> None:
    encoder = TrackEncoder(step_pb2.FRAME_ENCODING_H264)
    assert encoder.encode(_pictures(1), 30.0) is not None

    # An odd side is a size no encoder takes.
    assert encoder.encode([np.zeros((HEIGHT + 1, WIDTH, 3), np.uint8)], 30.0) is None
    after = encoder.encode(_pictures(1), 30.0)

    assert after is not None
    assert [f.key_frame for f in after] == [True]


def test_a_track_encoder_for_an_encoding_with_no_codec_encodes_nothing() -> None:
    encoder = TrackEncoder(step_pb2.FRAME_ENCODING_RAW)
    assert encoder.encode(_pictures(1), 30.0) is None
