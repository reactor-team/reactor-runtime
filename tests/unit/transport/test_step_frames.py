import asyncio
from collections.abc import Callable

import av
import numpy as np

from reactor_runtime.core import DecodedStep
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_runtime.transport.step_frames import (
    Pictures,
    StepFrameGate,
    StepFrameRefusal,
    TrackDecoder,
    decode_step_frames,
    encoded_step,
    track_event,
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


def _pictures(count: int) -> list[np.ndarray]:
    """A smooth scene sliding one pixel a frame, so the encoder predicts it from the last."""
    rng = np.random.default_rng(7)
    base = np.kron(
        rng.integers(0, 256, (HEIGHT // 8 + 2, WIDTH // 8 + 2, 3), np.uint8),
        np.ones((8, 8, 1), np.uint8),
    )
    return [np.ascontiguousarray(base[i : i + HEIGHT, i : i + WIDTH]) for i in range(count)]


def _encode(pictures: list[np.ndarray], **options: str) -> list[bytes]:
    """Encode *pictures* as an H.264 stream; one access unit per picture."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    # One keyframe per four frames, never one on a scene cut, so a stream past
    # its first frame starts on a predicted frame.
    codec.options = {
        "g": "4",
        "bf": "0",
        "tune": "zerolatency",
        "x264-params": "scenecut=0",
        **options,
    }
    packets = []
    for picture in pictures:
        packets += codec.encode(av.VideoFrame.from_ndarray(picture, format="rgb24"))
    packets += codec.encode(None)
    return [bytes(p) for p in packets]


def _reference(packets: list[bytes]) -> list[np.ndarray]:
    """What a plain decoder makes of *packets*, in presentation order."""
    decoder = av.CodecContext.create("h264", "r")
    frames = [f for p in packets for f in decoder.decode(av.Packet(p))] + decoder.decode(None)
    return [f.to_ndarray(format="rgb24") for f in frames]


def _h264(
    data: bytes,
    *,
    shape: list[int] | None = None,
    metadata: bytes | None = None,
    capture_time_us: int | None = None,
) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_H264,
        dtype="uint8",
        shape=[HEIGHT, WIDTH, 3] if shape is None else shape,
        data=data,
        metadata=metadata,
        capture_time_us=capture_time_us,
    )


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


def _refused(message: data_pb2.DataClientMessage) -> StepFrameRefusal:
    outcome = decode_step_frames(message, {}, VIDEO_INPUTS)
    assert isinstance(outcome, StepFrameRefusal)
    return outcome


# ── decode_step_frames ─────────────────────────────────────────────────────────


def test_h264_frames_decode_to_their_pictures_and_the_request_is_left_as_it_is() -> None:
    packets = _encode(_pictures(4))
    message = _step(*(_h264(p) for p in packets))
    sent = message.SerializeToString()

    pictures = _decode(message)

    assert sorted(pictures) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    for position, expected in enumerate(_reference(packets)):
        picture = pictures[0, position]
        assert (picture.dtype, picture.shape) == (np.uint8, (HEIGHT, WIDTH, 3))
        assert np.array_equal(picture, expected)
    assert message.SerializeToString() == sent


def test_a_stream_with_b_frames_decodes_one_picture_per_frame() -> None:
    packets = _encode(_pictures(8), bf="2", tune="film")

    pictures = _decode(_step(*(_h264(p) for p in packets)))

    assert sorted(pictures) == [(0, p) for p in range(len(packets))]


def test_raw_frames_beside_h264_ones_are_not_decoded() -> None:
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


def test_a_track_without_a_decoder_gets_one_and_keeps_it_for_the_next_request() -> None:
    packets = _encode(_pictures(8))
    decoders: dict[str, TrackDecoder] = {}

    _decode(_step(*(_h264(p) for p in packets[:4])), decoders)
    (decoder,) = decoders.values()
    # The second request starts at the stream's next keyframe, after the
    # decoder already decoded and flushed the first.
    pictures = _decode(_step(*(_h264(p) for p in packets[4:])), decoders)

    assert decoders == {"webcam": decoder}
    expected = _reference(packets[4:])
    assert all(np.array_equal(pictures[0, i], expected[i]) for i in range(4))


def test_a_decoder_decodes_the_next_request_cleanly_after_a_corrupt_one() -> None:
    packets = _encode(_pictures(4))
    decoders = {"webcam": TrackDecoder()}
    # Starts at a keyframe and passes the checks, but its second frame is a
    # picture's slices cut short, which the decoder conceals.
    corrupt = _step(_h264(packets[0]), _h264(packets[1][: len(packets[1]) // 3]))
    decode_step_frames(corrupt, decoders, VIDEO_INPUTS)

    pictures = _decode(_step(*(_h264(p) for p in packets)), decoders)

    expected = _reference(packets)
    assert all(np.array_equal(pictures[0, i], expected[i]) for i in range(4))


def test_h264_frames_for_a_track_that_is_not_an_input_video_track_open_no_decoder() -> None:
    packets = _encode(_pictures(4))
    decoders: dict[str, TrackDecoder] = {}

    outcome = decode_step_frames(
        _step(*(_h264(p) for p in packets), track="mic"), decoders, VIDEO_INPUTS
    )

    assert outcome == StepFrameRefusal(
        "media_input[0]",
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


def test_frames_that_do_not_start_at_a_keyframe_are_refused() -> None:
    packets = _encode(_pictures(4))
    refusal = _refused(_step(*(_h264(p) for p in packets[1:])))

    assert refusal.entry == "media_input[0].frames[0]"
    assert "keyframe" in refusal.reason


def test_a_frame_holding_a_p_frame_then_an_idr_is_refused() -> None:
    # A decoder would take the two as one packet and keep only the second picture.
    packets = _encode(_pictures(8))
    refusal = _refused(_step(_h264(packets[1] + packets[4])))

    assert refusal.entry == "media_input[0].frames[0]"
    assert "more than one access unit" in refusal.reason


def test_a_frame_holding_two_pictures_is_refused() -> None:
    packets = _encode(_pictures(4))
    refusal = _refused(_step(_h264(packets[0] + packets[1])))

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


def test_an_empty_h264_frame_is_refused() -> None:
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


def test_only_a_v1_control_publish_or_unpublish_is_a_track_event() -> None:
    assert track_event(_publish("webcam"), ProtocolVersion.V1, Channel.CONTROL) == (
        True,
        "webcam",
    )
    assert track_event(_unpublish("webcam"), ProtocolVersion.V1, Channel.CONTROL) == (
        False,
        "webcam",
    )
    assert track_event(_publish("webcam"), ProtocolVersion.V1, Channel.DATA) is None
    assert track_event(_publish("webcam"), ProtocolVersion.V0, Channel.CONTROL) is None
    ping = _wire(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    assert track_event(ping, ProtocolVersion.V1, Channel.CONTROL) is None
    assert track_event(b"\xff not protobuf", ProtocolVersion.V1, Channel.CONTROL) is None


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
    gate.submit(_publish("other"), ProtocolVersion.V1, Channel.CONTROL)
    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)

    gate.close()
    await _settle()

    assert gate.decoders == 0
