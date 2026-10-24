import asyncio

import av
import numpy as np

from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_runtime.transport.step_frames import (
    StepFrameGate,
    decode_step_frames,
    encoded_step,
)
from reactor_wire.v1 import common_pb2, data_pb2, model_pb2, platform_pb2, step_pb2

WIDTH, HEIGHT = 64, 48


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


def _wire(message: data_pb2.DataClientMessage) -> bytes:
    _, frame = V1Codec().encode(message)
    assert isinstance(frame, bytes)
    return frame


# ── decode_step_frames ─────────────────────────────────────────────────────────


def test_h264_frames_become_the_raw_frames_they_decode_to() -> None:
    packets = _encode(_pictures(4))
    message = _step(
        *(
            _h264(p, metadata=f"tag {i}".encode(), capture_time_us=1000 + i)
            for i, p in enumerate(packets)
        )
    )

    assert decode_step_frames(message) is None

    frames = message.request_step.media_input[0].frames
    assert [f.encoding for f in frames] == [step_pb2.FRAME_ENCODING_RAW] * 4
    assert all(f.dtype == "uint8" and list(f.shape) == [HEIGHT, WIDTH, 3] for f in frames)
    reference = _reference(packets)
    for frame, expected in zip(frames, reference, strict=True):
        assert frame.data == expected.tobytes()
    # Each frame keeps what its encoded frame carried.
    assert [f.metadata for f in frames] == [f"tag {i}".encode() for i in range(4)]
    assert [f.capture_time_us for f in frames] == [1000, 1001, 1002, 1003]


def test_a_stream_with_b_frames_decodes_one_picture_per_frame() -> None:
    packets = _encode(_pictures(8), bf="2", tune="film")
    message = _step(*(_h264(p, metadata=bytes([i])) for i, p in enumerate(packets)))

    assert decode_step_frames(message) is None

    frames = message.request_step.media_input[0].frames
    assert len(frames) == len(packets)
    assert all(f.encoding == step_pb2.FRAME_ENCODING_RAW for f in frames)
    # The tags stay on the frames they were sent with, in the order sent.
    assert [f.metadata for f in frames] == [bytes([i]) for i in range(len(packets))]


def test_raw_frames_beside_h264_ones_are_left_as_they_are() -> None:
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

    assert decode_step_frames(message) is None
    assert message.request_step.media_input[1].frames[0] == raw


def test_an_unknown_encoding_is_refused_naming_its_frame() -> None:
    (packet,) = _encode(_pictures(1))
    unknown = step_pb2.Frame(dtype="uint8", shape=[HEIGHT, WIDTH, 3], data=b"x")
    # Proto3 enums are open: a peer may send a value this build has no name for.
    setattr(unknown, "encoding", 99)  # noqa: B010
    refusal = decode_step_frames(_step(_h264(packet), unknown))

    assert refusal is not None
    assert refusal.entry == "media_input[0].frames[1]"
    assert "99 is not supported" in refusal.reason


def test_frames_that_do_not_start_at_a_keyframe_are_refused() -> None:
    packets = _encode(_pictures(4))
    refusal = decode_step_frames(_step(*(_h264(p) for p in packets[1:])))

    assert refusal is not None
    assert refusal.entry == "media_input[0].frames[0]"
    assert "keyframe" in refusal.reason


def test_a_frame_holding_a_p_frame_then_an_idr_is_refused() -> None:
    # A decoder would take the two as one packet and keep only the second picture.
    packets = _encode(_pictures(8))
    refusal = decode_step_frames(_step(_h264(packets[1] + packets[4])))

    assert refusal is not None
    assert refusal.entry == "media_input[0].frames[0]"
    assert "more than one access unit" in refusal.reason


def test_a_frame_holding_two_pictures_is_refused() -> None:
    packets = _encode(_pictures(4))
    refusal = decode_step_frames(_step(_h264(packets[0] + packets[1])))

    assert refusal is not None
    assert refusal.entry == "media_input[0].frames[0]"
    assert "more than one picture" in refusal.reason


def test_a_frame_with_no_picture_is_refused() -> None:
    from reactor_runtime.transport.step_frames import _nal_units

    (first, *_) = _encode(_pictures(4))
    parameter_sets = b"".join(
        b"\x00\x00\x00\x01" + unit for unit in _nal_units(first) if unit[0] & 0x1F in (7, 8)
    )
    refusal = decode_step_frames(_step(_h264(parameter_sets)))

    assert refusal is not None
    assert "holds no picture" in refusal.reason


def test_a_frame_that_decodes_to_another_size_is_refused() -> None:
    (packet,) = _encode(_pictures(1))
    refusal = decode_step_frames(_step(_h264(packet, shape=[720, 1280, 3])))

    assert refusal is not None
    assert refusal.entry == "media_input[0].frames[0]"
    assert f"[{HEIGHT}, {WIDTH}, 3]" in refusal.reason


def test_an_empty_h264_frame_is_refused() -> None:
    refusal = decode_step_frames(_step(_h264(b"")))

    assert refusal is not None
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
        self.replies: list[bytes] = []

    def gate(self) -> StepFrameGate:
        return StepFrameGate(
            lambda payload, version, channel: self.delivered.append(_as_bytes(payload)),
            lambda payload: self.replies.append(_as_bytes(payload)),
        )


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0.01)


async def test_a_plain_message_is_delivered_at_once() -> None:
    rec = _Recorder()
    rec.gate().submit(b"anything", ProtocolVersion.V1, Channel.DATA)

    assert rec.delivered == [b"anything"]


async def test_an_encoded_step_is_delivered_decoded_and_nothing_overtakes_it() -> None:
    rec = _Recorder()
    gate = rec.gate()
    packets = _encode(_pictures(4))
    command = _wire(
        data_pb2.DataClientMessage(request_id="data_8", command=model_pb2.Command(type="ping"))
    )

    gate.submit(_wire(_step(*(_h264(p) for p in packets))), ProtocolVersion.V1, Channel.DATA)
    gate.submit(command, ProtocolVersion.V1, Channel.DATA)
    assert rec.delivered == []  # the command waits behind the decode
    await _settle()

    assert len(rec.delivered) == 2
    step = data_pb2.DataClientMessage.FromString(rec.delivered[0])
    assert step.request_id == "data_7"
    assert [f.encoding for f in step.request_step.media_input[0].frames] == [
        step_pb2.FRAME_ENCODING_RAW
    ] * 4
    assert rec.delivered[1] == command
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
    await _settle()

    assert rec.delivered == [command]
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

    assert rec.delivered == []
    assert rec.replies == []
