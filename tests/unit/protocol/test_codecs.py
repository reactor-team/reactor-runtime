import json
from typing import Any

import pytest

from reactor_runtime import protocol
from reactor_runtime.protocol.common import dict_to_struct
from reactor_wire.v1 import (
    common_pb2,
    control_pb2,
    data_pb2,
    model_pb2,
    platform_pb2,
    track_pb2,
)

K = common_pb2.MessageKind
CLIENT = protocol.Direction.CLIENT
SERVER = protocol.Direction.SERVER
DATA = protocol.Channel.DATA
CONTROL = protocol.Channel.CONTROL

# Each case pairs a v1 message with its exact legacy v0 wire shape: the physical
# channel it rides on and the parsed JSON body. Sourced from the shipped-client
# protocol inventory.
CASES = [
    pytest.param(
        data_pb2.DataClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION,
            command=model_pb2.Command(
                type="set_prompt", data=dict_to_struct({"prompt": "hi", "strength": 0.5})
            ),
        ),
        CLIENT,
        DATA,
        {
            "scope": "application",
            "data": {"type": "set_prompt", "data": {"prompt": "hi", "strength": 0.5}},
        },
        id="command",
    ),
    pytest.param(
        data_pb2.DataClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION,
            command=model_pb2.Command(
                type="set_image",
                data=dict_to_struct({}),
                uploads={
                    "image": model_pb2.UploadReference(
                        upload_id="u1", name="a.png", mime_type="image/png", size=10
                    )
                },
            ),
        ),
        CLIENT,
        DATA,
        {
            "scope": "application",
            "data": {
                "type": "set_image",
                "data": {},
                "uploads": {
                    "image": {
                        "upload_id": "u1",
                        "name": "a.png",
                        "mime_type": "image/png",
                        "size": 10,
                    }
                },
            },
        },
        id="command-with-uploads",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION, ping=platform_pb2.Ping()
        ),
        CLIENT,
        DATA,
        {"scope": "runtime", "data": {"type": "ping", "data": {}}},
        id="ping",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_REQUEST, request_schema=platform_pb2.RequestSchema()
        ),
        CLIENT,
        DATA,
        {"scope": "runtime", "data": {"type": "requestSchema", "data": {}}},
        id="request-schema",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION,
            file_uploaded=platform_pb2.FileUploaded(
                upload_id="u1", name="a.png", mime_type="image/png", size=10
            ),
        ),
        CLIENT,
        DATA,
        {
            "scope": "runtime",
            "data": {
                "type": "fileUploaded",
                "data": {"upload_id": "u1", "name": "a.png", "mime_type": "image/png", "size": 10},
            },
        },
        id="file-uploaded",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_REQUEST,
            request_clip=platform_pb2.RequestClip(duration_seconds=5.0),
        ),
        CLIENT,
        DATA,
        {"scope": "runtime", "data": {"type": "requestClip", "data": {"duration_seconds": 5.0}}},
        id="request-clip",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_REQUEST, request_recording=platform_pb2.RequestRecording()
        ),
        CLIENT,
        DATA,
        {"scope": "runtime", "data": {"type": "requestRecording", "data": {}}},
        id="request-recording",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            request_id="ctrl_1",
            kind=K.MESSAGE_KIND_REQUEST,
            publish_track=track_pb2.PublishTrack(name="VideoIn"),
        ),
        CLIENT,
        CONTROL,
        {
            "type": "request",
            "method": "publish_track",
            "request_id": "ctrl_1",
            "data": {"name": "VideoIn"},
        },
        id="publish-track",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION, pause_track=track_pb2.PauseTrack(name="VideoIn")
        ),
        CLIENT,
        CONTROL,
        {"type": "notification", "event": "pause_track", "data": {"name": "VideoIn"}},
        id="pause-track",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION, resume_track=track_pb2.ResumeTrack(name="VideoIn")
        ),
        CLIENT,
        CONTROL,
        {"type": "notification", "event": "resume_track", "data": {"name": "VideoIn"}},
        id="resume-track",
    ),
    pytest.param(
        control_pb2.ControlClientMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION,
            unpublish_track=track_pb2.UnpublishTrack(name="VideoIn"),
        ),
        CLIENT,
        CONTROL,
        {"type": "notification", "event": "unpublish_track", "data": {"name": "VideoIn"}},
        id="unpublish-track",
    ),
    pytest.param(
        data_pb2.DataServerMessage(
            kind=K.MESSAGE_KIND_NOTIFICATION,
            message=model_pb2.ModelMessage(type="progress", data=dict_to_struct({"pct": 50.0})),
        ),
        SERVER,
        DATA,
        {"scope": "application", "data": {"type": "progress", "data": {"pct": 50.0}}},
        id="model-message",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            kind=K.MESSAGE_KIND_RESPONSE,
            model_schema=platform_pb2.ModelSchema(openapi=dict_to_struct({"openapi": "3.1.0"})),
        ),
        SERVER,
        DATA,
        {"scope": "runtime", "data": {"type": "modelSchema", "data": {"openapi": "3.1.0"}}},
        id="model-schema",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            kind=K.MESSAGE_KIND_RESPONSE,
            clip_ready=platform_pb2.ClipReady(
                session_id="s1",
                kind="snap",
                start_marker=1.0,
                end_marker=2.0,
                now_marker=1.5,
                predicted_ready_at_ms=1234,
                playlist_url="https://x/p.m3u8",
            ),
        ),
        SERVER,
        DATA,
        {
            "scope": "runtime",
            "data": {
                "type": "clipReady",
                "data": {
                    "session_id": "s1",
                    "kind": "snap",
                    "start_marker": 1.0,
                    "end_marker": 2.0,
                    "now_marker": 1.5,
                    "predicted_ready_at_ms": 1234,
                    "playlist_url": "https://x/p.m3u8",
                },
            },
        },
        id="clip-ready",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            kind=K.MESSAGE_KIND_RESPONSE,
            clip_ready=platform_pb2.ClipReady(
                session_id="s1",
                kind="recording",
                start_marker=1.0,
                end_marker=2.0,
                now_marker=1.5,
                predicted_ready_at_ms=1234,
                playlist_url="https://x/p.m3u8",
            ),
        ),
        SERVER,
        DATA,
        {
            "scope": "runtime",
            "data": {
                "type": "clipReady",
                "data": {
                    "session_id": "s1",
                    "kind": "recording",
                    "start_marker": 1.0,
                    "end_marker": 2.0,
                    "now_marker": 1.5,
                    "predicted_ready_at_ms": 1234,
                    "playlist_url": "https://x/p.m3u8",
                },
            },
        },
        id="clip-ready-no-marker",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            kind=K.MESSAGE_KIND_RESPONSE,
            clip_failed=platform_pb2.ClipFailed(reason="recorder disabled"),
        ),
        SERVER,
        DATA,
        {
            "scope": "runtime",
            "data": {"type": "clipFailed", "data": {"reason": "recorder disabled"}},
        },
        id="clip-failed",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            request_id="ctrl_1",
            kind=K.MESSAGE_KIND_RESPONSE,
            publish_track=track_pb2.PublishTrackResponse(),
        ),
        SERVER,
        CONTROL,
        {"type": "response", "method": "publish_track", "request_id": "ctrl_1", "data": {}},
        id="publish-track-response",
    ),
    pytest.param(
        control_pb2.ControlServerMessage(
            request_id="ctrl_1",
            kind=K.MESSAGE_KIND_RESPONSE,
            error=common_pb2.Error(code="PUBLISHER_SLOT_TAKEN", message="taken"),
        ),
        SERVER,
        CONTROL,
        {
            "type": "response",
            "method": "publish_track",
            "request_id": "ctrl_1",
            "error": {"code": "PUBLISHER_SLOT_TAKEN", "message": "taken"},
        },
        id="publish-track-error",
    ),
]


@pytest.mark.parametrize(("message", "direction", "channel", "legacy"), CASES)
def test_v0_encodes_legacy_wire(
    message: protocol.Message,
    direction: protocol.Direction,
    channel: protocol.Channel,
    legacy: dict[str, Any],
) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    out_channel, frame = codec.encode(message)
    assert out_channel is channel
    assert isinstance(frame, str)
    assert json.loads(frame) == legacy


@pytest.mark.parametrize(("message", "direction", "channel", "legacy"), CASES)
def test_v0_message_roundtrip(
    message: protocol.Message,
    direction: protocol.Direction,
    channel: protocol.Channel,
    legacy: dict[str, Any],
) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    out_channel, frame = codec.encode(message)
    assert codec.decode(frame, out_channel, direction) == message


@pytest.mark.parametrize(("message", "direction", "channel", "legacy"), CASES)
def test_v0_legacy_frame_is_stable(
    message: protocol.Message,
    direction: protocol.Direction,
    channel: protocol.Channel,
    legacy: dict[str, Any],
) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    decoded = codec.decode(json.dumps(legacy), channel, direction)
    _, frame = codec.encode(decoded)
    assert json.loads(frame) == legacy


@pytest.mark.parametrize(("message", "direction", "channel", "legacy"), CASES)
def test_v1_message_roundtrip(
    message: protocol.Message,
    direction: protocol.Direction,
    channel: protocol.Channel,
    legacy: dict[str, Any],
) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)
    out_channel, frame = codec.encode(message)
    # v1 maps each message to its own channel, unlike v0's legacy placement.
    assert out_channel is protocol.logical_channel(message)
    assert isinstance(frame, bytes)
    assert codec.decode(frame, out_channel, direction) == message


@pytest.mark.parametrize(
    ("message", "direction"),
    [
        (
            data_pb2.DataClientMessage(
                kind=K.MESSAGE_KIND_RESPONSE, error=common_pb2.Error(code="x")
            ),
            CLIENT,
        ),
        (
            data_pb2.DataServerMessage(
                kind=K.MESSAGE_KIND_RESPONSE, error=common_pb2.Error(code="x")
            ),
            SERVER,
        ),
        (
            control_pb2.ControlClientMessage(
                kind=K.MESSAGE_KIND_RESPONSE, error=common_pb2.Error(code="x")
            ),
            CLIENT,
        ),
    ],
)
def test_v1_roundtrips_error_variants(
    message: protocol.Message, direction: protocol.Direction
) -> None:
    # These correlated-error payloads exist only in v1 (v0 had no app-layer
    # error framing), so they are exercised here rather than in the v0 cases.
    codec = protocol.select(protocol.ProtocolVersion.V1)
    out_channel, frame = codec.encode(message)
    assert codec.decode(frame, out_channel, direction) == message


def test_every_payload_variant_is_covered() -> None:
    # Guard against adding a message to the schema without a v0 mapping + test.
    # Correlated-error payloads have no legacy representation; v0 rejects them.
    unsupported = {
        ("DataClientMessage", "error"),
        ("DataServerMessage", "error"),
        ("ControlClientMessage", "error"),
    }
    message_types: list[Any] = [
        data_pb2.DataClientMessage,
        data_pb2.DataServerMessage,
        control_pb2.ControlClientMessage,
        control_pb2.ControlServerMessage,
    ]
    covered: set[tuple[str, str | None]] = set()
    for case in CASES:
        message: Any = case.values[0]
        covered.add((type(message).__name__, message.WhichOneof("payload")))
    for message_type in message_types:
        name = message_type.__name__
        for field in message_type.DESCRIPTOR.oneofs_by_name["payload"].fields:
            key = (name, field.name)
            assert key in covered or key in unsupported, f"{name}.{field.name} has no v0 case"


@pytest.mark.parametrize(
    "message",
    [
        data_pb2.DataServerMessage(
            kind=K.MESSAGE_KIND_RESPONSE, error=common_pb2.Error(code="X", message="y")
        ),
        data_pb2.DataClientMessage(
            kind=K.MESSAGE_KIND_RESPONSE, error=common_pb2.Error(code="X", message="y")
        ),
    ],
)
def test_v0_rejects_messages_without_legacy_form(message: protocol.Message) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    with pytest.raises(protocol.UnsupportedMessageError):
        codec.encode(message)


@pytest.mark.parametrize(
    ("frame", "channel", "direction"),
    [
        ('{"scope": "bogus", "data": {}}', DATA, CLIENT),
        ('{"scope": "runtime", "data": {"type": "nope"}}', DATA, CLIENT),
        ('{"type": "request", "method": "nope", "request_id": "1", "data": {}}', CONTROL, CLIENT),
    ],
)
def test_v0_rejects_unknown_legacy_frames(
    frame: str, channel: protocol.Channel, direction: protocol.Direction
) -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    with pytest.raises(protocol.UnsupportedMessageError):
        codec.decode(frame, channel, direction)


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ("{}", protocol.ProtocolVersion.V0),
        ('{"scope": "runtime"}', protocol.ProtocolVersion.V0),
        (b'{"scope": "runtime"}', protocol.ProtocolVersion.V0),
        (b"  {\n}", protocol.ProtocolVersion.V0),
        (b"\x08\x01\x12\x04ping", protocol.ProtocolVersion.V1),
    ],
)
def test_sniff(frame: bytes | str, expected: protocol.ProtocolVersion) -> None:
    assert protocol.sniff(frame) is expected


def test_select_returns_versioned_codec() -> None:
    assert protocol.select(protocol.ProtocolVersion.V0).version is protocol.ProtocolVersion.V0
    assert protocol.select(protocol.ProtocolVersion.V1).version is protocol.ProtocolVersion.V1


_CLIP = {
    "session_id": "rec-1",
    "kind": "snap",
    "start_marker": 1.0,
    "end_marker": 2.0,
    "now_marker": 2.0,
    "predicted_ready_at_ms": 123,
    "playlist_url": "/clips?session_id=rec-1&start=1.000&end=2.000",
}


def test_v0_clip_ready_rides_the_data_channel_without_an_id() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    channel, frame = codec.encode_clip_ready("ctrl_1", _CLIP)
    assert channel is DATA
    decoded = codec.decode(frame, DATA, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.WhichOneof("payload") == "clip_ready"
    assert decoded.clip_ready.session_id == "rec-1"


def test_v1_clip_ready_rides_control_correlated_by_id() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)
    channel, frame = codec.encode_clip_ready("ctrl_1", _CLIP)
    assert channel is CONTROL
    decoded = codec.decode(frame, CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.request_id == "ctrl_1"
    assert decoded.clip_ready.kind == "snap"


def test_v1_clip_failed_carries_its_reason_and_id() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)
    channel, frame = codec.encode_clip_failed("ctrl_2", "recorder disabled")
    assert channel is CONTROL
    decoded = codec.decode(frame, CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.request_id == "ctrl_2"
    assert decoded.clip_failed.reason == "recorder disabled"


def test_v1_command_ack_is_a_bodyless_response() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)
    channel, frame = codec.encode_command_ack("req-1")
    assert channel is DATA
    decoded = codec.decode(frame, DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.request_id == "req-1"
    assert decoded.kind == K.MESSAGE_KIND_RESPONSE
    assert decoded.WhichOneof("payload") is None


def test_v1_command_error_carries_its_code_and_detail() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)
    channel, frame = codec.encode_command_error("req-2", "invalid_command", "value out of range")
    assert channel is DATA
    decoded = codec.decode(frame, DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.request_id == "req-2"
    assert decoded.kind == K.MESSAGE_KIND_RESPONSE
    assert decoded.error.code == "invalid_command"
    assert decoded.error.message == "value out of range"


def test_v0_rejects_command_ack_and_error() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)
    with pytest.raises(protocol.UnsupportedMessageError):
        codec.encode_command_ack("req-1")
    with pytest.raises(protocol.UnsupportedMessageError):
        codec.encode_command_error("req-2", "invalid_command", "value out of range")
