"""Frozen legacy JSON codec (wire version v0).

This module is the whole v0 surface: the raw JSON the shipped clients speak and
an explicit adapter to and from the ``reactor_wire.v1`` messages. It knows
only the legacy message set and is frozen, so it needs no maintenance as v1
evolves; ending v0 support is deleting this package.

Two legacy traits the adapter encodes:

- Channel placement. Legacy clients put every platform message on the data
  channel under ``{"scope": "runtime"}`` and use the control channel only for
  track verbs. The v1 control-channel messages are therefore split back across
  both physical channels here.
- No correlation. Legacy app commands and platform replies are fire-and-forget
  (clips were correlated by receipt order), so v0 frames carry no request_id
  except on the control channel's publish_track exchange.
"""

from __future__ import annotations

import json
from typing import Any

from reactor_runtime.protocol.base import (
    Channel,
    Codec,
    Direction,
    Message,
    ProtocolVersion,
    UnsupportedMessageError,
)
from reactor_runtime.protocol.common import (
    dict_to_struct,
    dict_to_upload_reference,
    struct_to_dict,
    upload_reference_to_dict,
)
from reactor_wire.v1 import (
    common_pb2,
    control_pb2,
    data_pb2,
    model_pb2,
    platform_pb2,
    track_pb2,
)

_KIND = common_pb2.MessageKind

_SCOPE_APPLICATION = "application"
_SCOPE_RUNTIME = "runtime"


class V0Codec(Codec):
    """Encodes messages as the legacy JSON shipped clients speak."""

    version = ProtocolVersion.V0

    def encode(self, message: Message) -> tuple[Channel, bytes | str]:
        """Render a message as legacy JSON on the channel its v0 form used."""
        if isinstance(message, data_pb2.DataClientMessage):
            return self._encode_command(message)
        if isinstance(message, data_pb2.DataServerMessage):
            return self._encode_model_message(message)
        if isinstance(message, control_pb2.ControlClientMessage):
            return self._encode_control_client(message)
        return self._encode_control_server(message)

    def decode(self, frame: bytes | str, channel: Channel, direction: Direction) -> Message:
        """Parse a legacy JSON frame into its reactor_wire.v1 message."""
        text = frame.decode("utf-8") if isinstance(frame, bytes | bytearray) else frame
        obj = json.loads(text)
        if channel is Channel.CONTROL:
            return self._decode_control(obj, direction)
        return self._decode_data(obj, direction)

    # -- encode: data channel -------------------------------------------------

    def _encode_command(self, message: data_pb2.DataClientMessage) -> tuple[Channel, str]:
        which = message.WhichOneof("payload")
        if which != "command":
            raise UnsupportedMessageError(f"v0 data channel cannot carry {which!r}")
        command = message.command
        inner: dict[str, Any] = {"type": command.type, "data": struct_to_dict(command.data)}
        if command.uploads:
            inner["uploads"] = {
                name: upload_reference_to_dict(ref) for name, ref in command.uploads.items()
            }
        return Channel.DATA, _dump({"scope": _SCOPE_APPLICATION, "data": inner})

    def _encode_model_message(self, message: data_pb2.DataServerMessage) -> tuple[Channel, str]:
        which = message.WhichOneof("payload")
        if which != "message":
            raise UnsupportedMessageError(f"v0 data channel cannot carry {which!r}")
        model_message = message.message
        inner = {"type": model_message.type, "data": struct_to_dict(model_message.data)}
        return Channel.DATA, _dump({"scope": _SCOPE_APPLICATION, "data": inner})

    # -- encode: control + platform ------------------------------------------

    def _encode_control_client(
        self, message: control_pb2.ControlClientMessage
    ) -> tuple[Channel, str]:
        which = message.WhichOneof("payload")
        if which == "ping":
            return Channel.DATA, _runtime("ping", {})
        if which == "request_schema":
            return Channel.DATA, _runtime("requestSchema", {})
        if which == "file_uploaded":
            upload = message.file_uploaded
            return Channel.DATA, _runtime(
                "fileUploaded",
                {
                    "upload_id": upload.upload_id,
                    "name": upload.name,
                    "mime_type": upload.mime_type,
                    "size": upload.size,
                },
            )
        if which == "request_clip":
            return Channel.DATA, _runtime(
                "requestClip", {"duration_seconds": message.request_clip.duration_seconds}
            )
        if which == "request_recording":
            return Channel.DATA, _runtime("requestRecording", {})
        if which == "publish_track":
            return Channel.CONTROL, _dump(
                {
                    "type": "request",
                    "method": "publish_track",
                    "request_id": message.request_id,
                    "data": {"name": message.publish_track.name},
                }
            )
        if which in ("pause_track", "resume_track", "unpublish_track"):
            track = getattr(message, which)
            return Channel.CONTROL, _dump(
                {"type": "notification", "event": which, "data": {"name": track.name}}
            )
        raise UnsupportedMessageError(f"v0 cannot encode control client payload {which!r}")

    def _encode_control_server(
        self, message: control_pb2.ControlServerMessage
    ) -> tuple[Channel, str]:
        which = message.WhichOneof("payload")
        if which == "model_schema":
            return Channel.DATA, _runtime(
                "modelSchema", struct_to_dict(message.model_schema.openapi)
            )
        if which == "clip_ready":
            return Channel.DATA, _runtime("clipReady", _clip_ready_to_dict(message.clip_ready))
        if which == "clip_failed":
            return Channel.DATA, _runtime("clipFailed", {"reason": message.clip_failed.reason})
        if which == "moderation":
            return Channel.DATA, _runtime("moderation", _moderation_to_dict(message.moderation))
        if which == "publish_track":
            return Channel.CONTROL, _dump(
                {
                    "type": "response",
                    "method": "publish_track",
                    "request_id": message.request_id,
                    "data": {},
                }
            )
        if which == "error":
            return Channel.CONTROL, _dump(
                {
                    "type": "response",
                    "method": "publish_track",
                    "request_id": message.request_id,
                    "error": {"code": message.error.code, "message": message.error.message},
                }
            )
        raise UnsupportedMessageError(f"v0 cannot encode control server payload {which!r}")

    # -- decode: control channel ---------------------------------------------

    def _decode_control(self, obj: dict[str, Any], direction: Direction) -> Message:
        if direction is Direction.CLIENT:
            return self._decode_control_request(obj)
        return self._decode_control_response(obj)

    def _decode_control_request(self, obj: dict[str, Any]) -> control_pb2.ControlClientMessage:
        kind = obj.get("type")
        data = obj.get("data") or {}
        if kind == "request" and obj.get("method") == "publish_track":
            return control_pb2.ControlClientMessage(
                request_id=str(obj.get("request_id", "")),
                kind=_KIND.MESSAGE_KIND_REQUEST,
                publish_track=track_pb2.PublishTrack(name=str(data.get("name", ""))),
            )
        if kind == "notification":
            event = obj.get("event")
            name = str(data.get("name", ""))
            if event == "pause_track":
                return control_pb2.ControlClientMessage(
                    kind=_KIND.MESSAGE_KIND_NOTIFICATION,
                    pause_track=track_pb2.PauseTrack(name=name),
                )
            if event == "resume_track":
                return control_pb2.ControlClientMessage(
                    kind=_KIND.MESSAGE_KIND_NOTIFICATION,
                    resume_track=track_pb2.ResumeTrack(name=name),
                )
            if event == "unpublish_track":
                return control_pb2.ControlClientMessage(
                    kind=_KIND.MESSAGE_KIND_NOTIFICATION,
                    unpublish_track=track_pb2.UnpublishTrack(name=name),
                )
        raise UnsupportedMessageError(f"unrecognized v0 control request: {obj!r}")

    def _decode_control_response(self, obj: dict[str, Any]) -> control_pb2.ControlServerMessage:
        if obj.get("type") != "response":
            raise UnsupportedMessageError(f"unrecognized v0 control response: {obj!r}")
        request_id = str(obj.get("request_id", ""))
        if "error" in obj:
            error = obj["error"] or {}
            return control_pb2.ControlServerMessage(
                request_id=request_id,
                kind=_KIND.MESSAGE_KIND_RESPONSE,
                error=common_pb2.Error(
                    code=str(error.get("code", "")), message=str(error.get("message", ""))
                ),
            )
        return control_pb2.ControlServerMessage(
            request_id=request_id,
            kind=_KIND.MESSAGE_KIND_RESPONSE,
            publish_track=track_pb2.PublishTrackResponse(),
        )

    # -- decode: data channel -------------------------------------------------

    def _decode_data(self, obj: dict[str, Any], direction: Direction) -> Message:
        scope = obj.get("scope")
        inner = obj.get("data") or {}
        if scope == _SCOPE_APPLICATION:
            return self._decode_application(inner, direction)
        if scope == _SCOPE_RUNTIME:
            return self._decode_runtime(inner, direction)
        raise UnsupportedMessageError(f"unrecognized v0 data envelope scope: {scope!r}")

    def _decode_application(self, inner: dict[str, Any], direction: Direction) -> Message:
        name = str(inner.get("type", ""))
        body = inner.get("data") or {}
        if direction is Direction.CLIENT:
            command = model_pb2.Command(type=name, data=dict_to_struct(body))
            for param, ref in (inner.get("uploads") or {}).items():
                command.uploads[param].CopyFrom(dict_to_upload_reference(ref))
            return data_pb2.DataClientMessage(kind=_KIND.MESSAGE_KIND_NOTIFICATION, command=command)
        return data_pb2.DataServerMessage(
            kind=_KIND.MESSAGE_KIND_NOTIFICATION,
            message=model_pb2.ModelMessage(type=name, data=dict_to_struct(body)),
        )

    def _decode_runtime(self, inner: dict[str, Any], direction: Direction) -> Message:
        kind = inner.get("type")
        data = inner.get("data") or {}
        if direction is Direction.CLIENT:
            return self._decode_runtime_client(kind, data)
        return self._decode_runtime_server(kind, data)

    def _decode_runtime_client(
        self, kind: str | None, data: dict[str, Any]
    ) -> control_pb2.ControlClientMessage:
        if kind == "ping":
            return control_pb2.ControlClientMessage(
                kind=_KIND.MESSAGE_KIND_NOTIFICATION, ping=platform_pb2.Ping()
            )
        if kind == "requestSchema":
            return control_pb2.ControlClientMessage(
                kind=_KIND.MESSAGE_KIND_REQUEST, request_schema=platform_pb2.RequestSchema()
            )
        if kind == "fileUploaded":
            return control_pb2.ControlClientMessage(
                kind=_KIND.MESSAGE_KIND_NOTIFICATION,
                file_uploaded=platform_pb2.FileUploaded(
                    upload_id=str(data.get("upload_id", "")),
                    name=str(data.get("name", "")),
                    mime_type=str(data.get("mime_type", "")),
                    size=int(data.get("size", 0)),
                ),
            )
        if kind == "requestClip":
            return control_pb2.ControlClientMessage(
                kind=_KIND.MESSAGE_KIND_REQUEST,
                request_clip=platform_pb2.RequestClip(
                    duration_seconds=float(data.get("duration_seconds", 0.0))
                ),
            )
        if kind == "requestRecording":
            return control_pb2.ControlClientMessage(
                kind=_KIND.MESSAGE_KIND_REQUEST, request_recording=platform_pb2.RequestRecording()
            )
        raise UnsupportedMessageError(f"unrecognized v0 runtime client message: {kind!r}")

    def _decode_runtime_server(
        self, kind: str | None, data: dict[str, Any]
    ) -> control_pb2.ControlServerMessage:
        if kind == "modelSchema":
            return control_pb2.ControlServerMessage(
                kind=_KIND.MESSAGE_KIND_RESPONSE,
                model_schema=platform_pb2.ModelSchema(openapi=dict_to_struct(data)),
            )
        if kind == "clipReady":
            return control_pb2.ControlServerMessage(
                kind=_KIND.MESSAGE_KIND_RESPONSE, clip_ready=_dict_to_clip_ready(data)
            )
        if kind == "clipFailed":
            return control_pb2.ControlServerMessage(
                kind=_KIND.MESSAGE_KIND_RESPONSE,
                clip_failed=platform_pb2.ClipFailed(reason=str(data.get("reason", ""))),
            )
        if kind == "moderation":
            return control_pb2.ControlServerMessage(
                kind=_KIND.MESSAGE_KIND_NOTIFICATION,
                moderation=_dict_to_moderation(data),
            )
        raise UnsupportedMessageError(f"unrecognized v0 runtime server message: {kind!r}")


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _runtime(message_type: str, data: dict[str, Any]) -> str:
    return _dump({"scope": _SCOPE_RUNTIME, "data": {"type": message_type, "data": data}})


def _clip_ready_to_dict(clip: platform_pb2.ClipReady) -> dict[str, Any]:
    return {
        "session_id": clip.session_id,
        "kind": clip.kind,
        "start_marker": clip.start_marker,
        "end_marker": clip.end_marker,
        "now_marker": clip.now_marker,
        "predicted_ready_at_ms": clip.predicted_ready_at_ms,
        "playlist_url": clip.playlist_url,
    }


def _moderation_to_dict(moderation: platform_pb2.Moderation) -> dict[str, Any]:
    return {
        "action": moderation.action,
        "input_kind": moderation.input_kind,
        "command": moderation.command,
        "categories": list(moderation.categories),
        "message": moderation.message,
    }


def _dict_to_moderation(data: dict[str, Any]) -> platform_pb2.Moderation:
    return platform_pb2.Moderation(
        action=str(data.get("action", "")),
        input_kind=str(data.get("input_kind", "")),
        command=str(data.get("command", "")),
        categories=[str(c) for c in data.get("categories") or []],
        message=str(data.get("message", "")),
    )


def _dict_to_clip_ready(data: dict[str, Any]) -> platform_pb2.ClipReady:
    return platform_pb2.ClipReady(
        session_id=str(data.get("session_id", "")),
        kind=str(data.get("kind", "")),
        start_marker=float(data.get("start_marker", 0.0)),
        end_marker=float(data.get("end_marker", 0.0)),
        now_marker=float(data.get("now_marker", 0.0)),
        predicted_ready_at_ms=int(data.get("predicted_ready_at_ms", 0)),
        playlist_url=str(data.get("playlist_url", "")),
    )
