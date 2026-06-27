"""Rendering runner events as Server-Sent Events.

The egress journal holds neutral :data:`~reactor_runtime.core.model.RunnerEvent`
facts; an HTTP egress route streams them out as SSE. This module is the one
place that decides a runner event's wire shape — a ``{"type": ...}`` envelope —
and frames it for the ``text/event-stream`` body, so a consumer can mirror the
runtime's view and resume from a sequence number it has seen.
"""

from __future__ import annotations

import json
from typing import Any

from reactor_runtime.core import (
    ClipReadyEvent,
    ErrorEvent,
    InboundCommandEvent,
    RunnerEvent,
    SessionMetricEvent,
    TransitionEvent,
)


def runner_event_to_dict(event: RunnerEvent) -> dict[str, Any]:
    """Render a runner event as a JSON-serialisable ``{"type": ...}`` envelope.

    Args:
        event: The runner event to render.

    Returns:
        A plain dict whose ``type`` names the event and whose remaining keys
        carry its fields.
    """
    if isinstance(event, TransitionEvent):
        transition = event.transition
        return {
            "type": "transition",
            "event": transition.event.name.lower(),
            "from": transition.from_state.name.lower(),
            "to": transition.to_state.name.lower(),
            "detail": dict(transition.detail),
        }
    if isinstance(event, InboundCommandEvent):
        conn_id = None if event.conn_id is None else int(event.conn_id)
        return {"type": "command", "name": event.name, "args": dict(event.args), "conn_id": conn_id}
    if isinstance(event, ClipReadyEvent):
        return {
            "type": "clip_ready",
            "session_id": event.session_id,
            "kind": event.kind,
            "start_marker": event.start_marker,
            "end_marker": event.end_marker,
            "now_marker": event.now_marker,
            "predicted_ready_at_ms": event.predicted_ready_at_ms,
            "playlist_url": event.playlist_url,
        }
    if isinstance(event, SessionMetricEvent):
        return {"type": "metric", "name": event.name, "value": event.value}
    if isinstance(event, ErrorEvent):
        return {"type": "error", "message": event.message}
    raise TypeError(f"unserialisable runner event: {type(event).__name__}")


def format_sse(seq: int, event: RunnerEvent) -> str:
    """Frame a runner event as one SSE message carrying its sequence number.

    Args:
        seq: The event's sequence number, emitted as the SSE ``id`` so a
            consumer can resume after it.
        event: The runner event to frame.

    Returns:
        A complete SSE message, terminated by the blank line that ends an event.
    """
    payload = json.dumps(runner_event_to_dict(event))
    return f"id: {seq}\ndata: {payload}\n\n"
